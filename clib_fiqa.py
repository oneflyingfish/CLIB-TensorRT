import torch
import pycuda.driver as cuda
from typing import List, Tuple, Dict
import os
import cv2
import numpy as np
from ytools.tensorrt import HostDeviceMem
import tensorrt as trt
from torchvision.transforms import transforms


def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx : min(ndx + bs, l)]


class CLIB_FIQA_Trt(object):
    """
    CLIB_FIQA can give a score of image

    Params
    ------
    - model_wts_path (optional, str) : path to mobilenetv2 model weights, defaults to the model file in ./mobilenetv2
    - max_batch_size (optional, int) : max batch size for embedder, only support 1 at this time
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    """

    def __init__(
        self,
        model_wts_path="model/deepsort_embedding_dynamic.engine",
        max_batch_size=1,
        bgr=True,
        log_level=0,
    ):
        self.log_level = log_level

        assert os.path.exists(
            model_wts_path
        ), f"Mobilenetv2 model path {model_wts_path} does not exists!"

        # init torch context
        a = torch.randn(size=(1, 1), device=torch.device("cuda:0"))
        del a

        cuda.init()
        self.gpu_id = 0
        self.gpu = cuda.Device(self.gpu_id)
        self.cuda_ctx = self.gpu.make_context()

        print(f"load model: {model_wts_path}")

        self.trt_engine = model_wts_path
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.trt_logger)
        with open(self.trt_engine, "rb") as f:
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())

        self.trt_context = self.trt_engine.create_execution_context()
        self.cuda_stream = cuda.Stream()

        self.torch_device = torch.device(f"cuda:{self.gpu_id}")
        self.torch_stream = torch.cuda.ExternalStream(
            self.cuda_stream.handle, device=self.torch_device
        )

        # Get the model inputs
        self.input_name = self.trt_engine.get_tensor_name(0)
        self.output_name = self.trt_engine.get_tensor_name(1)
        # set input shape
        self.max_input_shape = [
            int(dim)
            for dim in self.trt_engine.get_tensor_profile_shape(self.input_name, 0)[2]
        ]
        assert (
            max_batch_size <= self.max_input_shape[0]
        ), f"your batchsize is bigger than {self.max_input_shape[0]}, consider re-generate engine"

        if max_batch_size > 0:
            self.max_input_shape[0] = max_batch_size
        self.trt_context.set_input_shape(self.input_name, tuple(self.max_input_shape))
        # get I/O shape
        self.max_input_shape = [
            int(dim) for dim in self.trt_context.get_tensor_shape(self.input_name)
        ]
        self.max_output_shape = [
            int(dim) for dim in self.trt_context.get_tensor_shape(self.output_name)
        ]
        self.input_shape = None
        self.output_shape = None

        if self.log_level > 0:
            print(
                f'embedding input: "{self.input_name}", max shape: {self.max_input_shape}'
            )
            print(
                f'embedding output: "{self.output_name}", max_shape: {self.max_output_shape}'
            )

        self.input_mem = HostDeviceMem(
            self.max_input_shape, np.float32, self.cuda_stream
        )
        self.output_mem = HostDeviceMem(
            self.max_output_shape, np.float32, self.cuda_stream
        )

        self.set_input_shape(self.max_input_shape)

        self.input_width = self.max_input_shape[-1]
        self.input_height = self.max_input_shape[-2]

        self.max_batch_size = self.max_input_shape[0]
        self.input_bgr = bgr

        self.img_norm = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.warmup()

    def release(self):
        self.torch_stream.synchronize()
        self.cuda_stream.synchronize()
        del self.torch_stream
        del self.trt_context
        del self.trt_engine
        del self.trt_runtime
        self.cuda_ctx.detach()

    def set_input_shape(self, shape):
        shape = [int(dim) for dim in shape]
        if shape != self.input_shape:
            self.input_shape = shape
            self.trt_context.set_input_shape(self.input_name, tuple(self.input_shape))
            # get output shape
            self.output_shape = [
                int(dim) for dim in self.trt_context.get_tensor_shape(self.output_name)
            ]

            self.input_mem.set_shape(self.input_shape)
            self.output_mem.set_shape(self.output_shape)

            self.trt_context.set_tensor_address(
                self.input_name, int(self.input_mem.ptr())
            )
            self.trt_context.set_tensor_address(
                self.output_name, int(self.output_mem.ptr())
            )
            if self.log_level > 1:
                print(
                    f"reset input shape to input shape: {self.input_shape}, output shape: {self.output_shape}"
                )

    def warmup(self, times=3):
        for _ in range(times):
            if self.log_level > 1:
                print("warmup")
            self.inference(np.ndarray(self.input_shape, dtype=np.float32))

    def inference(
        self, img_datas: np.ndarray | torch.Tensor, require="np"
    ) -> np.ndarray | torch.Tensor:
        self.set_input_shape(img_datas.shape)

        if isinstance(img_datas, np.ndarray):
            self.input_mem.set_numpy(img_datas)
        else:
            self.input_mem.set_torch(img_datas)
        self.trt_context.execute_async_v3(
            stream_handle=self.cuda_stream.handle,
        )

        if require == "np":
            return [self.output_mem.read_numpy()]
        else:
            return [self.output_mem.read_torch()]

    def preprocess(self, np_images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        """
        input_imags = []
        for np_image in np_images:
            input_imags.append(
                cv2.resize(np_image, (self.input_width, self.input_height))
            )
        input_imags = np.stack(input_imags)

        with torch.cuda.stream(self.torch_stream):
            input_imags = torch.from_numpy(input_imags).cuda(non_blocking=False)
            self.torch_stream.synchronize()
            self.cuda_stream.synchronize()
            if self.input_bgr:
                input_imags = torch.flip(input_imags, dims=[-1])
            input_imags = (input_imags / 255.0).moveaxis(-1, 1)
            tensors = self.img_norm(input_imags)  # type: torch.Tensor
        return batch(tensors.contiguous(), self.max_batch_size)

    def push_ctx(self):
        self.cuda_ctx.push()

    def pop_ctx(self):
        self.cuda_ctx.pop()

    def predict(self, np_images) -> np.ndarray:
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr, Batch size is N

        Returns
        ------
        np.ndarray, shape is (N,1), and item is float32 means score of each image

        """
        all_feats = []  # type: List[np.ndarray]
        batchs = self.preprocess(np_images)
        for batch in batchs:
            all_feats.extend(self.inference(batch)[0])
        return np.stack(all_feats)
