import torch
from torchvision.transforms import transforms
import numpy as np
import cv2
import os
from ytools.tensorrt import CalibratorDatasetObject


class CalibratorDataset(CalibratorDatasetObject):
    def __init__(
        self,
        calibration_image_folder,
        input_shape=(-1, 3, 224, 224),
        batch_size=1,
        skip_frame=1,
        dataset_limit=1 * 1000,
    ):
        self.image_folder = calibration_image_folder

        self.preprocess_flag = True
        self.datasets = None

        self.dataset_limit = dataset_limit
        self.skip_frame = skip_frame

        (_, _, self.height, self.width) = input_shape
        self.batch_size = batch_size
        self.norm = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.init_data()

    def init_data(self):
        self.preprocess_flag = False
        imgs = self.load_pre_data(
            self.image_folder, size_limit=self.dataset_limit, skip=self.skip_frame
        )  # (k*b+m,c,h,w)

        self.datasets = [
            [np.ascontiguousarray(item)]
            for item in np.split(
                imgs[: len(imgs) // self.batch_size * self.batch_size, ...],
                len(imgs) // self.batch_size,
                axis=0,
            )
        ]

        print(
            f"finish init calibration in cpu: datasize={len(self)}*{self.shape(0)}, type={self.dtype(0)}"
        )

    def preprocess(self, np_image: np.ndarray, bgr=True):
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
        if bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image

        input_image = cv2.resize(np_image_rgb, (self.width, self.height))
        # trans = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        input_image = self.norm(
            (torch.from_numpy(input_image) / 255.0).moveaxis(-1, 0)
        )  # type: torch.Tensor
        # input_image = input_image.view(
        #     1, 3, self.height, self.width
        # )  # type:torch.Tensor

        return input_image.cpu().numpy()

    def load_pre_data(self, imgs_folder, size_limit=0, skip=20):
        imgs_path = []
        imgs = []
        with open(os.path.join(imgs_folder, "data.txt"), "r") as fp:
            imgs_path = [s.replace("\n", "") for s in fp.readlines() if len(s) > 1]

        idx = -1
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            idx += 1

            if idx % skip == 0:
                imgs.append(self.preprocess(img))
                idx = 0

            if size_limit > 0 and len(imgs) >= size_limit:
                break

        assert len(imgs) > 0, "empty datas"

        return np.stack(imgs)
