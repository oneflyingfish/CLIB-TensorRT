from ytools.tensorrt import save_engine,MyEntropyCalibrator
from calibrator_dataset import CalibratorDataset

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

input_model_path = "model/CLIB-FIQA_R50_decode_e2e.onnx"
output_model_path = "model/CLIB-FIQA_R50_decode_e2e.engine"

calibration_dataset_path = "dataset/"  # some *.mp4 videos in fold that can run yolo , need no special name and ratio
data_set = CalibratorDataset(
    calibration_dataset_path,
    input_shape=(1, 3, 224, 224),
    batch_size=1,
    skip_frame=1,
    dataset_limit=1 * 1000,
)
calibrator = MyEntropyCalibrator(
    data_loader=data_set, cache_file="model/CLIB-FIQA_R50_decode_e2e.cache"
)

save_engine(
    input_model_path,
    output_model_path,
    fp16_mode=True,
    int8_mode=True,
    min_batch=1,
    optimize_batch=1,
    max_batch=1,
    calibrator=calibrator,
)