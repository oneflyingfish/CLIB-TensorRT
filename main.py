from clib_fiqa import CLIB_FIQA_Trt
import cv2

model_trt = CLIB_FIQA_Trt("model/CLIB-FIQA_R50_decode_e2e.engine", bgr=True)

for i in range(7):
    img = cv2.imread(f"CLIB_FIQA/samples/{i}.jpg")
    if img is None:
        continue
    print("score:", model_trt.predict([img])[0])
model_trt.release()
