from clib_fiqa import CLIB_FIQA_Trt
import cv2
from CLIB_FIQA.model import clip
from CLIB_FIQA.utilities import *
from CLIB_FIQA.inference import backboneSet, img_tensor
from itertools import product
import numpy as np
import onnxruntime as ort
import torch.nn.functional as F
import torch
from ytools.bench import test_time
import torchvision.transforms as T
import types
import math
import time

# define raw
quality_list = ["bad", "poor", "fair", "good", "perfect"]
blur_list = ["hazy", "blurry", "clear"]
occ_list = ["obstructed", "unobstructed"]
pose_list = ["profile", "slight angle", "frontal"]
exp_list = ["exaggerated expression", "typical expression"]
ill_list = ["extreme lighting", "normal lighting"]
joint_texts = torch.cat(
    [
        clip.tokenize(
            f"a photo of a {b}, {o}, and {p} face with {e} under {l}, which is of {q} quality"
        )
        for b, o, p, e, l, q in product(
            blur_list, occ_list, pose_list, exp_list, ill_list, quality_list
        )
    ]
).cuda()
clip_model = "model/weights/RN50.pt"
clip_weights = "model/weights/CLIB-FIQA_R50.pth"
raw_model = backboneSet(clip_model)
raw_model = load_net_param(raw_model, clip_weights)


@torch.no_grad()
def post(output: torch.Tensor) -> torch.Tensor:
    logits_per_image = F.softmax(output, dim=1).reshape(output.size(0), -1, 5).sum(1)
    anchor_bins = (
        torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).cuda().unsqueeze(0).transpose(0, 1)
    )
    one_scores = (logits_per_image.to(torch.float32) @ anchor_bins).squeeze(1)
    return one_scores


@torch.no_grad()
def raw_infer(self, img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize([224, 224]),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    img_tensor = transform(img)
    data = img_tensor.unsqueeze(dim=0).cuda()
    output_raw, _ = self.forward(data, joint_texts)
    return post(output_raw).cpu().numpy()


setattr(raw_model, "predict", types.MethodType(raw_infer, raw_model))

# define opt version
model_trt = CLIB_FIQA_Trt("model/CLIB-FIQA_R50_decode_e2e.engine", bgr=True)


# eval
def get_eval_data():
    imgs_path = []
    imgs = []
    paths = []
    with open(os.path.join("dataset/", "data.txt"), "r") as fp:
        imgs_path = [s.replace("\n", "") for s in fp.readlines() if len(s) > 1]

    idx = -1
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        if img is not None:
            imgs.append(img)
            paths.append(img_path)
    return imgs, paths


@test_time()
@torch.no_grad()
def run_raw(imgs):
    scores = []
    for img in imgs:
        scores.append(raw_model.predict(img).item(0))
    return scores


@test_time()
@torch.no_grad()
def run_opt(imgs):
    scores = []
    for img in imgs:
        scores.append(model_trt.predict([img])[0].item(0))
    return scores


imgs, img_paths = get_eval_data()
s1 = time.time()
score_opts = np.array(run_opt(imgs))
s2 = time.time()
score_raws = np.array(run_raw(imgs))
s3 = time.time()

losses = np.abs(score_opts - score_raws)
losses_percentage = losses / score_raws
print(
    f"speedup x{(s3-s2)/(s2-s1):.2f}, avg loss percentage: {losses_percentage.mean().item(0)*100:.1f}%, max loss percentage: {losses_percentage.max().item(0)*100:.1f}%, abs mean loss: {np.mean(losses).item(0):.3f}, abs max loss: {np.max(losses).item(0):.3f}"
)

bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
hist, bin_edges = np.histogram(losses, bins=bins)
freq = hist / len(losses)
for i in range(len(bins) - 1):
    print(f"abs loss probability in [{bins[i]}, {bins[i+1]}]: {freq[i]*100:.3f}%")


print("-------------------------------")

imgs = []
for i in range(1, 7):
    imgs.append(cv2.imread(f"CLIB_FIQA/samples/{i}.jpg"))
score_opts = np.array(run_opt(imgs))
score_raws = np.array(run_raw(imgs))

losses = np.abs(score_opts - score_raws)
losses_percentage = losses / score_raws
print(f"raw score: {score_raws}")
print(f"opt score: {score_opts}")
print(f"loss percentage: {losses_percentage}")
print(f"abs loss: {losses}")
model_trt.release()
