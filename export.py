from CLIB_FIQA.model import clip
from CLIB_FIQA.utilities import *
from CLIB_FIQA.inference import backboneSet, img_tensor
from itertools import product
import numpy as np
import onnxruntime as ort
import torch.nn.functional as F
import torch
from ytools.bench import test_time


@test_time()
def post(output: torch.Tensor) -> torch.Tensor:
    logits_per_image = F.softmax(output, dim=1).reshape(output.size(0), -1, 5).sum(1)
    anchor_bins = (
        torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).cuda().unsqueeze(0).transpose(0, 1)
    )
    one_scores = (logits_per_image.to(torch.float32) @ anchor_bins).squeeze(1)
    return one_scores


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
model = backboneSet(clip_model)
model = load_net_param(model, clip_weights)

print(joint_texts.shape)

dummy_image = img_tensor("CLIB_FIQA/samples/1.jpg").cuda()  # type:torch.Tensor
a, _ = model.forward(dummy_image, joint_texts)
# print(a[..., :10])

# model.forward = model.forward_text
# torch.onnx.export(
#     model,
#     (joint_texts),
#     "model/CLIB-FIQA_R50_encode.onnx",
#     export_params=True,
#     opset_version=14,
#     do_constant_folding=True,
#     input_names=["text"],
#     output_names=["text_feature"],
# )

joint_features = model.forward_text(joint_texts)  # type:torch.Tensor

model.forward = model.forward_all_features
torch_out = model.forward(dummy_image, joint_features).detach()
# torch.onnx.export(
#     model,
#     (dummy_image, joint_features),
#     "model/CLIB-FIQA_R50_decode.onnx",
#     export_params=True,
#     opset_version=14,
#     do_constant_folding=True,
#     input_names=["image", "text_feature"],
#     output_names=["output"],
# )

model.forward = model.forward_all_features_e2e
torch.onnx.export(
    model,
    (dummy_image, joint_features),
    "model/CLIB-FIQA_R50_decode_e2e_raw.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["image", "text_feature"],
    output_names=["scores"],
)

# joint_features.detach().cpu().numpy().dump("model/features.npy")


# validate
onnxruntime_backends = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
session = ort.InferenceSession(
    "model/CLIB-FIQA_R50_decode_e2e_raw.onnx",
    providers=onnxruntime_backends,
)


output = session.run(
    ["scores"],
    {
        "image": dummy_image.cpu().numpy(),
        "text_feature": joint_features.detach().cpu().numpy(),
    },
)[0]

scores = torch.from_numpy(output).cuda()
torch.cuda.synchronize()
print("onnx raw:", scores)
print("torch:", post(torch_out))

# modify text_feature to initializer

import onnx
from onnx import numpy_helper

onnx_model = onnx.load("model/CLIB-FIQA_R50_decode_e2e_raw.onnx")
text_feature_tensor = numpy_helper.from_array(
    joint_features.detach().cpu().numpy(), name="text_feature"
)
onnx_model.graph.initializer.append(text_feature_tensor)
for i, input_node in enumerate(onnx_model.graph.input):
    if input_node.name == "text_feature":
        onnx_model.graph.input.pop(i)
        break
onnx.save(onnx_model, "model/CLIB-FIQA_R50_decode_e2e.onnx")

session = ort.InferenceSession(
    "model/CLIB-FIQA_R50_decode_e2e.onnx",
    providers=onnxruntime_backends,
)


output = session.run(
    ["scores"],
    {
        "image": dummy_image.cpu().numpy(),
    },
)[0]
scores = torch.from_numpy(output).cuda()
print("onnx:", scores)
