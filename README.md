# CLIB-TensorRT
speedup CLIB-FIQA by TensorRT，only focus only on image quality.

# usage

```bash
git clone --recurse-submodules https://github.com/oneflyingfish/CLIB-TensorRT.git
cd CLIB-TensorRT

# if you have no model/CLIB-FIQA_R50_decode_e2e.engine, view # generate engine learn how to generate
python3 main.py
```

# generate engine
> only do this if you need to quant by yourself to `generate model/CLIB-FIQA_R50_decode_e2e.engine`

```bash
cd CLIB-TensorRT
wget https://github.com/oneflyingfish/CLIB-TensorRT/releases/download/v1.0.0/dataset.zip
wget https://github.com/oneflyingfish/CLIB-TensorRT/releases/download/v1.0.0/model.zip
unzip dataset.zip
unzip model.zip
python3 export.py
python3 quant.py
```
