# CLIB-TensorRT
speedup CLIB-FIQA by TensorRT，only focus only on image quality.

# usage

```bash
git clone --recurse-submodules https://github.com/oneflyingfish/CLIB-TensorRT.git
cd CLIB-TensorRT
python3 main.py
```

# generate engine
> only do this if you need to quant by yourself to `generate model/CLIB-FIQA_R50_decode_e2e.engine`

```bash
cd CLIB-TensorRT
wget https://github.com/oneflyingfish/CLIB-TensorRT/releases/download/v0.1.0/dataset.zip
unzip dataset.zip
python3 export.py
python3 quant.py
```
