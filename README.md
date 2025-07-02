# CLIB-TensorRT
speedup CLIB-FIQA by TensorRT，only focus only on image quality.

# usage

```bash
cd CLIB-TensorRT
git submodule init
git submodule update
# if you have no model/CLIB-FIQA_R50_decode_e2e.engine, view # generate engine learn how to generate
python3 main.py
```

# generate engine
> only do this if you need to quant by yourself to `generate model/CLIB-FIQA_R50_decode_e2e.engine`

```bash
python3 export.py
python3 quant.py        # you need to set `calibration_dataset_path` to your calibration set path
```
