# IOH
IOH prediction project


## Installation
```bash
conda create -n ioh python=3.9
conda activate ioh
cd path/to/IOH
pip install -r requirements.txt
```
For pytorch-forecasting: manually go to "path/to/anaconda3/envs/ioh/lib/python3.9/site-packages/pytorch_forecasting/data/encoders.py" and
replace all appearances of "np.float" to "float". Note: replace whole word matches only.
This workaround is needed because of a version mismatch between
pytorch_forecasting and numpy. Solution from [here](https://github.com/jdb78/pytorch-forecasting/issues/1236).

Afterwards:
```bash
pip install numpy==1.26.4
```

For pytorch-forecasting: manually go to "path/to/anaconda3/envs/ioh/lib/python3.9/site-packages/pyemd" and change it into "PyEMD".

