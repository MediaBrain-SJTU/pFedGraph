# pFedGraph algorithm on mindspore

## Introduction

This code transfers pFedGraph algorithm from torch version to mindspore version and verifies that it also works well on mindspore.

## Environment

We use mindspore in the environment as follows.

```python
asttokens          3.0.1
astunparse         1.6.3
certifi            2025.10.5
cffi               2.0.0
charset-normalizer 3.4.4
clarabel           0.11.1
contourpy          1.3.3
cvxpy              1.7.3
cycler             0.12.1
dill               0.4.0
download           0.3.5
fonttools          4.60.1
idna               3.11
Jinja2             3.1.6
joblib             1.5.2
kiwisolver         1.4.9
MarkupSafe         3.0.3
matplotlib         3.10.7
mindspore-dev      2.6.0.dev20250323
numpy              1.26.4
osqp               1.0.5
packaging          25.0
pillow             12.0.0
pip                25.3
protobuf           6.33.1
psutil             7.1.3
pycparser          2.23
pyparsing          3.2.5
python-dateutil    2.9.0.post0
requests           2.32.5
safetensors        0.7.0
scikit-learn       1.7.2
scipy              1.16.3
scs                3.2.9
setuptools         80.9.0
six                1.17.0
threadpoolctl      3.6.0
tqdm               4.67.1
urllib3            2.5.0
wheel              0.45.1
```

We strongly recommend using mindspore version 2.0 instead of other versions, which may trigger unknown error.

To run mindspore successfully, you may need to add command

```python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:{YOUR_ENVIRONMENT_PATH}/lib
```

before running mindspore.

## Run this code

To run this code, you can directly use

```console
cd tensorflow
bash run_cifar10.sh
```