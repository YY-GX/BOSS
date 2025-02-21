# Installation
```shell
conda create -n libero python=3.10.0
# To ensure pip installation location is same as conda
conda install --force-reinstall pip

# Libero related packages
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install robosuite
cd ./YourLocation/Libero && pip install -e .

# CV related packages
# sam2 only support python >= 3.10.0
pip install sam2
pip install groundingdino-py
pip install wandb h5py timm dds_cloudapi_sdk scikit-image
pip install numpy==1.23.1
```

```shell
pip install --force-reinstall 
```