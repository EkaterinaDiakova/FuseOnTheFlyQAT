
import os
import torch
import torch.nn as nn
from Preprocess import datapool
from utils import train, val, seed_all, get_logger

from NonNormedModel import get_norm_model
from QAT_utils import FuseOnTheFlyQAT, train_fuse_ste_qat


get_ipython().system('pip uninstall opencv-python opencv-contrib-python')
get_ipython().system('pip install opencv-python-headless')


args = {
    'workers': 4,
    'batch_size': 200,
    'seed': 42,
    'suffix': '',
    'dataset': 'cifar10',
    'model': 'resnet20',
    'identifier': 'resnet20_L[16]_normed_on_th_added_act',  
    'device': '0',
    'time': 0
}


train_loader, val_loader = datapool(args['dataset'], args['batch_size'])

model_norm_pooling = get_norm_model(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args['device']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_norm_pooling.to(device)
model_norm_pooling.set_T(0)
model_norm_pooling.set_L(16)

acc = val(model_norm_pooling, val_loader, device, 0)
print(f"Validation accuracy: {acc}")

model_qat = FuseOnTheFlyQAT(model_norm_pooling)
model_qat_trained = train_fuse_ste_qat(model_qat, train_loader, val_loader, device, epochs=1)

