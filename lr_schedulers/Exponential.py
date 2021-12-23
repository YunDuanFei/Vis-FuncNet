import torch.optim as optim
import numpy as np
from . import config

def lambda_loffe(epoch, alph=config.expoloffe_alph):
    return np.exp(-alph*epoch)

def lambda_raffel(epoch, T_mid=config.exporaffel_T):
    return np.exp(1/np.sqrt(max(epoch, T_mid))-1/np.sqrt(T_mid))

# ExpoloffeLR
def expoloffe_lr(optimizer):
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_loffe)
    return scheduler

# ExporaffelLR
def exporaffel_lr(optimizer):
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_raffel)
    return scheduler