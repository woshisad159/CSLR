# ----------------------------------------
# Written by Jing Li
# ----------------------------------------
from Train import train
from ReadConfig import readConfig
import random
import os
import numpy as np
import torch
import torch.backends.cudnn

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def main():
    # Read configuration file
    configParams = readConfig()
    # If isTrain is true, it is the training mode; if isTrain is false, it is the verification mode
    train(configParams, isTrain=False)

if __name__ == '__main__':
    seed_torch(10)
    main()

