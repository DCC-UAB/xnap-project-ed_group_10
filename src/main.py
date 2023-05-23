import os
import random
import wandb

import config

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

from utils.train_best_params import *
from test import *
from utils.utils import *

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def model_pipeline(do_train=True, do_test=True) -> nn.Module:
    
    # make the model, data, and optimization problem
    model, train_loader, test_loader, criterion, optimizer, scheduler = make(config)

    if do_train:
        # and use them to train the model
        train(model, train_loader, criterion, optimizer, scheduler)

    if do_test:
        # and test its final performance
        test(model, test_loader)

    return model

def main(train_best_params=False):
    
    # + ---------------------- 
    # | Get the best params (activated only once)
    # + ----------------------
    
    if train_best_params:
        main_train_best_params()
    
    # + ----------------------
    # | Train and Test the model
    # + ----------------------
    
    run_name = "main_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    time_start = time.time()

    cfg = dict(
        epochs=config.epochs,
        classes=config.num_classes,
        batch_size=config.batch_size,
        learning_rate=config.lr,
        dataset="Business Dataset",
        architecture="Context Transformer",
        device=config.device
    )
    
    wandb.init(project="bussiness_uab", 
               name=run_name,
               config=cfg)
            
    model = model_pipeline()

    time_end = time.time()
    print("Total execution time:", time_end - time_start)

    wandb.finish()
    
    

if __name__ == "__main__":
    
    main(train_best_params=False)

