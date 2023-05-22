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
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def model_pipeline(cfg:dict, do_train=True, do_test=True) -> nn.Module:
    
    # tell wandb to get started
    with wandb.init(project="bussiness_uab", config=cfg):
      
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        
        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)

        if do_train:
            # and use them to train the model
            train(model, train_loader, criterion, optimizer, config)

        if do_test:
            # and test its final performance
            test(model, test_loader)

    return model


if __name__ == "__main__":
    
    run_name = "main_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="bussiness_uab", 
               name=run_name)
    
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
        
    model = model_pipeline(cfg)

    time_end = time.time()
    print("Total execution time:", time_end - time_start)

    wandb.finish()
