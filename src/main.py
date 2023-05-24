import os
import random
import wandb
import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from beautifultable import BeautifulTable

import config

from train import train
from test import test
from inference import inference_test
from utils.utils import *


# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def model_pipeline(do_train=True, do_test=True, do_inference=True, 
                   run_name="main_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) -> nn.Module:
    
    # make the model, data, and optimization problem
    model, train_loader, test_loader, criterion, optimizer, scheduler = make()

    if do_train:
        # and use them to train the model
        train(model, train_loader, criterion, optimizer, scheduler, run_name)

    if do_test:
        # and test its final performance
        test(model, test_loader, run_name)
        
    if do_inference:
        # and test its final performance
        inference_test()

    return model


def main():
    
    print("\n# --------------------------------------------------")
    print("| Starting Test and Test of the moodel...")
    print("# --------------------------------------------------\n")
    
    table = BeautifulTable()
    table.column_headers = ["Parameter", "Value"]
    table.append_row(["Device", config.device])
    table.append_row(["Number of epochs", config.epochs])
    table.append_row(["Batch size", config.batch_size])
    table.append_row(["Learning rate", config.lr])
    table.append_row(["Image size", config.image_size])
    table.append_row(["Number of classes", config.num_classes])
    table.append_row(["Number of channels", config.channels])
    table.append_row(["Dimension", config.dim])
    table.append_row(["Depth", config.depth])
    table.append_row(["Number of heads", config.heads])
    table.append_row(["MLP dimension", config.mlp_dim])
    print(table)
    
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
        pretained_model=config.pretrained_model,
        text_model=config.text_model,
        device=config.device,
        data_augmentation=config.data_augmentation,
    )
    
    wandb.init(project="bussiness_uab", 
               notes=None,
               name=run_name,
               config=cfg)
    
    os.makedirs(os.path.join("./results", run_name), exist_ok=True)
    
    with open(os.path.join("./results", run_name, "config.txt"), "w") as f:
        for key, value in cfg.items():
            f.write("{}: {}\n".format(key, value))

    model_pipeline(run_name=run_name)

    time_end = time.time()
    print("Total execution time:", time_end - time_start)

    wandb.finish()


if __name__ == "__main__":
    main()
