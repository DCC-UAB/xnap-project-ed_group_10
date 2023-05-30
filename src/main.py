import os
import random
import wandb
import datetime
import numpy as np
import torch
import torch.nn as nn
import time

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
    model, train_loader, test_loader, val_loader, criterion, optimizer, scheduler = make()

    if do_train:
        # and use them to train the model
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, run_name)

    if do_test:
        # and test its final performance
        test(test_loader, run_name=run_name)
        
    if do_inference:
        # and test its final performance
        inference_test()

    return model


def main():
    
    
    print("\n# --------------------------------------------------")
    print("| Starting Test and Test of the moodel...")
    print("| Device:", config.device)
    print("| Number of epochs:", config.epochs)
    print("| Batch size:", config.batch_size)
    print("| Learning rate:", config.lr)
    print("| Image size:", config.image_size)
    print("| Number of classes:", config.num_classes)
    print("| Number of channels:", config.channels)
    print("| Dimension:", config.dim)
    print("| Depth:", config.depth)
    print("| Number of heads:", config.heads)
    print("| MLP dimension:", config.mlp_dim)
    print("| Dataset: Business Dataset")
    print("| Architecture: Context Transformer")
    print("| Pretained_model:", config.pretrained_model)
    print("| Text_model:", config.text_model)
    print("# --------------------------------------------------\n")
    
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
        scheduler=config.scheduler,
        optimizer=config.optimizer,
        dataset="Business Dataset",
        architecture="Context Transformer",
        pretained_model=config.pretrained_model,
        text_model=config.text_model,
        device=config.device,
    )
    
    wandb.init(project="bussiness_uab", 
               notes=None,
               name=run_name,
               config=cfg)
    
    os.makedirs("./results/" + run_name, exist_ok=True)
    
    with open("./results/" + run_name + "/config.txt", "a") as f:
        f.write("Date: {}\n".format(datetime.datetime.now().strftime("%Y %m %d, %H %M %S")))
        for key, value in cfg.items():
            f.write("{}: {}\n".format(key, value))

    model_pipeline(run_name=run_name)

    time_end = time.time()
    print("Total execution time:", time_end - time_start)

    wandb.finish()


if __name__ == "__main__":
    main()
