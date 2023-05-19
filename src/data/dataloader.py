# + --------------------------------------
# | Description:
# |  This file contains the dataloader class
# |  for the ConText dataset.
# + --------------------------------------

import os
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

import config
from data.conTextDataset import conTextDataset


class Dataloader():
    def __init__(self, train_dir, test_dir, batch_size, num_workers):

        # data_transforms_train --> This is the data augmentation and normalization we will use for the training data
        # data_transforms_test  --> This is the normalization we will use for the validation data
        
        self.data_transforms_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(config.image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                                 [0.229, 0.224, 0.225])
            ])
        
        self.data_transforms_test = torchvision.transforms.Compose([
                torchvision.transforms.Resize(config.image_size),
                torchvision.transforms.CenterCrop(config.image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                                 [0.229, 0.224, 0.225])
            ])
        
    def get_loaders(self):
        
        # Create the dataset and dataloader
        
        train_set = conTextDataset(config.json_file, config.img_dir, config.txt_dir, True, self.data_transforms_train)
        test_set  = conTextDataset(config.json_file, config.img_dir, config.txt_dir, False, self.data_transforms_test)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, config.num_workers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, config.num_workers)
        
        return train_loader, test_loader