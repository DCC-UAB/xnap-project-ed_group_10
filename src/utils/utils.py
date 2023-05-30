import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.convNet import *
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from einops import rearrange
import fasttext
import fasttext.util

from models.conTextTransformer import ConTextTransformer
from data.dataloader import *
import config


def make():
    
    # Make the data
    train_loader, test_loader, val_loader = Dataloader().get_loaders()

    # Make the model
    model = ConTextTransformer(
        image_size=config.image_size,
        num_classes=config.num_classes,
        channels=config.channels,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads, 
        mlp_dim=config.mlp_dim
    ).to(config.device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    params_to_update = []
    for _, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(params_to_update, lr=config.lr)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_to_update, lr=config.lr)
    
    # The scheduler will update the learning rate after every epoch to achieve a better convergence
    if config.scheduler == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=config.gamma)
    elif config.scheduler == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.05, verbose=True)

        
    return model, train_loader, test_loader, val_loader, criterion, optimizer, scheduler


def context_inference(model, img_filename, OCR_tokens):
    
    fasttext_model = fasttext.load_model('cc.en.300.bin')
    
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_filename).convert('RGB')
    img = img_transforms(img)
    img = torch.unsqueeze(img, 0)
    
    text = np.zeros((1, 64, 300))
    for i,w in enumerate(OCR_tokens):
        text[0,i,:] = fasttext_model.get_word_vector(w)

    output = F.softmax(model(img.to(config.device), torch.tensor(text).to(config.device)), dim=1)
    return output.cpu().detach().numpy()
