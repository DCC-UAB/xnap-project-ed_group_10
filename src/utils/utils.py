import wandb
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

from models.conTextTransformer import conTextTransformer
from data.dataloader import *


def make_loader(train=True):
    
    train = "train" if train else "test"
    loader = Dataloader().get_loaders(train)    
    return loader


def make(config):
    
    # Make the data
    train_loader = make_loader(train=True)
    test_loader = make_loader(train=False)

    # Make the model
    model = conTextTransformer(
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
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.Adam(params_to_update, lr=config.lr)
        
    return model, train_loader, test_loader, criterion, optimizer


def context_inference(img_filename, OCR_tokens):
    
    fasttext.util.download_model('en', if_exists='ignore')  # English
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

    output = F.softmax(model(img.to(device), torch.tensor(text).to(device)), dim=1)
    return output.cpu().detach().numpy()
