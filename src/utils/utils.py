import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.convNet import *
from PIL import Image
import numpy as np
import easyocr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from einops import rearrange
import fasttext
import fasttext.util


def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


def context_inference(img_filename, OCR_tokens):
    img = Image.open(img_filename).convert('RGB')
    img = img_transforms(img)
    img = torch.unsqueeze(img, 0)
    
    text = np.zeros((1, 64, 300))
    for i,w in enumerate(OCR_tokens):
        text[0,i,:] = fasttext_model.get_word_vector(w)

    output = F.softmax(model(img.to(device), torch.tensor(text).to(device)), dim=1)
    return output.cpu().detach().numpy()