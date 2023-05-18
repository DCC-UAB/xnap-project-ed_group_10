from tqdm.auto import tqdm
import wandb

import time,os,json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from einops import rearrange

from  ... import config
from data.conTextDataset import conTextDataset
from models.conTextTransformer import conTextTransformer


def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data_img, data_txt, txt_mask, target) in enumerate(data_loader):
        data_img = data_img.to(config.device)
        data_txt = data_txt.to(config.device)
        txt_mask = txt_mask.to(config.device)
        target = target.to(config.device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data_img, data_txt, txt_mask), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data_img)) + '/' + '{:5}'.format(total_samples) +
                 ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())

    
def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data_img, data_txt, txt_mask, target in data_loader:
            data_img = data_img.to(config.device)
            data_txt = data_txt.to(config.device)
            txt_mask = txt_mask.to(config.device)
            target = target.to(config.device)
            output = F.log_softmax(model(data_img, data_txt, txt_mask), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
        '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    return correct_samples / total_samples


def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
    images, labels = images.to(config.device), labels.to(config.device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    

if __name__ == "__main__":
    
    json_file = '/datatmp/datasets/ConText/annotations/split_0.json'
    img_dir = "../data/data_images/data/JPEGImages"
    txt_dir = "/datatmp/datasets/ConText/ocr_labels/"
    input_size = 256
    data_transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data_transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    train_set = conTextDataset(json_file, img_dir, txt_dir, True, data_transforms_train)
    test_set  = conTextDataset(json_file, img_dir, txt_dir, False, data_transforms_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)
        
        
    N_EPOCHS = 50
    start_time = time.time()

    model = conTextTransformer(image_size=input_size, num_classes=28, channels=3, dim=256, depth=2, heads=4, mlp_dim=512)
    model.to(config.device)
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.Adam(params_to_update, lr=0.0001)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)

    train_loss_history, test_loss_history = [], []
    best_acc = 0.

    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        acc = evaluate(model, test_loader, test_loss_history)
        if acc>best_acc: torch.save(model.state_dict(), '../results/all_best_params.pth')
        scheduler.step()

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
