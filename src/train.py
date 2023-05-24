import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import sys
import pickle as pkl
from torchviz import make_dot

import config


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    # This part of the code will use the log_softmax function to compute the log of the softmax function
    # and then use the nll_loss function to compute the negative log likelihood loss.
    # This is done because the log_softmax function is numerically more stable than the softmax function.
    
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
        
        # Print loss every 100 batches
        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data_img)) + '/' + '{:5}'.format(total_samples) +
                ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
    return i

    
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
    print('\nTRAIN - Average train loss: ' + '{:.4f}'.format(avg_loss) +
        '  Accuracy:' + ' {:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    return correct_samples / total_samples, avg_loss


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


def train_log(acc, example_ct, epoch, loss, lr):
    # Where the magic happens
    wandb.log({"epoch": epoch, "train_accuracy": acc, "loss": loss, "lr":lr}, step=example_ct)
    print(f"\nTRAIN - Accuracy after {str(example_ct).zfill(5)} examples: {acc:.3f}\n")
    

def train(model, train_loader, criterion, optimizer, scheduler):
    
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Keep track of loss and accuracy
    train_loss_history, test_loss_history = [], []
    best_acc = 0.
    example_ct = 0  # number of examples seen
    
    # Train the model that has the best hyperparameters
    for epoch in range(1, config.epochs + 1):
        
        print('\nEpoch:', epoch)
        iter_in_epoch = train_epoch(model, optimizer, train_loader, train_loss_history)
        acc, loss = evaluate(model, train_loader, test_loss_history)
        
        example_ct += len(train_loader.dataset)
        
        # Save the model with the best accuracy
        if acc>best_acc: 
            torch.save(model.state_dict(), './src/models/all_best_params.pth')
            wandb.save("all_best_params.pth")
            
            # # Exportar un plot de la estructura de la red
            
            # model.named_parameters()))
            # vis_graph.format = 'png'
            # vis_graph.directory = './src/models'
            # # vis_graph.view()
            
            best_acc = acc
            
        scheduler.step()
        
        train_log(acc, example_ct, epoch, loss, optimizer.param_groups[0]['lr'])
