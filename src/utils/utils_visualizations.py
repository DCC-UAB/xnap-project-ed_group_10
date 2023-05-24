import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.convNet import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_batch(batch):
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels:', labels)
    
    
def show_batch_with_predictions(batch, model, device="cuda"):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels:', labels)
    print('predictions:', predicted)
    
    
def show_batch_with_predictions_and_confidence(batch, model, device="cuda"):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    confidence = torch.nn.functional.softmax(outputs, dim=1)
    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels:', labels)
    print('predictions:', predicted)
    print('confidence:', confidence)
    
    
def make_loss_plot(loss_history, path):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.savefig(path)
    plt.close()
    
    
def make_accuracy_plot(accuracy_history, path):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_history)
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy')
    plt.savefig(path)
    plt.close()
    
    
def show_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()
    
    
def show_image_with_prediction(image, model, device="cuda"):
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print('prediction:', predicted)
    plt.figure(figsize=(10, 10))
    plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
    plt.show()
    
    
def show_image_with_prediction_and_confidence(image, model, device="cuda"):
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    confidence = torch.nn.functional.softmax(output, dim=1)
    print('prediction:', predicted)
    print('confidence:', confidence)
    plt.figure(figsize=(10, 10))
    plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
    plt.show()
    
    
def show_image_with_prediction_and_confidence_and_label(image, model, device="cuda"):
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    confidence = torch.nn.functional.softmax(output, dim=1)
    print('prediction:', predicted)
    print('confidence:', confidence)
    plt.figure(figsize=(10, 10))
    plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
    plt.show()
