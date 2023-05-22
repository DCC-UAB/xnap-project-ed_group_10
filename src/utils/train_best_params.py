import wandb
import time,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import datetime

# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(parent_dir))

from config import *
from models.conTextTransformer import ConTextTransformer
from data.dataloader import *


def train(model, train_loader, criterion, optimizer, scheduler):
    
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Keep track of loss and accuracy
    train_loss_history, test_loss_history = [], []
    best_acc = 0.
    example_ct = 0  # number of examples seen
    
    # Train the model that has the best hyperparameters
    for epoch in range(1, config.num_epochs + 1):
        
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        acc = evaluate(model, test_loader, test_loss_history)
        
        example_ct += len(train_loader.dataset)
        
        # Save the model with the best accuracy
        if acc>best_acc: torch.save(model.state_dict(), './all_best_params.pth')
        scheduler.step()
        
        # Log metrics to visualize performance        
        train_log(acc, example_ct, epoch)


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


def train_log(acc, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "acc": acc}, step=example_ct)
    print(f"Accuracy after {str(example_ct).zfill(5)} examples: {acc:.3f}")
    

if __name__ == "__main__":
    
    run_name = "train_best_params_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="bussiness_uab", 
               name=run_name)
    
    print("# --------------------------------------------------")
    print("Starting training best parameters for the model...")
    print("Device:", config.device)
    print("Number of epochs:", config.num_epochs)
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.lr)
    print("Image size:", config.image_size)
    print("Number of classes:", config.num_classes)
    print("Number of channels:", config.channels)
    print("Dimension:", config.dim)
    print("Depth:", config.depth)
    print("Number of heads:", config.heads)
    print("MLP dimension:", config.mlp_dim)
    print("# --------------------------------------------------")
    
    train_loader, test_loader = Dataloader().get_loaders()
    start_time = time.time()
    
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
        
    # Update only the parameters where requires_grad is True
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    optimizer = torch.optim.Adam(params_to_update, lr=config.lr)
    
    # The scheduler will update the learning rate after every epoch to achieve a better convergence
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=config.gamma)
    
    criterion = nn.CrossEntropyLoss()


    train(model, train_loader, criterion, optimizer, scheduler)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    
    wandb.finish()
