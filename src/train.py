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
from utils.utils import get_data, make_loader, make, context_inference


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
    
    fasttext.util.download_model('en', if_exists='ignore')  # English
    fasttext_model = fasttext.load_model('cc.en.300.bin')
    
    model = conTextTransformer(image_size=256, num_classes=28, channels=3, dim=256, depth=2, heads=4, mlp_dim=512)

    !wget https://raw.githubusercontent.com/lluisgomez/ConTextTransformer/main/all_best.pth

    model.load_state_dict(torch.load('all_best.pth'))
    model.to(device)
    model.eval()
    
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class_labels = {
        1: "Bakery", 
        2: "Barber", 
        3: "Bistro", 
        4: "Bookstore", 
        5: "Cafe", 
        6: "ComputerStore", 
        7: "CountryStore", 
        8: "Diner", 
        9: "DiscounHouse", 
        10: "Dry Cleaner", 
        11: "Funeral", 
        12: "Hotspot", 
        13: "MassageCenter", 
        14: "MedicalCenter", 
        15: "PackingStore", 
        16: "PawnShop", 
        17: "PetShop", 
        18: "Pharmacy", 
        19: "Pizzeria", 
        20: "RepairShop", 
        21: "Restaurant", 
        22: "School", 
        23: "SteakHouse", 
        24: "Tavern", 
        25: "TeaHouse", 
        26: "Theatre", 
        27: "Tobacco", 
        28: "Motel"}
    
    !wget -q https://gailsbread.co.uk/wp-content/uploads/2017/11/Summertown-1080x675.jpg

    from IPython.display import Image as ShowImage
    ShowImage('Summertown-1080x675.jpg')
    
    OCR_tokens = [] # Let's imagine our OCR model does not recognize any text

    probs = context_inference('Summertown-1080x675.jpg', OCR_tokens)
    class_id = np.argmax(probs)
    print('Prediction without text: {} ({})'.format(class_labels[class_id+1], probs[0,class_id]))


    OCR_tokens = ['GAIL', 'ARTISAN', 'BAKERY'] # Simulate a perfect OCR output

    probs = context_inference('Summertown-1080x675.jpg', OCR_tokens)
    class_id = np.argmax(probs)
    print('Prediction with text:\t {} ({})'.format(class_labels[class_id+1], probs[0,class_id]))
    
    # + ---------------------------------------------------------
    # | Now we can play a bit with the model ... try different images, contraste the predictions when using different words as OCR tokens, simulate OCR errors, etc.
    # + ---------------------------------------------------------
    
    # add your code here ... 


    # + ---------------------------------------------------------
    
    