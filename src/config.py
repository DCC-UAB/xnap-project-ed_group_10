import torch

# Data
json_file = '/datatmp/datasets/ConText/annotations/split_0.json'
img_dir = "../data/data_images/data/JPEGImages"
txt_dir = "/datatmp/datasets/ConText-WordAnnotations"
data_dir = 'data/'
train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'
num_classes = 28
batch_size = 32
num_workers = 8

# Model
image_size = 256
num_layers = 12
num_heads = 12
dim_feedforward = 3072
dropout = 0.1
pretrained_weights = 'bert-base-uncased'
pretrained_backbone = True

# Training
lr = 1e-4
num_epochs = 1
patience = 5
early_stopping = True
save_dir = 'models/saved_models/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gamma = 0.1