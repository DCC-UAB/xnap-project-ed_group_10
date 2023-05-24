import torch

# Data
json_file = "./data/data_images/data/ImageSets/0/split_0.json"
img_dir = "./data/data_images/data/JPEGImages/"
txt_dir = "./data/data_images/data/ocr_labels/"
data_dir = 'data/'
train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'
num_classes = 28
batch_size = 16
num_workers = 6

# Model
image_size = 256
dim = 256
depth = 2
channels = 3
heads = 4
num_layers = 12
mlp_dim = 512
num_heads = 12
dim_feedforward = 3072
dropout = 0.1
pretrained_weights = 'bert-base-uncased'
pretrained_backbone = True
max_num_words = 64
pretained_model = "resnet50"
text_model = "fasttext"
dropout = 0
bert_model = "bert-base-uncased"

# Training
lr = 1e-4
epochs = 2
patience = 5
early_stopping = True
save_dir = 'models/saved_models/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gamma = 0.1
data_augmentation = False
