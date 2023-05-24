# + ------------------------------------
# | Model for the conTextDataset
# + ------------------------------------

import time,os,json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import fasttext
import fasttext.util

import config

class ConTextDataset(Dataset):
    def __init__(self, json_file, root_dir, root_dir_txt, 
                 train=True, transform=None):
        
        with open(json_file) as f:
            data = json.load(f)
        self.train = train
        self.root_dir = root_dir
        self.root_dir_txt = root_dir_txt
        self.transform = transform
        if (self.train):
            self.samples = data['train']
        else:
            self.samples = data['test']

        fasttext.util.download_model('en', if_exists='ignore')  # English
        self.fasttext = fasttext.load_model('cc.en.300.bin')
        
        self.dim_fasttext = self.fasttext.get_dimension()
        self.max_num_words = config.max_num_words


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.samples[idx][0]+'.jpg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text = np.zeros((self.max_num_words, self.dim_fasttext))
        text_mask = np.ones((self.max_num_words,), dtype=bool)
        text_name = os.path.join(self.root_dir_txt, self.samples[idx][0]+'.json')
        with open(text_name) as f:
            data = json.load(f)

        words = []
        if 'textAnnotations' in data.keys():
            for i in range(1,len(data['textAnnotations'])):
                word = data['textAnnotations'][i]['description']
                if len(word) > 2: words.append(word)

        words = list(set(words))
        for i,w in enumerate(words):
            if i>=self.max_num_words: break
            text[i,:] = self.fasttext.get_word_vector(w)
            text_mask[i] = False
        
        target = self.samples[idx][1] - 1

        return image, text, text_mask, target