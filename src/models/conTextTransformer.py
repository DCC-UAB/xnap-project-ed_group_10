# | Model for the conTextTransformer
# + ------------------------------------
# + ------------------------------------

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


class ConTextTransformer(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        
        resnet50 = torchvision.models.resnet50(pretrained=True) #Carregar el model pre-entrenat resnet50
        modules=list(resnet50.children())[:-2] #Agafar tots els layers menys els dos últims
        self.resnet50=nn.Sequential(*modules) #Crear un model amb els layers anteriors
        for param in self.resnet50.parameters(): #Congelar els paràmetres del model
            param.requires_grad = False
        self.num_cnn_features = 64  # 8x8 Num característiques de la CNN
        self.dim_cnn_features = 2048 # 8x8x2048 Dimensió de les característiques de la CNN
        self.dim_fasttext_features = 300 # Dimensió de les característiques de FastText

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_cnn_features + 1, dim)) # Embedding de posició. Matriu de incrustació on capturem informació de la posició de les característiques extretes de la CNN
        self.cnn_feature_to_embedding = nn.Linear(self.dim_cnn_features, dim) # Capa lineal on passem les caracterísitques de la CNN de dimensió 2048 a dimensió dim
        self.fasttext_feature_to_embedding = nn.Linear(self.dim_fasttext_features, dim) # Capa lineal on passem les caracterísitques de FastText de dimensió 300 a dimensió dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # Parametre entrenable on capturem la informació global del context com les característiques locals de les imatges i els textos d'entrada
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True) # Capa individual del encoder del transformer. Aquesta capa té com a entrada les característiques de la CNN i FastText i com a sortida les característiques de la CNN i FastText amb més informació contextual
        encoder_norm = nn.LayerNorm(dim) # Normalització de les característiques de la CNN i FastText. LayerNorm redueix la covariancia
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth) # Consta de num_layers capes del encoder del transformer. Apila i contecta els blocs de encoder_layer

        self.to_cls_token = nn.Identity() #Capa identitat que no altera les dades. Posteriorment la utilitzem per agafar la informació del cls_token

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim), #Espai dimensional més gran per poder captar millor la informació
            nn.GELU(), #Funcio d'activacio que introdueix no linearitat i millor la capacitat de captar relacions compelxes entre les dades
            nn.Linear(mlp_dim, num_classes) #Capa lineal que ens dona la probabilitat de que la img sigui de cada classe
        )

    def forward(self, img, txt, mask=None):
        x = self.resnet50(img)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.cnn_feature_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x2 = self.fasttext_feature_to_embedding(txt.float())
        x = torch.cat((x,x2), dim=1)

        #tmp_mask = torch.zeros((img.shape[0], 1+self.num_cnn_features), dtype=torch.bool)
        #mask = torch.cat((tmp_mask.to(device), mask), dim=1)
        #x = self.transformer(x, src_key_padding_mask=mask)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)