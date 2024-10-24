import re
import joblib
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

memory = joblib.memory.Memory("./cache", mmap_mode="r", verbose=0)

NUM_CHANNELS = 1
IMG_HEIGHT = 64
IMG_WIDTH = 256
toTensor = transforms.ToTensor()

################################# Image preprocessing:

def preprocess_image_from_file(path, unfolding=False, reduce=False):
    x = Image.open(path).convert("L")  # Convert to grayscale
    
    if not unfolding:
        # Not preserving aspect ratio
        #new_width = x.width // 4
        #x = x.resize((new_width, IMG_HEIGHT))  # Resize
        
        # Resize (preserving aspect ratio)
        new_width = int(
            IMG_HEIGHT * x.size[0] / x.size[1]
        )
        x = x.resize((new_width, IMG_HEIGHT))
    
    # If the boolean flag is True, rotate -90 degrees and flip horizontally
    else:
        reduction_factor = 2
        new_width = x.width // reduction_factor
        new_height = x.height // reduction_factor
        
        if reduce:
            reduce_factor = 0.5
            new_width = int(new_width * reduce_factor)
            new_height = int(new_height * reduce_factor)
        
        x = x.resize((new_width, new_height))
        x = x.rotate(-90, expand=True)  # Rotate by -90 degrees
    
    x = toTensor(x)  # Convert to tensor (normalizes to [0, 1])
    #print(x.shape)
    #sys.exit()
    return x


################################# Transcript preprocessing:

def preprocess_transcript_from_file(path, w2i, ds_name, encoding_type="char"):
    if (encoding_type == "char") or (encoding_type == "new_gabc" and ds_name in ["einsiedeln_lyrics", "salzinnes_lyrics"]):
        with open(path, "r") as file:
            y = file.read().strip()
            return torch.tensor([w2i[c] for c in y])
        
    elif encoding_type == "new_gabc" and ds_name in ["einsiedeln_music", "salzinnes_music"]:
        with open(path, "r") as file:
            content = file.read().strip()
            tokens = []
            i = 0
            temp = ''
            while i < len(content):
                if content[i] == " " or content[i] == ")":
                    if temp:
                        tokens.append(temp)
                    tokens.append(content[i])
                    temp = ''
                elif content[i] == "(":
                    tokens.append(content[i])
                else:
                    temp += content[i]
                i += 1
            if temp:
                tokens.append(temp)
            return torch.tensor([w2i[token] for token in tokens])                
    
    elif encoding_type == "new_gabc" and ds_name in ["einsiedeln", "salzinnes"]:
        with open(path, "r") as file:
            content = file.read().strip()
        tokens = []
        i = 0
        while i < len(content):
            if content[i] == '(':
                tokens.append(content[i])
                i += 1
                temp = ''
                while i < len(content) and content[i] != ')':
                    temp += content[i]
                    i += 1
                for token in temp.split():
                    tokens.append(token)
                if i < len(content):
                    tokens.append(content[i])
                    i += 1
            else:
                tokens.append(content[i])
                i += 1
        return torch.tensor([w2i[token] for token in tokens])
    
    elif encoding_type == "music_aware":
        with open(path, "r") as file:
            content = file.read().strip()
            i = 0
            tokens = []
            while i < len(content):
                #print(content[i:i+3])
                if content[i:i+3] == "<m>":  # Comprueba si el caracter actual tiene etiqueta musical
                    # Si es así, extrae el caracter musical con la etiqueta
                    if i + 3 < len(content):  # Asegura que no se sale del rango
                        tokens.append("<m>" + content[i+3])
                        i += 3  # Salta al siguiente caracter después de <m>
                    else:
                        break
                else:
                    # Añade el caracter normal al vocabulario
                    tokens.append(content[i])
                i += 1
        return torch.tensor([w2i[c] for c in tokens])


################################# CTC Preprocessing:


def pad_batch_images(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x


def pad_batch_transcripts(x):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(torch.int32)
    return x


def ctc_batch_preparation(batch):
    x, xl, y, yl, img_path = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Zero-pad transcripts to maximum batch transcript length
    y = pad_batch_transcripts(y)
    yl = torch.tensor(yl, dtype=torch.int32)
    return x, xl, y, yl, img_path
