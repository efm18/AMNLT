import re
import joblib
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from torch.nn.utils.rnn import pad_sequence

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
        
        # CRNN
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

def resize_and_pad_batch_images_strict(images, min_required_width=800):
    """
    Resize all images so their width after rescaling to fixed height is ≥ min_required_width.
    Then resize all to a common height and pad to max width.
    Returns a list of [C, H, W] tensors.
    """
    pre_resized = []
    width_required_heights = []
    
    images = [to_pil_image(img) if isinstance(img, torch.Tensor) else img for img in images]

    # Step 1: compute required scale so width >= min_required_width
    for img in images:
        w, h = img.size
        scale_to_800 = max(min_required_width / w, 1.0)
        new_h = int(h * scale_to_800)
        new_w = int(w * scale_to_800)
        pre_resized.append(img.resize((new_w, new_h), resample=Image.BICUBIC))
        width_required_heights.append(new_h)

    # Step 2: determine common target height
    common_height = max(width_required_heights)

    # Step 3: resize all to common_height, then pad width
    resized_padded = []
    max_width = 0

    # Resize to common height, track max width
    resized_imgs = []
    for img in pre_resized:
        w, h = img.size
        scale = common_height / h
        new_w = int(w * scale)
        resized = img.resize((new_w, common_height), resample=Image.BICUBIC)
        resized_imgs.append(resized)
        max_width = max(max_width, new_w)

    # Pad to max width and convert to tensor
    for img in resized_imgs:
        mode = img.mode
        channels = len(img.getbands())
        pad_color = (255,) * channels
        padded = Image.new(mode, (max_width, common_height), color=pad_color)
        padded.paste(img, (0, 0))
        tensor = TF.to_tensor(padded)
        resized_padded.append(tensor)
        
    #for img in resized_padded:
    #    print(img.shape)

    return resized_padded

def pad_batch_transcripts(x):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(torch.int32)
    return x


def ctc_batch_preparation(batch):
    x, xl, y, yl, img_path = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    #print(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Zero-pad transcripts to maximum batch transcript length
    y = pad_batch_transcripts(y)
    yl = torch.tensor(yl, dtype=torch.int32)
    return x, xl, y, yl, img_path


################################# TrOCR Preprocessing:

def trocr_batch_preparation(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}
