import numpy as np
import torch
import skimage
import os
import itertools
from skimage import transform as stf
from PIL import Image
from torch.utils.data import Dataset
import gin
import sys

# Random Projective Transformation
def RndTform(img, val):
    Ih, Iw = img[0].shape[:2]
    sgn = torch.randint(0, 2, (1,)).item() * 2 - 1
    dw, dh = (val, 0) if sgn > 0 else (0, val)
    rd = lambda d: torch.empty(1).uniform_(-d, d).item()
    fd = lambda d: torch.empty(1).uniform_(-dw, d).item()

    tl_top, tl_left = rd(dh), fd(dw)
    bl_bottom, bl_left = rd(dh), fd(dw)
    tr_top, tr_right = rd(dh), fd(min(Iw * 3/4 - tl_left, dw))
    br_bottom, br_right = rd(dh), fd(min(Iw * 3/4 - bl_left, dw))

    tform = stf.ProjectiveTransform()
    tform.estimate(
        np.array([(tl_left, tl_top), (bl_left, Ih - bl_bottom), 
                  (Iw - br_right, Ih - br_bottom), (Iw - tr_right, tr_top)]),
        np.array([[0, 0], [0, Ih - 1], [Iw - 1, Ih - 1], [Iw - 1, 0]])
    )

    corners = np.array([[0, 0], [0, Ih - 1], [Iw - 1, Ih - 1], [Iw - 1, 0]])
    corners = tform.inverse(corners)
    minc, minr = corners[:, 0].min(), corners[:, 1].min()
    maxc, maxr = corners[:, 0].max(), corners[:, 1].max()
    output_shape = np.around((maxr - minr + 1, maxc - minc + 1))

    tform = stf.SimilarityTransform(translation=(minc, minr)) + tform
    tform.params /= tform.params[2, 2]

    ret = []
    for i in range(len(img)):
        img2 = stf.warp(img[i], tform, output_shape=output_shape, cval=1.0)
        img2 = stf.resize(img2, (Ih, Iw), preserve_range=True).astype(np.float32)
        ret.append(img2)

    return ret

@gin.configurable
def SameTrCollate(batch, prjAug, prjVal):
    images, labels = zip(*batch)
    images = [image.transpose((1, 2, 0)) for image in images]
    if prjAug:
        images = [RndTform([image], val=prjVal)[0] for image in images]
    image_tensors = [torch.from_numpy(np.array(image, copy=False)) for image in images]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    image_tensors = image_tensors.permute(0, 3, 1, 2)
    return image_tensors, labels

class myLoadDS(Dataset):
    def __init__(self, flist, dpath, gt_path=None, ralph=None, fmin=True, mln=None):
        self.fns = get_files(flist, dpath)

        if ralph is None:
            dataset_dir = os.path.dirname(dpath)
            dataset_name = os.path.basename(dataset_dir)
            vocab_dir = os.path.join(dataset_dir, "vocab")
            alph = get_alphabet(vocab_dir, dataset=dataset_name)
            self.alph = alph
            self.ralph = dict(zip(alph.values(), alph.keys()))
        else:
            self.ralph = ralph

        self.tlbls = get_labels(self.fns)

        if mln is not None:
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns = np.asarray(self.fns)[filt].tolist()

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        timgs = get_images(self.fns[index])
        timgs = timgs.transpose((2, 0, 1))
        return (timgs, self.tlbls[index])

def get_files(nfile, dpath):
    with open(nfile, 'r') as f:
        fnames = [os.path.join(dpath, x.strip()) for x in f.readlines()]
    return fnames

def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]
    y = min(int(y * max_h / x), max_w)
    x = max_h
    return np.array(Image.fromarray(img).resize((y, x)))

@gin.configurable
def get_images(fname, max_w, max_h, nch):
    try:
        # 🛠️ Convert to RGB explicitly to drop alpha channels
        image_data = np.array(Image.open(fname).convert("RGB"))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)
        if nch == 3 and image_data.shape[2] != 3:
            image_data = np.tile(image_data, (1, 1, 3))

        image_data = np.pad(
            image_data,
            ((0, 0), (0, max_w - image_data.shape[1]), (0, 0)),
            mode='constant',
            constant_values=(1.0)
        )
    except IOError as e:
        print('Could not read:', fname, ':', e)
    return image_data

def preprocess_gt(gabc, label_file):    
    if "Solesmes" in label_file or "GregoSynth" in label_file:
        muaw_gabc = []
        i = 0
        while i < len(gabc):
            if gabc[i:i+3] == "<m>":
                if i + 3 < len(gabc):
                    muaw_gabc.append(gabc[i:i+4])
                    i += 4
                else:
                    break
            else:
                muaw_gabc.append(gabc[i])
                i += 1
                
        gabc = muaw_gabc
        
    elif "Einsiedeln" in label_file or "Salzinnes" in label_file:
        new_gabc = []
        i = 0
        while i < len(gabc):
            if gabc[i] == "(":
                new_gabc.append(gabc[i])
                i += 1
                temp = ""
                while i < len(gabc) and gabc[i] != ")":
                    temp += gabc[i]
                    i += 1
                    
                for token in temp.split():
                    new_gabc.append(token)
                    
                if i < len(gabc):
                    new_gabc.append(gabc[i])
                    i += 1
            else:
                new_gabc.append(gabc[i])
                i += 1
        
        gabc = new_gabc
    
    return gabc
    

def get_labels(fnames):
    labels = []
    for image_file in fnames:
        base = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(os.path.dirname(image_file).replace("Images", "GT"), base + '.gabc')

        try:
            with open(label_file, 'r') as f:
                raw_label = ' '.join(f.read().split())
                label = preprocess_gt(raw_label, label_file)
                labels.append(label)
        except Exception as e:
            print(f"Warning: Failed to read label for {image_file}: {e}")
            labels.append([])

    return labels


def get_alphabet(dir_path, dataset=""):
    w2i_path = os.path.join(dir_path, f"{dataset}w2i.npy")
    if not os.path.exists(w2i_path):
        raise FileNotFoundError(f"Vocabulary file not found: {w2i_path}")
    alph = np.load(w2i_path, allow_pickle=True).item()
    return alph
