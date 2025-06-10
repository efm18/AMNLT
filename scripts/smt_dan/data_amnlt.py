import random
import re
import cv2
import torch
import numpy as np
import cv2
import sys
import gc

from ExperimentConfig import ExperimentConfig
from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from utils import check_and_retrieveVocabulary
from rich import progress
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision import transforms

from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

@memory.cache
def load_set(path, base_folder, fileformat, notation, reduce_ratio=1.0, fixed_size=None):
    x = []
    y = []
    with open(path) as datafile:
        lines = datafile.readlines()
        for line in progress.track(lines):
            excerpt = line.replace("\n", "")
            try:
                if notation == "music_aware":
                    gt_folder = "GT_music_aware"
                else:
                    gt_folder = "GT"
                with open(f"Data/{base_folder}/{gt_folder}/{'.'.join(excerpt.split('.')[:-1])}.gabc") as gabcfile:
                    gabc_content = gabcfile.read()
                    fname = ".".join(excerpt.split('.')[:-1])
                    img = cv2.imread(f"Data/{base_folder}/Images/{fname}{fileformat}")
                    if fixed_size != None:
                        width = fixed_size[1]
                        height = fixed_size[0]
                    elif img.shape[1] > 3056:
                        width = int(np.ceil(3056 * reduce_ratio))
                        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))
                    else:
                        width = int(np.ceil(img.shape[1] * reduce_ratio))
                        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))

                    img = cv2.resize(img, (width, height))
                    y.append([content + '\n' for content in gabc_content.split("\n")])
                    x.append(img)
                    
            except Exception:
                print(f'Error reading Data/{base_folder}/{excerpt}')

    return x, y

@memory.cache
def load_set_online(path, base_folder, fileformat, notation, reduce_ratio= 1.0, fixed_size=None):
    img_paths = []
    img_dims = []
    y = []
    
    with open(path) as datafile:
        lines = datafile.readlines()
        for line in progress.track(lines):
            excerpt = line.strip()  # Clean the line
            try:
                #gt_folder = "GT_music_aware" if notation == "music_aware" else "GT"
                gt_folder = "GT"

                with open(f"Data/{base_folder}/{gt_folder}/{'.'.join(excerpt.split('.')[:-1])}.gabc") as gabcfile:
                    gabc_content = gabcfile.read()

                    fname = ".".join(excerpt.split('.')[:-1])
                    img_path = f"Data/{base_folder}/Images/{fname}{fileformat}"
                    
                    img_paths.append(img_path)
                    
                    # Load the image to get its dimensions
                    img = cv2.imread(img_path)
                    if img is not None:
                        if fixed_size != None:
                            width = fixed_size[1]
                            height = fixed_size[0]
                        elif img.shape[1] > 3056:
                            width = int(np.ceil(3056 * reduce_ratio))
                            height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))
                        else:
                            width = int(np.ceil(img.shape[1] * reduce_ratio))
                            height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))

                        img_dims.append((height, width))  # (height, width)
                    else:
                        img_dims.append((0, 0))  # Handle error case if needed
                    
                    y.append([content + '\n' for content in gabc_content.split("\n")])

            except Exception as e:
                print(f"Error reading data from {base_folder}/{excerpt}: {e}")
    
    return img_paths, y, img_dims

def batch_preparation_img2seq(data):
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]

    max_image_width = max(128, max([img.shape[2] for img in images]))
    max_image_height = max(256, max([img.shape[1] for img in images]))

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.zeros(size=[len(dec_in),max_length_seq])
    y = torch.zeros(size=[len(gt),max_length_seq])

    for i, seq in enumerate(dec_in):
        decoder_input[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[:-1]]))
    
    for i, seq in enumerate(gt):
        y[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[1:]]))
    
    return X_train, decoder_input.long(), y.long()

class OMRIMG2SEQDataset(Dataset):
    def __init__(self, augment=False) -> None:
        self.teacher_forcing_error_rate = 0.2
        self.x = None
        self.y = None
        self.augment = augment

        super().__init__()
    
    def apply_teacher_forcing(self, sequence):
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
                errored_sequence[token] = np.random.randint(0, len(self.w2i))
        
        return errored_sequence

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.augment:
            x = augment(self.x[index])
        else:
            x = convert_img_to_tensor(self.x[index])
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width
    
    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']
    
    def get_dictionaries(self):
        return self.w2i, self.i2w
    
    def get_i2w(self):
        return self.i2w
    
class AMNLTSingleSystem(OMRIMG2SEQDataset):
    def __init__(self, data_path, base_folder, fileformat, notation, reduce_ratio, augment=False) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.x, self.y = load_set(data_path, base_folder, fileformat, notation, reduce_ratio=reduce_ratio)
        self.notation = notation
        self.y = self.preprocess_gt(self.y)
        self.tensorTransform = transforms.ToTensor()
        self.num_sys_gen = 1
        self.fixed_systems_num = False

    def get_width_avgs(self):
        widths = [image.shape[1] for image in self.x]
        return np.average(widths), np.max(widths), np.min(widths)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return len(self.x)
    
    def preprocess_gt(self, Y):
        for idx, gabc in enumerate(Y):
            if self.notation == "char":
                gabc = list(gabc[0].strip())
                
            elif self.notation == "music_aware":
                gabc = gabc[0]
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
                
            elif self.notation == "new_gabc":
                gabc = gabc[0]
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
                
            Y[idx] = ['<bos>'] + gabc + ['<eos>']
        return Y
    
class AMNLTSingleSystemOnTheFly(OMRIMG2SEQDataset):
    def __init__(self, data_path, base_folder, fileformat, notation, reduce_ratio, augment=False) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.img_paths, self.y, self.img_dims = load_set_online(data_path, base_folder, fileformat, notation, reduce_ratio=reduce_ratio)
        self.notation = notation
        self.reduce_ratio = reduce_ratio
        self.y = self.preprocess_gt(self.y)
        self.tensorTransform = transforms.ToTensor()
        self.num_sys_gen = 1
        self.fixed_systems_num = False

    def get_width_avgs(self):
        widths = [dims[1] for dims in self.img_dims]  # Extract widths from cached dimensions
        return np.average(widths), np.max(widths), np.min(widths)
    
    def get_max_hw(self):
        max_height = max([dims[0] for dims in self.img_dims])
        max_width = max([dims[1] for dims in self.img_dims])
        return max_height, max_width
    
    def __getitem__(self, index):
        # Load image on-the-fly when __getitem__ is called
        img_path = self.img_paths[index]
        
        # Read the image from disk
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Error loading image at {img_path}")

        # Apply resizing based on fixed size or ratio
        if self.fixed_systems_num:
            width = self.fixed_systems_num[1]
            height = self.fixed_systems_num[0]
        elif img.shape[1] > 3056:
            width = int(np.ceil(3056 * self.reduce_ratio))
            height = int(np.ceil(max(img.shape[0], 256) * self.reduce_ratio))
        else:
            width = int(np.ceil(img.shape[1] * self.reduce_ratio))
            height = int(np.ceil(max(img.shape[0], 256) * self.reduce_ratio))

        # Resize the image
        img = cv2.resize(img, (width, height))

        # Apply augmentation if enabled
        if self.augment:
            img = augment(img)
        else:
            img = convert_img_to_tensor(img)

        # Convert ground truth to tensor
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)
        
        return img, decoder_input, y
    
    def __len__(self):
        return len(self.img_paths)
    
    def preprocess_gt(self, Y):
        for idx, gabc in enumerate(Y):
            if self.notation == "char":
                gabc = list(gabc[0].strip())
                
            elif self.notation == "music_aware":
                gabc = gabc[0]
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
                
            elif self.notation == "new_gabc":
                gabc = gabc[0]
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
                
            Y[idx] = ['<bos>'] + gabc + ['<eos>']
        return Y

class AMNLTDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.img_format = config.img_format
        self.notation = config.transcript_format
        self.reduce_ratio = config.reduce_ratio
        if self.vocab_name not in ["GregoSynth", "GregoSynth_music_aware"]:
            self.train_set = AMNLTSingleSystem(data_path=f"{self.data_path}/train_gt_fold.dat", base_folder=self.vocab_name, fileformat=self.img_format, notation=self.notation, reduce_ratio=self.reduce_ratio, augment=True)
            self.val_set = AMNLTSingleSystem(data_path=f"{self.data_path}/val_gt_fold.dat", base_folder=self.vocab_name, fileformat=self.img_format, notation=self.notation, reduce_ratio=self.reduce_ratio)
            self.test_set = AMNLTSingleSystem(data_path=f"{self.data_path}/test_gt_fold.dat", base_folder=self.vocab_name, fileformat=self.img_format, notation=self.notation, reduce_ratio=self.reduce_ratio)
            
        else:
            self.train_set = AMNLTSingleSystemOnTheFly(data_path=f"{self.data_path}/train_gt_fold.dat", base_folder=self.vocab_name, fileformat=self.img_format, notation=self.notation, reduce_ratio=self.reduce_ratio, augment=True)
            self.val_set = AMNLTSingleSystemOnTheFly(data_path=f"{self.data_path}/val_gt_fold.dat", base_folder=self.vocab_name, fileformat=self.img_format, notation=self.notation, reduce_ratio=self.reduce_ratio)
            self.test_set = AMNLTSingleSystemOnTheFly(data_path=f"{self.data_path}/test_gt_fold.dat", base_folder=self.vocab_name, fileformat=self.img_format, notation=self.notation, reduce_ratio=self.reduce_ratio)

        if self.notation == "music_aware":
            vocab_name = self.vocab_name + "_music_aware"
        else:
            vocab_name = self.vocab_name
            
        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab", f"{vocab_name}")

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)