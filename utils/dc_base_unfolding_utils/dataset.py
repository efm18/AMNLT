import json
import os
import re
import sys
from PIL import Image

import torch
from torch.utils.data import Dataset

from my_utils.data_preprocessing import (
    preprocess_image_from_file,
    preprocess_transcript_from_file,
)

################################################################################################ Single-source:


class CTCDataset(Dataset):
    def __init__(
        self,
        name,
        samples_filepath,
        transcripts_folder,
        img_folder,
        model_name,
        train=True,
        da_train=False,
        width_reduction=2,
        encoding_type="char",
    ):
        self.name = name
        self.model_name = model_name
        self.train = train
        self.da_train = da_train
        self.width_reduction = width_reduction
        self.encoding_type = encoding_type

        # Get image paths and transcripts
        self.X, self.Y = self.get_images_and_transcripts_filepaths(
            samples_filepath, img_folder, transcripts_folder
        )
        
        self.printbatch = False

        # Check and retrieve vocabulary
        vocab_name = f"w2i_{self.encoding_type}.json"
        vocab_folder = os.path.join(os.path.join("data", self.name), "vocab")
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary(transcripts_folder)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_name == "trocr":
            # TrOCR expects PIL image resized to 224x224 and tokenized label
            image = Image.open(self.X[idx]).convert("RGB").resize((224, 224))
            label = open(self.Y[idx], "r", encoding="utf-8").read().strip()

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
            label_ids = self.processor.tokenizer(label, return_tensors="pt").input_ids.squeeze()
            return {"pixel_values": pixel_values, "labels": label_ids}

        
        unfolding = False
        if self.model_name == "fcn" or self.model_name == "crnnunfolding" or self.model_name == "cnnt2d" or self.model_name == "van":
            unfolding = True
            
        if "gregoeli" in self.name and self.model_name in ["fcn", "crnnunfolding", "cnnt2d", "van"]:
            reduce = True
        else:
            reduce = False

        # CTC Training setting
        x = preprocess_image_from_file(self.X[idx], unfolding=unfolding, reduce=reduce)
        y = preprocess_transcript_from_file(self.Y[idx], self.w2i, self.name, self.encoding_type)
        
        img_path = self.X[idx]
        
        if self.train:
            # x.shape = [channels, height, width]
            if self.model_name == "fcn" or self.model_name == "crnnunfolding" or self.model_name == "cnnt2d" or self.model_name == "van":
                return x, (x.shape[2] // 8) * (x.shape[1] // 32), y, len(y), img_path
            elif self.model_name == "crnn":
                return x, x.shape[2] // self.width_reduction, y, len(y), img_path
            elif self.model_name == "ctc_van":
                return x, x.shape[2] // 8, y, len(y), img_path
            
        if self.printbatch:
            try:
                # Open the image from the provided path
                image = Image.open(img_path)
                # Save the image to a new location, e.g., as 'output_image.png'
                image.save("output_image.png")
                print(f"Image saved as 'output_image.png'")
            except Exception as e:
                print(f"Error loading or saving image: {e}")
            
            # Save the transcript (y) as a text file
            with open("output_transcript.txt", "w") as f:
                f.write(str(y))
            
            print(f"Transcript saved as 'output_transcript.txt'")
            
            # Set printbatch to False after saving
            self.printbatch = False
            
        return x, y, img_path
    
    def get_mx_hw(self):
        
        reduce = False
        
        if self.model_name == "cnnt2d" and self.name.startswith("gregoeli"):
            reduce = True
        
        max_height = max_width = 0
        for img_path in self.X:
            x = preprocess_image_from_file(img_path, unfolding=True, reduce=reduce)
            max_height = max(max_height, x.shape[1])
            max_width = max(max_width, x.shape[2])
        return max_height, max_width

    def get_images_and_transcripts_filepaths(
        self, img_dat_file_path, img_directory, transcripts_directory
    ):
        images = []
        transcripts = []
        
        # Images and transcripts are in different directories
        # Image filepath example: {image_name}.jpg
        # Transcript filepath example: {image_name}.jpg.txt

        # We are using the agnostic encoding for the transcripts

        # Read the .dat file to get the image file names
        with open(img_dat_file_path, "r") as file:
            img_files = file.read().splitlines()

        for img_file in img_files:
            file_name_without_extension, _ = os.path.splitext(img_file)
            img_path = os.path.join(img_directory, img_file)

            transcript_file = file_name_without_extension + ".gabc"
            transcript_path = os.path.join(transcripts_directory, transcript_file)

            if os.path.exists(img_path) and os.path.exists(transcript_path):
                images.append(img_path)
                transcripts.append(transcript_path)

        return images, transcripts

    def check_and_retrieve_vocabulary(self, transcripts_dir):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary(transcripts_dir)
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w
    
    def make_vocabulary(self, transcripts_dir):
        vocab = set()
        print(self.encoding_type)
        if (self.encoding_type == "char") or (self.encoding_type == "new_gabc" and self.name in ["einsiedeln_lyrics", "salzinnes_lyrics"]):
            for transcript_file in os.listdir(transcripts_dir):
                with open(os.path.join(transcripts_dir, transcript_file), "r") as file:
                    chars = file.read().strip()
                    vocab.update(chars)
        elif self.encoding_type == "music_aware":
            for transcript_file in os.listdir(transcripts_dir):
                with open(os.path.join(transcripts_dir, transcript_file), "r") as file:
                    # Leer todo el contenido del archivo
                    content = file.read().strip()
                    i = 0
                    while i < len(content):
                        if content[i:i+3] == "<m>":  # Check if the token starts with the musical tag
                            # If so, extract the musical character with the tag
                            if i + 3 < len(content):  # Ensure it doesn't go out of range
                                vocab.add("<m>" + content[i+3])
                                i += 3  # Skip to the next character after <m>
                            else:
                                break
                        else:
                            # Add the normal character to the vocabulary
                            vocab.add(content[i])
                        i += 1
        elif self.encoding_type == "new_gabc" and self.name in ["einsiedeln", "salzinnes"]:
            for transcript_file in os.listdir(transcripts_dir):
                with open(os.path.join(transcripts_dir, transcript_file), "r") as file:
                    content = file.read()
                    i = 0
                    while i < len(content):
                        if content[i] == '(':
                            # Tokenize '('
                            vocab.add(content[i])
                            i += 1
                            # Collect the characters inside the parentheses
                            temp = ''
                            while i < len(content) and content[i] != ')':
                                temp += content[i]
                                i += 1
                            # Split the collected characters by spaces and add them as tokens
                            for token in temp.split():
                                vocab.add(token)
                            # Tokenize ')'
                            if i < len(content):
                                vocab.add(content[i])
                                i += 1
                        else:
                            # Tokenize each character outside parentheses
                            vocab.add(content[i])
                            i += 1
                            
        elif self.encoding_type == "new_gabc" and self.name in ["einsiedeln_music", "salzinnes_music"]:
            for transcript_file in os.listdir(transcripts_dir):
                with open(os.path.join(transcripts_dir, transcript_file), "r") as file:
                    content = file.read().strip()
                    i = 0
                    temp = ''
                    while i < len(content):
                        if content[i] == " " or content[i] == ")":
                            if temp:
                                vocab.add(temp)
                            vocab.add(content[i])
                            temp = ''
                        elif content[i] == "(":
                            vocab.add(content[i])
                        else:
                            temp += content[i]
                        i += 1
                    if temp:
                        vocab.add(temp)
                    
        vocab = sorted(vocab)

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w