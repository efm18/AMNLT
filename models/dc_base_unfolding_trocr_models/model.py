import random
import os
import shutil
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.nn import CTCLoss
from torchinfo import summary
from torchvision.utils import save_image

from my_utils.augmentations import AugmentStage
from my_utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS
from my_utils.metrics import compute_metrics, ctc_greedy_decoder
from models.modules import CRNN
from models.modules import E2EScore_FCN, E2EScore_CRNN, E2EScore_CNNT2D, E2EScore_VAN, VANCTCModel

from transformers import VisionEncoderDecoderModel, TrOCRProcessor, PreTrainedTokenizerFast

class CTCTrainedCRNN(LightningModule):
    def __init__(self, w2i, i2w, ctc, use_augmentations=True, ytest_i2w=None, check_train=False, ds_name=None):
        super(CTCTrainedCRNN, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        # Model
        self.model = CRNN(output_size=len(self.w2i) + 1)
        self.summary = summary(self.model, input_size=(1, NUM_CHANNELS, IMG_HEIGHT, 256))
        # Augmentations
        self.augment = AugmentStage() if use_augmentations else lambda x: x
        # Loss
        self.ctc = ctc
        self.compute_ctc_loss = CTCLoss(
            blank=len(self.w2i), zero_infinity=False
        )  # The target index cannot be blank!
        # Predictions
        self.Y = []
        self.YHat = []
        self.raw_ctcs = []
        
        # Check if the model can learn
        self.check_train = check_train
        
        # To save the predictions in a file
        self.ds_name = ds_name
        
        parts = self.ds_name.split("_")
        if "music" in parts:
            self.music = True
        else:
            self.music = False
        
        # Threshold for very large losses
        self.losses = []
        self.img_paths = []
        self.threshold = 4

    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def forward(self, x):
        return self.model(x)
    
    def check_image(self, x, loss, img_path, avg_loss, loss_dir):      
        if abs(loss - avg_loss) > self.threshold:
            loss_path = os.path.join(loss_dir, os.path.basename(img_path))
            x.save(loss_path)

    def training_step(self, batch, batch_idx):
        x, xl, y, yl, img_path = batch
        x = self.augment(x)
        yhat = self.model(x)
        # ------ CTC Requirements ------
        # yhat: [batch, frames, vocab_size]
        yhat = yhat.log_softmax(dim=2)        
        yhat = yhat.permute(1, 0, 2).contiguous()
        # ------------------------------
        
        loss = self.compute_ctc_loss(yhat, y, xl, yl)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        
        self.losses.append(loss.item())
        self.img_paths.append(img_path[0])
        
        if self.check_train:
            yhat = self.model(x)[0]
            yhat = yhat.log_softmax(dim=-1).detach().cpu()
            yhat = ctc_greedy_decoder(yhat, self.i2w)
            y = [self.ytest_i2w[i.item()] for i in y[0]]
            metric = compute_metrics(y_true=[y], y_pred=[yhat], music=self.music)
            for k, v in metric.items():
                self.log(f"train_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        
        return loss
    
    def on_training_epoch_end(self):
        avg_loss = 0.0
        
        for loss in self.losses:
            avg_loss += loss
            
        avg_loss /= len(self.losses)
        
        loss_dir = "large_losses"
        if os.path.exists(loss_dir):
            shutil.rmtree(loss_dir)
        os.makedirs(loss_dir, exist_ok=True)
        
        for loss, img_path in zip(self.losses, self.img_paths):
            x = Image.open(img_path)
            self.check_image(x, loss, img_path, avg_loss, loss_dir)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch  # batch_size = 1
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)[0]
        
        if self.ctc == "greedy":
            yhat = yhat.log_softmax(dim=-1).detach().cpu()
            self.raw_ctcs.append(yhat)
            yhat = ctc_greedy_decoder(yhat, self.i2w)

        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0]]
        # Append to later compute metrics
        
        self.Y.append(y)
        self.YHat.append(yhat)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        print(self.music)
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat, music=self.music)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
            
        with open(f"predictions_{self.ds_name}.txt", 'w') as f:
            for pred, gt in zip(self.YHat, self.Y):
                pred_str = ''.join(pred)
                f.write(f"{pred_str}\n")
                
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)

class LightningE2EModelUnfolding(LightningModule):
    def __init__(self, w2i, i2w, model, use_augmentations=True, ytest_i2w=None, encoding_type="char", mh=0, mw=0):
        super(LightningE2EModelUnfolding, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        # Model
        if model == "fcn":
            self.model = E2EScore_FCN(1, len(self.w2i) + 1)
        elif model == "crnnunfolding":
            self.model = E2EScore_CRNN(1, len(self.w2i) + 1)
        elif model == "cnnt2d":
            self.model = E2EScore_CNNT2D(1, len(self.w2i) + 1, mh, mw)
        elif model == "van":
            self.model = E2EScore_VAN(1, len(self.w2i) + 1)
        self.summary()
        # Augmentations
        self.augment = AugmentStage() if use_augmentations else lambda x: x
        # Loss
        self.compute_ctc_loss = CTCLoss(
            blank=len(self.w2i)
        )  # The target index cannot be blank!
        # Predictions
        self.Y = []
        self.YHat = []
        # For correctly computing the CER
        self.encoding_type = encoding_type
        
        # Threshold for very large losses
        self.losses = []
        self.img_paths = []
        self.threshold = 4
        
    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])
     
    # Checking if a bigger learning rate helps   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, xl, y, yl, img_path = batch
        x = self.augment(x)
        yhat = self.model(x)
        loss = self.compute_ctc_loss(yhat, y, torch.tensor(yhat.size()[0], dtype=torch.long), yl)
        # Save the best and worst predictions
        
        self.losses.append(loss)
        self.img_paths.append(img_path[0])
        
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def check_image(self, x, loss, img_path, avg_loss, loss_dir):      
        if abs(loss - avg_loss) > self.threshold:
            loss_path = os.path.join(loss_dir, os.path.basename(img_path))
            x.save(loss_path)

    def validation_step(self, batch, batch_idx):
        x, y, img_path = batch  # batch_size = 1
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)
        yhat = yhat.permute(1,0,2).contiguous()
        yhat = yhat[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat = ctc_greedy_decoder(yhat, self.i2w)
        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0]]
        
        if self.encoding_type == "music_aware":
            # Remove the <m> tags for correct evaluation
            filtered_y = []
            filtered_yhat = []
            i = 0
            while i < len(yhat):
                if i <= len(yhat) - 3 and yhat[i] == '<' and yhat[i+1] == 'm' and yhat[i+2] == '>':
                    # Saltar los siguientes tres tokens
                    i += 3
                else:
                    filtered_yhat.append(yhat[i])
                    i += 1
            i = 0
            while i < len(y):
                if i <= len(y) - 3 and y[i] == '<' and y[i+1] == 'm' and y[i+2] == '>':
                    # Saltar los siguientes tres tokens
                    i += 3
                else:
                    filtered_y.append(y[i])
                    i += 1
            # Append to later compute metrics
            self.Y.append(filtered_y)
            self.YHat.append(filtered_yhat)
            self.img_paths.append(img_path[0])
        
        else:
            self.Y.append(y)
            self.YHat.append(yhat)
            self.img_paths.append(img_path[0])

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        with open('predictions.txt', 'w') as f:
            for pred, img_path in zip(self.YHat, self.img_paths):
                pred_str = ''.join(pred)
                #f.write(f"{img_path}\t{pred_str}\n")
                f.write(f"{pred_str}\n")
                
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat, music=False)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)

######### PR REVIEW ##########

class CTCTrainedTrOCR(LightningModule):
    def __init__(self, ds_name, tokenizer_path, processor_name="microsoft/trocr-base-handwritten", lr=5e-5):
        super().__init__()
        self.save_hyperparameters()

        # Load base processor (for vision feature extractor)
        base_processor = TrOCRProcessor.from_pretrained(processor_name)

        # Load your custom tokenizer from saved path
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Reconstruct processor with custom tokenizer
        self.processor = TrOCRProcessor(
            feature_extractor=base_processor.feature_extractor,
            tokenizer=tokenizer,
        )

        # Load pre-trained TrOCR model
        self.model = VisionEncoderDecoderModel.from_pretrained(processor_name)

        # Set tokenizer and decoding config
        self.model.config.vocab_size = tokenizer.vocab_size
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = tokenizer.pad_token_id

        self.lr = lr
        self.ds_name = ds_name
        self.music = "music" in ds_name

        self.Y = []
        self.YHat = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        pred_ids = self.model.generate(batch["pixel_values"])
        preds = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        targets = self.processor.batch_decode(batch["labels"], skip_special_tokens=True)

        self.YHat.extend([list(p) for p in preds])
        self.Y.extend([list(t) for t in targets])

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self, name="val"):
        metrics = compute_metrics(self.Y, self.YHat, music=self.music)
        for k, v in metrics.items():
            self.log(f"val_{k}", v, prog_bar=True)

        with open(f"predictions_{self.ds_name}.txt", 'w') as f:
            for pred in self.YHat:
                f.write("".join(pred) + "\n")

        self.Y.clear()
        self.YHat.clear()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(name="test")