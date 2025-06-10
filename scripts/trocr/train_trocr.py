import random
import numpy as np
import torch
import gc
import os

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from transformers import TrOCRProcessor, PreTrainedTokenizerFast

from models.model import CTCTrainedTrOCR
from my_utils.dataset import CTCDataset
from my_utils.data_preprocessing import trocr_batch_preparation
from data.config import DS_CONFIG

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_trocr(
    ds_name: str,
    encoding_type="char",
    epochs=1000,
    patience=20,
    batch_size=8,
    project="AMNLT",
    group="Baseline-TrOCR",
    entity="el_iseo",
):
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Training TrOCR on dataset: {ds_name}")

    # 🔧 Use your custom tokenizer instead of default
    tokenizer_path = f"trocr_utils/tokenizer_trocr_custom/{ds_name}"
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"Custom tokenizer not found at: {tokenizer_path}")

    # Load vision extractor from base model
    base_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Create new processor using base vision extractor + custom tokenizer
    processor = TrOCRProcessor(feature_extractor=base_processor.feature_extractor, tokenizer=tokenizer)

    # Load datasets
    train_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["train"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        model_name="trocr",
        encoding_type=encoding_type,
    )
    val_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["val"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        model_name="trocr",
        train=False,
        encoding_type=encoding_type,
    )
    test_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["test"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        model_name="trocr",
        train=False,
        encoding_type=encoding_type,
    )

    # Inject processor
    train_ds.processor = processor
    val_ds.processor = processor
    test_ds.processor = processor

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=trocr_batch_preparation,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=trocr_batch_preparation,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=trocr_batch_preparation,
    )

    # 🔧 Pass tokenizer path into the model
    model = CTCTrainedTrOCR(ds_name=ds_name, tokenizer_path=tokenizer_path)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=f"{ds_name}_{encoding_type}_trocr",
            monitor="val_cer",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val_cer",
            mode="min",
            patience=patience,
            verbose=True,
        ),
    ]

    # Logger
    logger = WandbLogger(
        project=project,
        group=group,
        name=f"TROCR-{encoding_type.upper()}-{ds_name}",
        entity=entity,
        log_model=False,
    )

    # Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        precision="16-mixed",
    )

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    best_model_path = callbacks[0].best_model_path
    model = CTCTrainedTrOCR.load_from_checkpoint(best_model_path, ds_name=ds_name, tokenizer_path=tokenizer_path)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    import fire
    fire.Fire(train_trocr)
