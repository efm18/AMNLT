import gc
import random

import fire
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

from my_utils.data_preprocessing import ctc_batch_preparation
from my_utils.dataset import CTCDataset
from data.config import DS_CONFIG
from models.model import CTCTrainedCRNN, LightningE2EModelUnfolding, CTCTrainedVAN

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(
    ds_name,
    model_name,
    encoding_type="char",
    epochs=1000,
    patience=20,
    batch_size=16,   # Change to 16 if using CRNN with post-processing or 1 if using FCN (space-issue in graphics card)
    use_augmentations=True,
    metric_to_monitor="val_cer",
    project="AMNLT",
    group="Baseline-CharacterLevel",
    entity="el_iseo",
    fine_tune=False,
    pretrain_model_path=None,
    ctc="greedy"
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if dataset exists
    if ds_name not in DS_CONFIG.keys():
        raise NotImplementedError(f"Dataset {ds_name} not implemented")

    # Experiment info
    print(f"Running experiment: {project} - {group} - {model_name}")
    print(f"\tDataset(s): {ds_name}")
    print(f"\tEncoding type: {encoding_type}")
    print(f"\tAugmentations: {use_augmentations}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tMetric to monitor: {metric_to_monitor}")

    # Get datasets
    train_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["train"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        model_name=model_name,
        encoding_type=encoding_type,
    )        
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        collate_fn=ctc_batch_preparation,
    )  # prefetch_factor=2
    val_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["val"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        model_name=model_name,
        train=False,
        encoding_type=encoding_type,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=20
    )  # prefetch_factor=2
    test_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["test"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        model_name=model_name,
        train=False,
        encoding_type=encoding_type,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=20
    )  # prefetch_factor=2
    
    if model_name == "cnnt2d":
        train_max_h, train_max_w = train_ds.get_mx_hw()
        val_max_h, val_max_w = val_ds.get_mx_hw()
        test_max_h, test_max_w = test_ds.get_mx_hw()

        # Calcular los máximos globales de altura y anchura
        max_h = max(train_max_h, val_max_h, test_max_h)
        max_w = max(train_max_w, val_max_w, test_max_w)
    else:
        max_h, max_w = 0, 0

    # Model
    if(fine_tune):
        if(model_name == "crnn"):
            model = CTCTrainedCRNN.load_from_checkpoint(pretrain_model_path, strict=False)
        elif(model_name == "fcn"):
            model = LightningE2EModelUnfolding.load_from_checkpoint(pretrain_model_path, strict=False)
    else:
        if(model_name == "crnn"):
            model = CTCTrainedCRNN(
                w2i=train_ds.w2i, i2w=train_ds.i2w, ctc=ctc, use_augmentations=use_augmentations, ds_name=ds_name
            )
        elif(model_name == "fcn" or model_name == "crnnunfolding" or model_name == "cnnt2d" or model_name == "van"):
            model = LightningE2EModelUnfolding(
                w2i=train_ds.w2i, i2w=train_ds.i2w, model=model_name, use_augmentations=use_augmentations, mh=max_h, mw=max_w
            )
        elif(model_name == "ctc_van"):
            model = CTCTrainedVAN(
                w2i=train_ds.w2i, i2w=train_ds.i2w, ctc=ctc, use_augmentations=use_augmentations, ds_name=ds_name
            )
    if(model_name == "crnn"):
        train_ds.width_reduction = model.model.cnn.width_reduction

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=f"{ds_name}_{encoding_type}_{model_name}_{ctc}",
            monitor=metric_to_monitor,
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor=metric_to_monitor,
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        ),
    ]
    trainer = Trainer(
        logger=WandbLogger(
            project=project,
            group=group,
            name=f"{model_name.upper()}-{encoding_type.upper()}-Train-{ds_name}",
            log_model=False,
            entity=entity,
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        deterministic=False,  # If True, raises error saying that CTC loss does not have this behaviour
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
        fast_dev_run=False,  # Set to True to check if everything is working
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    if(model_name == "crnn"):
        model = CTCTrainedCRNN.load_from_checkpoint(
            callbacks[0].best_model_path, ytest_i2w=test_ds.i2w
        )
    elif(model_name == "fcn" or model_name == "crnnunfolding" or model_name == "cnnt2d" or model_name == "van"):
        model = LightningE2EModelUnfolding.load_from_checkpoint(
            callbacks[0].best_model_path, ytest_i2w=test_ds.i2w
        )
    elif(model_name == "ctc_van"):
        model = CTCTrainedVAN.load_from_checkpoint(
            callbacks[0].best_model_path, ytest_i2w=test_ds.i2w
        )
    model.freeze()
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)