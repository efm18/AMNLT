import fire
import json
import torch
from data_amnlt import AMNLTDataset
from smt_trainer import SMT_Trainer
from dan_trainer import DAN_Trainer

from AMNLT.configs.smt_dan_config.ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_path, dataset_name):
    
    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))
    
    datamodule = AMNLTDataset(config=config.data)

    max_height, max_width = datamodule.train_set.get_max_hw()
    max_len = datamodule.train_set.get_max_seqlen()

    model_wrapper = SMT_Trainer(maxh=max_height, maxw=max_width, maxlen=max_len, 
                                out_categories=len(datamodule.train_set.w2i), padding_token=datamodule.train_set.w2i["<pad>"], 
                                in_channels=1, w2i=datamodule.train_set.w2i, i2w=datamodule.train_set.i2w, 
                                d_model=256, dim_ff=256, num_dec_layers=8)
    
    wandb_logger = WandbLogger(project='DAN_AMNLT', group=dataset_name, name=f"DAN_{dataset_name}", log_model=False)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{dataset_name}/", filename=f"{dataset_name}_DAN", 
                                   monitor="val_CER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=10000, 
                      check_val_every_n_epoch=5, 
                      logger=wandb_logger, callbacks=[checkpointer],
                      precision="16-mixed")
    
    trainer.fit(model_wrapper,datamodule=datamodule)

    model = SMT_Trainer.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, datamodule=datamodule)

def launch(config_path, dataset_name):
    main(config_path, dataset_name)

if __name__ == "__main__":
    fire.Fire(launch)