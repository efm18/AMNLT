import torch
import os
import gc

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from transformers import TrOCRProcessor, PreTrainedTokenizerFast

from models.model import CTCTrainedTrOCR
from my_utils.dataset import CTCDataset
from my_utils.data_preprocessing import trocr_batch_preparation
from data.config import DS_CONFIG


def test_trocr(
    ds_name: str,
    checkpoint_path: str,
    encoding_type="char",
    batch_size=1,
):
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🔍 Testing TrOCR on dataset: {ds_name}")
    print(f"📦 Loading model from checkpoint: {checkpoint_path}")

    # Load tokenizer
    tokenizer_path = f"trocr_utils/tokenizer_trocr_custom/{ds_name}"
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"Custom tokenizer not found at: {tokenizer_path}")

    base_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    processor = TrOCRProcessor(feature_extractor=base_processor.feature_extractor, tokenizer=tokenizer)

    # Load test dataset
    test_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["test"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        model_name="trocr",
        train=False,
        encoding_type=encoding_type,
    )
    test_ds.processor = processor

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=trocr_batch_preparation,
    )

    # Load model
    model = CTCTrainedTrOCR.load_from_checkpoint(checkpoint_path, ds_name=ds_name, tokenizer_path=tokenizer_path)

    # Run test
    trainer = Trainer(precision="16-mixed")
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    import fire
    fire.Fire(test_trocr)
