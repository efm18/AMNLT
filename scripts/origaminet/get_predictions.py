import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import gin

from AMNLT.utils.origami_utils.utils import CTCLabelConverter
from AMNLT.models.origami_model.cnv_model import OrigamiNet, ginM
from test import validation
import AMNLT.utils.origami_utils.ds_load as ds_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rSeed(sd):
    import random
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

def load_model(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model

def test(opt):
    gin.parse_config_file(opt.gin)
    rSeed(ginM('manualSeed'))

    test_data_path = ginM('test_data_path')
    test_data_list = ginM('test_data_list')
    test_batch_size = ginM('test_batch_size')
    workers = ginM('workers')
    AMP = ginM('AMP')

    valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=workers,
        collate_fn=ds_load.SameTrCollate
    )

    model = OrigamiNet()
    model = model.to(device)
    model.eval()

    if opt.checkpoint:
        model = load_model(opt.checkpoint, model)

    converter = CTCLabelConverter(valid_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    class DummyOpts:
        def __init__(self):
            self.eval_output_dir = opt.eval_output_dir or os.path.dirname(opt.checkpoint)
            self.rank = 0
            self.DDP = False

    parO = DummyOpts()
    opt.eval_output_dir = parO.eval_output_dir
    os.makedirs(opt.eval_output_dir, exist_ok=True)

    with torch.no_grad():
        test_loss, accuracy, norm_ED, tot_ED, bleu, preds_str, labels, infer_time = validation(
            model, criterion, valid_loader, converter, opt, parO
        )

    print(f"\nValidation complete")
    print(f"Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, norm_ED: {norm_ED:.4f}, CER: {tot_ED:.4f}, BLEU: {bleu*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', required=True, help='Path to .gin config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--eval_output_dir', default=None, help='Where to save predictions.txt')

    opt = parser.parse_args()
    test(opt)
