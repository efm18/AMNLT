import torch
import torch.distributed as dist

from copy import deepcopy
from collections import OrderedDict

from itertools import chain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    def __init__(self, character):
        dict_character = list(character)
        self.dict = {char: i + 1 for i, char in enumerate(dict_character)}  # 0 is reserved for CTC blank
        self.character = ['[blank]'] + dict_character

    def encode(self, text_batch):
        """Expects either a list of token strings or a list of lists of token strings"""
        #print(text_batch)
        if isinstance(text_batch[0], str):
            # Single sequence
            length = [len(text_batch)]
            flat_tokens = [self.dict[token] for token in text_batch]
        else:
            # Batch of sequences
            length = [len(seq) for seq in text_batch]
            flat_tokens = [self.dict[token] for seq in text_batch for token in seq]

        return (
            torch.tensor(flat_tokens, dtype=torch.int32).to(device),
            torch.tensor(length, dtype=torch.int32).to(device)
        )


    def decode(self, text_index, length):
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i] < len(self.character):
                    char_list.append(self.character[t[i]])

            texts.append(char_list)  # list of tokens

            index += l
        return texts



class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        return self.sum / float(self.n_count) if self.n_count != 0 else 0


class Metric(object):
    def __init__(self, parO, name=''):
        self.name = name
        self.sum = torch.tensor(0., device=device).double()
        self.n = torch.tensor(0., device=device)
        self.pO = parO

    def update(self, val):
        if self.pO.DDP:
            rt = val.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= dist.get_world_size()
            self.sum += rt.detach().cpu().double()
        else:
            self.sum += val.detach().double()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n.double()


class ModelEma:
    def __init__(self, model, decay=0.9999, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path, mapl=None):
        checkpoint = torch.load(checkpoint_path, map_location=mapl)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                name = 'module.' + k if self.ema_has_module and not k.startswith('module') else k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            print("=> Loaded state_dict_ema")
        else:
            print("=> Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model, num_updates=-1):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        _cdecay = min(self.decay, (1 + num_updates) / (10 + num_updates)) if num_updates >= 0 else self.decay

        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * _cdecay + (1. - _cdecay) * model_v)
