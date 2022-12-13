import unittest
import os
import argparse

from torch import optim
from torch.optim.lr_scheduler import StepLR

# from openprotein.core.config import DataConfig
from openprotein.data import Uniref, MaskedConverter, Alphabet
# from openprotein.piplines import Train
from openprotein.models import Esm1b
from openprotein.piplines import Accuracy

class MainTest(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        proteinseq_toks = {
            'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                     'X', 'B', 'U', 'Z', 'O', '.', '-']
        }
        self.converter = MaskedConverter.build_convert(proteinseq_toks)
        alphabet = Alphabet.build_alphabet(proteinseq_toks)
        self.data = Uniref("./resources/uniref50/valid")

        args = {'num_layers': 33, 'embed_dim': 1280, 'logit_bias': True, 'ffn_embed_dim': 5120, 'attention_heads': 20, 'max_positions':1024, 'emb_layer_norm_before': True, 'checkpoint_path': None}
        args = argparse.Namespace(**args)
        self.model = Esm1b(args, alphabet)
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        self.metrics = Accuracy()

    def test_main(self):
        f = lambda x: self.converter(x)
        dl = self.data.get_dataloader(collate_fn=f)
        # train
        self.model.cuda("1")

        for origin_tokens, masked_tokens, target_tokens in dl:
            result = self.model(masked_tokens)['logits']
            print(self.metrics(target_tokens, result))
            break

