import os
import logging
import lmdb
import numpy as np
import pickle as pkl
from typing import Sequence, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq import utils
from fairseq.data import FairseqDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from openprotein.utils import Alphabet, set_cpu_num
from .data_process import MaskedConverter

logger = logging.getLogger(__name__)


class Uniref50MLMDataset(FairseqDataset):

    def __init__(self, data_dir, split, alphabet) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.alphebet = alphabet
        self.batch_converter = MaskedConverter(alphabet)
        self.data_path = os.path.join(data_dir, 'uniref50/train')

        self.prompt_toks = []

    def open_lmdb(self):
        self.env = lmdb.open(self.data_path, create=False, subdir=True, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.data_size = int(self.txn.get('data_size'.encode()).decode())
        self.data_lens = pkl.loads(self.txn.get('data_lens'.encode()))
        
    def __getitem__(self, index):
        sequence = self.txn.get(str(index).encode()).decode()
        return sequence

    def __len__(self):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        return self.data_size

    def collater(self, raw_batch: Sequence[Dict]):
        origin_tokens, masked_tokens, masked_targets = self.batch_converter(raw_batch, prompt_toks=self.prompt_toks)
        return origin_tokens, masked_tokens, masked_targets, self.prompt_toks
    
    def size(self, index):
        return self.data_lens[index]
    
    def num_tokens(self, index):
        return self.data_lens[index]

    def num_tokens_vec(self, indices):
        return np.array([self.num_tokens(index) for index in indices])


@dataclass
class Uniref50MLMTaskConfig(FairseqDataclass):
    data: str = field(default=MISSING)

@register_task("uniref50_mlm", dataclass=Uniref50MLMTaskConfig)
class Uniref50MLMTask(FairseqTask):
    cfg: Uniref50MLMTaskConfig
    """Task for training masked language models (e.g., BERT, RoBERTa)"""

    def __init__(self, cfg: Uniref50MLMTaskConfig, alphabet):
        super().__init__(cfg)
        self.alphabet = alphabet

    @classmethod
    def setup_task(cls, cfg: Uniref50MLMTaskConfig, **kwargs):
        set_cpu_num(4)
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        alphabet = Alphabet.build_alphabet()
        logger.info(f"Alphabet: {len(alphabet)} types")
        return cls(cfg, alphabet)

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        self.datasets[split] = Uniref50MLMDataset(self.cfg.data, split, self.alphabet)


    @property
    def source_dictionary(self):
        return self.alphabet

    @property
    def target_dictionary(self):
        return self.alphabet