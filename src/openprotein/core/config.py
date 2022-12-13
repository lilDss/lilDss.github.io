from typing import *

import os

class DataConfig(object):
    """

    """
    def __init__(self) -> None:
        """

        """
        self.seed = 1
        self.file_path = os.path.dirname(__file__)
        self.path = "./resources/uniref50/valid"
        self.dataset_name = "rice"
        self.data_process_method = "load_every_data_different_len"
        self.max_len = 99
        self.batch_size = 128
        self.is_shuffle = True
        self.is_drop_last = True
        self.num_workers = int(os.cpu_count() / 2)
        self.pin_memory = True
        self.ori_data_size = 1
        self.tar_data = None
        self.tar_data_size = 0.1
        self.test_size = 0.2

