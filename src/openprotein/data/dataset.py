import os
import logging

from openprotein.core import DataConfig, Components

import lmdb

from functools import partial
from typing import *
from struct import pack, unpack
import numpy as np

from openprotein.data.process import MaskedConverter
from openprotein.utils.dtype import convert_to_str, convert_to_bytes

# TODO: use attnotion to modify
from openprotein.utils.utils import Cache


# class DatasetFactory(type):
#     def __init__(self, args):
#         if args.backend == "pt":
#             return PTDataFactory(args)
class Data(metaclass=Components):
    """
    A basic interface for data, data subclasses need to implement this interface

    Args:
        path (str): path for the dataset

    """
    def __init__(self, path):
        self._data = DataFactory.load(self, path)

    def __str__(self) -> str:
        return self.__name__

    def get_data(self):
        """
        A interface to return the dataset

        Args:
            No Args

        Returns:
            Dataset
        """
        # raise NotImplement
        return self._data.get_data()

    def get_dataloader(self, batch_size: int = None, shuffle: bool = False, sampler=None,
                       batch_sampler=None,
                       num_workers: int = 0, collate_fn=None,
                       pin_memory: bool = False, drop_last: bool = False):
        """
        A interface to return the dataloader of the current dataset

        Args:
            batch_size(int, optional): how many samples per batch to load (default: 1).
            shuffle(bool, optional): set to True to have the data reshuffled at every epoch (default: False).
            sampler(Sampler or Iterable, optional): defines the strategy to draw samples from the dataset.
            batch_sampler(Sampler or Iterable, optional): like sampler, but returns a batch of indices at a time.
            num_workers(int, optional): how many subprocesses to use for data loading. (default: 0)
            collate_fn(Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
            pin_memory(bool, optional):
                If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
            drop_last (bool, optional): (default: False)
                set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
                If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.

        Returns:
            DataLoader
        """
        # raise NotImplemented
        return self._data.get_dataloader(batch_size, shuffle, sampler,
                       batch_sampler,
                       num_workers, collate_fn,
                       pin_memory, drop_last)


class DataFactory(object):
    """
    According to the specific data of the operating environment,load DataFactory (PTDataFactory or MSDataFactory)

    Args:
        path (str): path for the dataset file.

    Returns:
        Data
    """
    # def __init__(self, path: str):
    #     # super().__init__(b)
    #     self.convert(path)

    @staticmethod
    def load(cls: Data, path: str) -> Data:
        """
        Factory method, according to the operating environment and parameters, load the specific factory
        """
        if cls._backend == "pt":
            return PTDataFactory(path)
        elif cls._backend == "ms":
            raise NotImplemented

class PTDataFactory(Data):
    """
    Factory for generating PyTorch datasets

    Args:
        path (str): path for the dataset file.

    Raises:
        ImportError: torch is not installed
    """
    # use hook to modify behavior
    # TODO: use annotation to modify behavior
    try:
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        logging.error("No module named torch")
        raise ImportError("No module named torch") from e

    def __init__(self, path: str):
        # self.args = args
        # self.__dict__.update(args.__dict__)
        self._dataset = self.PTDataset(path)

    # @Cache
    def get_data(self) -> Dataset:
        """
        Get the PyTorch implementation of the current dataset

        Args:
            No args.

        Returns:
             Dataset
        """
        return self._dataset

    # def _get_data(self):
    #     self._dataset = self.PTDataset(self.path)

    # @Cache
    def get_dataloader(self, batch_size: int = None, shuffle: bool = False, sampler=None,
                       batch_sampler=None,
                       num_workers: int = 0, collate_fn=None,
                       pin_memory: bool = False, drop_last: bool = False) -> DataLoader:
        """
        Get the PyTorch implementation of Dataloader for the current dataset

        Args:
            batch_size (int, optional): how many samples per batch to load (default: 1).
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch (default: False).
            sampler (Sampler or Iterable, optional): defines the strategy to draw samples from the dataset.
            batch_sampler (Sampler or Iterable, optional): like sampler, but returns a batch of indices at a time.
            num_workers (int, optional): how many subprocesses to use for data loading. (default: 0)
            collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
            pin_memory (bool, optional):
                If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
            drop_last (bool, optional): (default: False)
                set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
                If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.

        Returns:
            DataLoader
        """
        # bs = batch_size if batch_size else self.batch_size
        return self.DataLoader(self._dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                               num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                               drop_last=drop_last)

    class PTDataset(Dataset):
        """
        PyTorch's Dataset implementation class

        Args:
            lmdb_path(str): path for a lmdb dataset
        """

        def __init__(self, lmdb_path: str, categories: List[str] = ["train", "valid", "test"]):
            # self._lmdb_path = lmdb_path
            # self._categories = categories # TODO: 不区分train, valid, test
            self._load_lmdb(lmdb_path)
            self._data_size = int(self._cur.get("data_size".encode()).decode())

        def _load_lmdb(self, lmdb_path):
            """
            Load the dataset from lmdb path

            Args:
                lmdb_path: path for a lmdb dataset

            Raises:
                FileNotFoundError: if no available file is found.
            """
            # read_lmdb = partial(lmdb.open, create=False, subdir=True, readonly=True, lock=False)
            try:
                self._data = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                self._cur = self._data.begin(write=False).cursor()
                logging.info(f"load {self.__class__} sucessfully")
            except Exception as e:
                logging.warning(e)
                raise FileNotFoundError(e) from e

        def __len__(self):
            return self._data_size

        def __getitem__(self, index: Union[str, int, slice, list]):
            # TODO: next support slice, single or multiple
            if isinstance(index, slice):
                start = index.start if index.start else 0
                stop = index.stop if index.stop else self._data_size
                step = index.step if index.step else 1
                index = list(range(start, stop, step))
                # return convert_to_str(self._cur.getmulti(index))
                return self._get_multi_data(index)
            elif isinstance(index, list):
                return self._get_multi_data(index)
            else:
                return self._cur.get(str(index).encode()).decode()
            # index = convert_to_bytes(index)
            # return convert_to_str(self._cur.getmulti(index))

        def _get_multi_data(self, index: list) -> Tuple:
            """
            2-tuples containing (index, data), use index to get multiple sets of data

            Args:
                index (list): index used to fetch the value

            Returns:
                a tuple which contains the value(data) indexed by index
            """
            index = self._convert_to_bytes(index)
            result = self._convert_to_str(self._cur.getmulti(index))
            return list(zip(*result))[1]

        def _convert_to_bytes(self, obj: Union[int, str, List[Union[str, int]]]):
            """
            convert the input to bytes
            Args:
                obj (int or str or List[Union[str, int]]): the input to be converted

            Returns:
                the bytes type of the input

            Raises:
                ValueError: The number must be greater than zero
                TypeError: The data must be int or str or list[Union[str, int]]
            """
            if isinstance(obj, int):
                return self._convert_to_bytes(str(obj))
            if isinstance(obj, str):
                if int(obj) < 0:
                    raise ValueError(f'The number must be greater than zero, get {obj}')
                return obj.encode()
            elif isinstance(obj, list):
                return list(map(lambda x: self._convert_to_bytes(x), obj))
            else:
                raise TypeError(f"The data must be int or str or list[Union[str, int]], get {obj}")

        def _convert_to_str(self, obj: Union[bytes, List[bytes]]):
            """
            convert the input to string
            Args:
                obj (bytes or list[bytes]): the input to be converted

            Returns:
                the String type of the input

            Raises:
                TypeError: The data type must be bytes or list[bytes]
            """
            if isinstance(obj, bytes):
                return obj.decode()
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return list(map(lambda x: self._convert_to_str(x), obj))
            else:
                raise TypeError(f"Error {obj}. The data type must be bytes or list[bytes]")
