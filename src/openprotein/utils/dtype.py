from collections import UserList
from typing import *
from numpy import array
from struct import pack, unpack

def convert_to_bytes(obj: Union[int, str, List[Union[str, int]]]):
    # recurrence
    if isinstance(obj, int):
        return convert_to_bytes(str(obj))
    if isinstance(obj, str):
        if int(obj) < 0:
            raise ValueError(f'The number must be greater than zero, get {obj}')
        return obj.encode()
    elif isinstance(obj, list):
        return list(map(lambda x: convert_to_bytes(x), obj))
    else:
        raise TypeError(f"The data must be int or str or list[Union[str, int]], get {obj}")

def convert_to_str(obj: Union[bytes, List[bytes]]):
    # recurrence
    if isinstance(obj, bytes):
        return obj.decode()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return list(map(lambda x: convert_to_str(x), obj))
    else:
        raise TypeError(f"Error {obj}. The data type must be bytes or list[bytes]")

class Array(UserList):
    def __init__(self, data):
        super(Array, self).__init__(data)

    def append(self, item):

        self.data.append(item)


    def insert(self, i, item):
        self.data.insert(i, item)

    def extend(self, other):
        if isinstance(other, UserList):
            self.data.extend(other.data)
        else:
            self.data.extend(other)