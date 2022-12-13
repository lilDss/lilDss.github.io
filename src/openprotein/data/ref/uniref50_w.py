from numpy import delete
from tqdm import tqdm
import os
seq = {}
# 53625855
with open('../uniref50.fasta', 'r') as f:
    for line in tqdm(f):
        if line.startswith('>'):
            name = line.replace('>', '').split()[0]
            seq[name] = ''
        else:
            seq[name] += line.replace('\n', '').strip()
# 52207272
refine_seq = {}

for name, sequence in tqdm(seq.items()):
    if len(sequence) <= 1022:
        refine_seq[name] = sequence

del seq


import torch
from torch.utils.data import random_split
train_seq = {}
valid_seq = {}
test_seq = {}
refine_keys = list(refine_seq.keys())
length = len(refine_keys)
train_length, valid_length = int(length * 0.8) , int(length * 0.1)
train_keys, valid_keys, test_keys = random_split(refine_keys, [train_length, valid_length, length-train_length-valid_length], generator=torch.Generator().manual_seed(42))
print(len(train_keys))


import lmdb
import pickle as pkl
splits = ['train', 'valid', 'test']
keys = {'train': train_keys, 'valid': valid_keys, 'test': test_keys}
for split in splits:
    length = []
    env = lmdb.open(f'../../../resources/uniref/{split}', map_size=107374182400)
    with env.begin(write=True) as txn:
        for idx, key in tqdm(enumerate(keys[split])):
            txn.put(str(idx).encode(), refine_seq[key].encode())
            length.append(len(refine_seq[key]))
        print(length[:5])
        txn.put('data_lens'.encode(), pkl.dumps(length))
        txn.put('data_size'.encode(), str(idx+1).encode())
        print(idx+1)
pass

    # env = lmdb.open(f'../../resources/uniref/uniref', map_size=107374182400)
    # with env.begin(write=True) as txn:
    #     for idx, key in tqdm(enumerate(refine_keys)):
    #         txn.put(str(idx).encode(), refine_seq[key].encode())
    #         length.append(len(refine_seq[key]))
    #     print(length[:5])
    #     txn.put('data_lens'.encode(), pkl.dumps(length))
    #     txn.put('data_size'.encode(), str(idx+1).encode())
    #     print(idx+1)
