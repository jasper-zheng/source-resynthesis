import torch
import torchaudio
from torch.utils.data import Dataset
import random
import numpy as np
import os

from glob import glob

class Datasets(Dataset):
    '''
       mix_path: file path of mixed audio (type: str)
       src_path: file path of ground truth sources (type: [str,...])
       sr: sample rate (type: int)
       chunk_size: framesize (type: int)
    '''

    def __init__(self, mix_path=None, src_path=None, sr=8000, chunk_size=32000):
        super(Datasets, self).__init__()
        self.mix_list = []
        for filename in glob(f'{mix_path}/*'):
          if filename[-4:] == '.wav':
            f,_ = torchaudio.load(filename)
            if f.shape[-1] > chunk_size:
              self.mix_list.append(filename)
        self.mix_list.sort()

        self.src_lists = []
        for p in src_path:
          src_list = []
          for filename in glob(f'{p}/*'):
            if filename[-4:] == '.wav':
              f,_ = torchaudio.load(filename)
              if f.shape[-1] > chunk_size:
                src_list.append(filename)
          src_list.sort()
          self.src_lists.append(src_list)

        print(f'loaded {len(self.mix_list)} mix\nloaded sources:')
        for l in self.src_lists:
          print(f'\t- {len(l)}')

        self.sr = sr
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):

        mix, sr = torchaudio.load(self.mix_list[idx])
        assert sr == self.sr
        src = [torchaudio.load(src_list[idx])[0] for src_list in self.src_lists]

        assert mix.shape[-1] >= self.chunk_size
        if mix.shape[-1] == self.chunk_size:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, mix.shape[-1] - self.chunk_size)

        return{
            'mix': mix[:,rand_start:rand_start+self.chunk_size],
            'src': [s[:,rand_start:rand_start+self.chunk_size] for s in src]
        }
