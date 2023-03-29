# coding: utf-8
import os.path as osp
import pickle

import lmdb
import six
from torch.utils.data import Dataset

try:
    import mc
except ImportError as E:
    pass

from PIL import Image


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class LMDB(Dataset):
    def __init__(self, root, transforms=None, length=40, no_label=False, **kwargs):
        super().__init__()
        self.root = root
        print(self.root)
        self.length = length
        self.env = None
        self.transforms = transforms
        self.no_label = no_label

    def _init_db(self):
        self.env = lmdb.open(self.root, subdir=osp.isdir(self.root),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()

        txn = self.txn
        key: bytes = u"{}".format(index).encode('ascii')
        byteflow = txn.get(key)
        unpacked = loads_data(byteflow)

        # load img.
        imgbuf = unpacked[0]
        label = unpacked[1]

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        if self.transforms != None:
            img = self.transforms(img)

        if self.no_label:
            return img
        else:
            return img, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

    def get_length(self):
        return self.length

    def get_sample(self, idx):
        return self.__getitem__(idx)
