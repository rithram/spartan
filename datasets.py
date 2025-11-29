import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab


# Function to tokenize input string
def ctok(text: str):
  return [
    c for c in text.replace('(', '').replace(')','').split(' ')
    if c != ''
  ]


# Token iterator for vocab generator
def yield_tokens(
    D: pd.DataFrame,
    max_len: int = None,
    PAD: str = None,
    NGTK: int = 0,
    GTK: str = "GTK",
):
  gtks = [] if NGTK == 0 else [f"[{GTK}{i+1}]" for i in range(NGTK)]
  for idx, row in D.iterrows():
    src = ctok(row['Source'])
    pad_len = (
      max(0, max_len - len(src))
      if max_len is not None else 0
    )
    if max_len is None or pad_len == 0:
      yield src + gtks
    else:
      assert PAD is not None
      yield src + [PAD]*pad_len + gtks


def lenupdate(max_len, mask_size):
  ret = max_len
  if mask_size != -1:
    nblocks = max_len // mask_size
    if nblocks * mask_size < max_len:
      ret = (nblocks + 1) * mask_size
    assert ret % mask_size == 0
    print(
      f"Changing max length from {max_len}->{ret}"
    )
  return ret


class ListOpsDataset(Dataset):
  def __init__(
      self,
      fname: dict,
      vocab: Vocab = None,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = None,
      split: str = None,
  ):
    super().__init__()
    self.PAD = "[PAD]"
    self.vocab = vocab
    assert "train" in fname and "val" in fname
    assert split == "train" or split == "val"
    data = pd.read_csv(fname[split], sep='\t')
    # find max sequence length
    seq_lens = [
      len(item) for item in yield_tokens(pd.concat([
        pd.read_csv(fname["train"], sep="\t"),
        pd.read_csv(fname["val"], sep="\t"),
      ]))
    ]
    self.max_length = np.max(seq_lens)
    print(
      f"Sequence lengths ranging from {np.min(seq_lens)}->{self.max_length}"
    )
    # adjust maxlen to match mask_size
    self.max_length = lenupdate(self.max_length, mask_size)
    self.ngtk = ngtk
    # create token vocab if not provided
    if vocab is None:
      print("generating a vocab")
      self.vocab = build_vocab_from_iterator(
        yield_tokens(
          pd.concat([
            pd.read_csv(fname["train"], sep="\t"),
            pd.read_csv(fname["val"], sep="\t"),
          ]),
          max_len=self.max_length,
          PAD=self.PAD,
          NGTK=self.ngtk
        )
      )
    else:
      print("utilizing existing vocab")
    self.NTOKENS = len(self.vocab)
    print(
      f"Created vocab with {len(self.vocab)} tokens "
      f"with {self.PAD} token index: {self.vocab([self.PAD])}\n"
      f" (Max) sequence length: {self.max_length}"
    )
    print("Tokens:", self.vocab.lookup_tokens(range(len(self.vocab))))
    self.X = torch.tensor([
      self.vocab(item) for item in yield_tokens(
        data, max_len=self.max_length, PAD=self.PAD, NGTK=self.ngtk,
      )], dtype=torch.long)
    self.y = torch.tensor(data['Target'], dtype=torch.long)
    print(f"Data: {self.X.shape}, {self.y.shape}")

  def __len__(self):
    return self.X.size(0)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
      sequences, batch_first=True, padding_value=self.vocab(self.PAD)
    )
    return sequences_padded, labels


def lenblocks(lmin, lmax, lstep):
  if lmin == lmax:
    return [lmin]
  if lmin + lstep > lmax:
    return [lmin, lmax]
  lens = np.arange(lmin, lmax, step=lstep).tolist()
  if lens[-1] < lmax:
    lens += [lmax]
  return lens


class ParityCheck(Dataset):
  def __init__(
      self,
      min_length: int,
      max_length: int,
      len_step: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = None,
      split: str = None,
  ):
    super().__init__()
    assert min_length <= max_length
    assert min_length + len_step <= max_length
    self.NLABELS = 2
    self.PAD = 2
    self.NTOKENS = 3 + ngtk
    self.min_length = min_length
    self.max_length = max_length
    # adjust maxlen to match mask_size
    self.max_length_padded = lenupdate(max_length, mask_size)

    self.seq_lens = lenblocks(self.min_length, self.max_length, len_step)
    print(f"Seq lens: {self.seq_lens}")
    nsamples_per_len = (nsamples // len(self.seq_lens))
    self.X = torch.full(
      (nsamples_per_len * len(self.seq_lens), self.max_length_padded + ngtk),
      fill_value=self.PAD, dtype=torch.long
    )
    for i in range(ngtk):
      cidx = self.max_length_padded + i
      val = self.PAD + (i+1)
      self.X[:, cidx] = val
    y = []
    idx = 0
    for i, l in enumerate(self.seq_lens):
      XX = torch.randint(0, 2, (nsamples_per_len, l))
      yy = torch.sum(XX, dim=-1) % 2
      self.X[idx:idx+XX.shape[0], :XX.shape[1]] = XX
      y += [yy]
      idx += XX.shape[0]
    self.y = torch.concat(y).type(torch.long)
    print(
      f"Data: {self.X.shape}, {self.y.shape}, "
      f"(1: {torch.sum(self.y)}/{self.y.size(0)}, "
      f"0: {self.y.size(0)-torch.sum(self.y)}/{self.y.size(0)})"
    )
    self.xlen = self.X.size(1)

  def __len__(self):
    return self.X.size(0)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
      sequences, batch_first=True, padding_value=self.PAD
    )
    return sequences_padded, labels


class UnequalPairs(Dataset):
  def __init__(
      self,
      min_length: int,
      max_length: int,
      len_step: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = None,
      split: str = None,
  ):
    super().__init__()
    assert min_length <= max_length
    assert min_length + len_step <= max_length
    self.NLABELS = 2
    self.PAD = 2
    self.NTOKENS = 3 + ngtk
    self.min_length = min_length
    self.max_length = max_length
    # adjust maxlen to match mask_size
    self.max_length_padded = lenupdate(max_length, mask_size)

    self.seq_lens = lenblocks(self.min_length, self.max_length, len_step)
    print(f"Seq lens: {self.seq_lens}")
    nsamples_per_len = (nsamples // len(self.seq_lens))
    self.X = torch.full(
      (nsamples_per_len * len(self.seq_lens), self.max_length_padded + ngtk),
      fill_value=self.PAD, dtype=torch.long,
    )
    for i in range(ngtk):
      cidx = self.max_length_padded + i
      val = self.PAD + (i+1)
      self.X[:, cidx] = val
    y = []
    idx = 0
    for i, l in enumerate(self.seq_lens):
      XX = torch.randint(0, 2, (nsamples_per_len, l))
      self.X[idx:idx+XX.shape[0], :XX.shape[1]] = XX
      unequal_pairs = torch.logical_xor(XX[:, :-1], XX[:, 1:])
      yy = torch.sum(unequal_pairs, axis=-1) % 2
      y += [yy]
      idx += XX.shape[0]
    self.y = torch.concat(y).type(torch.long)
    print(
      f"Data: {self.X.shape}, {self.y.shape}, "
      f"(1: {torch.sum(self.y)}/{self.y.size(0)}, "
      f"0: {self.y.size(0)-torch.sum(self.y)}/{self.y.size(0)})"
    )
    self.xlen = self.X.size(1)

  def __len__(self):
    return self.X.size(0)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
      sequences, batch_first=True, padding_value=self.PAD
    )
    return sequences_padded, labels


class MissingDuplicate(Dataset):
  def __init__(
      self,
      min_length: int,
      max_length: int,
      len_step: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = None,
      split: str = None,
  ):
    super().__init__()
    assert min_length <= max_length
    assert min_length + len_step <= max_length
    self.NLABELS = 2
    self.MISSING = 2
    self.PAD = 3
    self.NTOKENS = 4 + ngtk
    self.min_length = min_length
    self.max_length = max_length
    # adjust maxlen to match mask_size
    self.max_length_padded = lenupdate(max_length, mask_size)

    self.seq_lens = lenblocks(self.min_length, self.max_length, len_step)
    print(f"Seq lens: {self.seq_lens}")
    nsamples_per_len = (nsamples // len(self.seq_lens))
    self.X = torch.full(
      (nsamples_per_len * len(self.seq_lens), self.max_length_padded + ngtk),
      fill_value=self.PAD, dtype=torch.long,
    )
    for i in range(ngtk):
      cidx = self.max_length_padded + i
      val = self.PAD + (i+1)
      self.X[:, cidx] = val
    y = []
    idx = 0
    for i, l in enumerate(self.seq_lens):
      X = torch.randint(0, 2, (nsamples_per_len, l // 2))
      XX = torch.concat([X, X], axis=-1)
      colflips = torch.randint(0, XX.shape[1], (nsamples_per_len,))
      rows = torch.arange(XX.shape[0])
      y += [XX[(rows, colflips)].clone()]
      XX[(rows, colflips)] = self.MISSING
      self.X[idx:idx+XX.shape[0], :XX.shape[1]] = XX
      idx += XX.shape[0]
    self.y = torch.concat(y).type(torch.long)
    print(
      f"Data: {self.X.shape}, {self.y.shape}, "
      f"(1: {torch.sum(self.y)}/{self.y.size(0)}, "
      f"0: {self.y.size(0)-torch.sum(self.y)}/{self.y.size(0)})"
    )
    self.xlen = self.X.size(1)

  def __len__(self):
    return self.X.size(0)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
      sequences, batch_first=True, padding_value=self.PAD
    )
    return sequences_padded, labels


class NNCH(Dataset):
  def __init__(
      self,
      slength: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = 0,
      task: str = "cycle_navigation",
      split: str = "train",
  ):
    super().__init__()
    assert task in [
      "cycle_navigation",
      "modular_arithmetic_brackets",
      "solve_equation",
      "stack_manipulation",
    ]
    assert split in ["train", "val"]
    assert seed in [
      153476998,
      235510766,
      205084180,
      129959864,
      149896109,
      140439608,
      145017219,
      162518970,
      245293503,
      152674004,
    ]
    self.max_length = slength
    # adjust maxlen to match mask_size
    self.max_length_padded = lenupdate(slength, mask_size)

    fname = (
      f"task:{task}__tlen:{self.max_length}__"
      f"seed:{seed}__nsamples:{nsamples}__{split}.pkl"
    )
    ddir = f"data/nnch/{task}"
    import os
    dpath = os.path.join(ddir, fname)
    assert os.path.exists(dpath)
    print(f"Loading data from:\n{dpath}")

    with open(dpath, "rb") as f:
      ddict = pickle.load(f)
    print(f"Loaded data size: {ddict['input'].shape}, {ddict['output'].shape}")
    assert ddict["input"].shape[0] == ddict["output"].shape[0]
    assert ddict["input"].shape[1] == self.max_length
    self.ONTOKENS = ddict["input"].shape[2]
    self.PAD = ddict["input"].shape[2]
    self.NTOKENS = self.PAD + ngtk + 1
    self.NLABELS = ddict["output"].shape[1]

    self.sequences = torch.tensor(
      ddict["input"].dot(np.arange(self.ONTOKENS)),
      dtype=torch.long
    )
    self.y = torch.tensor(
      ddict["output"].dot(np.arange(self.NLABELS)),
      dtype=torch.long
    )
    print(f"Tokenized data size: {self.sequences.shape}, {self.y.shape}")

    # add pads
    self.extra = torch.full(
      (self.sequences.size(0), self.max_length_padded + ngtk - self.max_length),
      fill_value=self.PAD, dtype=torch.long,
    )
    for i in range(ngtk):
      self.extra[:, self.max_length_padded + i - self.max_length] = self.PAD + i + 1

    self.X = torch.hstack([self.sequences, self.extra])
    self.xlen = self.X.size(1)

    unique, counts = torch.unique(self.y, return_counts=True)
    lstr = (', ').join([f"{l}:{c}/{self.y.size(0)}" for l, c in zip(unique, counts)])
    print(f"Data: {self.X.shape}, {self.y.shape}, ({lstr})")

  def __len__(self):
    return self.X.size(0)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
      sequences, batch_first=True, padding_value=self.vocab(self.PAD)
    )
    return sequences_padded, labels


class CycNav(NNCH):
  def __init__(
      self,
      min_length: int,
      max_length: int,
      len_step: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = 0,
      split: str = "train",
  ):
    assert min_length == max_length
    assert len_step == 0
    super().__init__(
      slength=min_length,
      nsamples=nsamples,
      mask_size=mask_size,
      ngtk=ngtk,
      seed=seed,
      task="cycle_navigation",
      split=split,
    )

class MAB(NNCH):
  def __init__(
      self,
      min_length: int,
      max_length: int,
      len_step: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = 0,
      split: str = "train",
  ):
    assert min_length == max_length
    assert len_step == 0
    super().__init__(
      slength=min_length,
      nsamples=nsamples,
      mask_size=mask_size,
      ngtk=ngtk,
      seed=seed,
      task="modular_arithmetic_brackets",
      split=split,
    )

class SolEq(NNCH):
  def __init__(
      self,
      min_length: int,
      max_length: int,
      len_step: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = 0,
      split: str = "train",
  ):
    assert min_length == max_length
    assert len_step == 0
    super().__init__(
      slength=min_length,
      nsamples=nsamples,
      mask_size=mask_size,
      ngtk=ngtk,
      seed=seed,
      task="solve_equation",
      split=split,
    )

class StackMan(NNCH):
  def __init__(
      self,
      min_length: int,
      max_length: int,
      len_step: int,
      nsamples: int,
      mask_size: int = -1,
      ngtk: int = 0,
      seed: int = 0,
      split: str = "train",
  ):
    assert min_length == max_length
    assert len_step == 0
    super().__init__(
      slength=min_length,
      nsamples=nsamples,
      mask_size=mask_size,
      ngtk=ngtk,
      seed=seed,
      task="stack_manipulation",
      split=split,
    )

if __name__ == "__main__":

  mask_sizes = [-1, 7]
  ngs = [0, 3]
  for Data in [CycNav, MAB, SolEq, StackMan]:
    for (ms, ng) in zip(mask_sizes, ngs):
      print(f"Testing {Data.__name__} mask size: {ms}, #GTK: {ng}")
      dset = Data(
        min_length=40,
        max_length=40,
        len_step=0,
        nsamples=5000,
        mask_size=ms,
        ngtk=ng,
        seed=129959864,
        split="train",
      )
      print(dset.X[:2, :10])
      print(dset.y[:5])
      dl = DataLoader(dset, batch_size=5, shuffle=True)
      for X, y in dl:
        print(f"Inputs: {X.shape}")
        print(f"Outputs: {y.shape}")
        break
      print("-"*40)
    print("="*40)
  
  for Data in [UnequalPairs, ParityCheck, MissingDuplicate]:
    for (ms, ng) in zip(mask_sizes, ngs):
      print(f"Testing {Data.__name__} mask size: {ms}, #GTK: {ng}")
      dset = Data(
        min_length=8,
        max_length=10,
        len_step=1,
        nsamples=10,
        mask_size=ms,
        ngtk=ng,
        seed=129959864,
        split="train",
      )
      print(dset.X)
      print(dset.y)
      dl = DataLoader(dset, batch_size=5, shuffle=True)
      for X, y in dl:
        print(f"Inputs: {X.shape}")
        print(f"Outputs: {y.shape}")
        break
      print("-"*40)
    print("="*40)

  mask_sizes = [-1, 7]
  ngs = [0, 3]
  fname = {
    "train": "data/listops/D10-A10-l500-L600-5k-2k-2k/basic_train.tsv",
    "val": "data/listops/D10-A10-l500-L600-5k-2k-2k/basic_val.tsv",
  }    
  import os
  assert os.path.exists(fname["train"])
  assert os.path.exists(fname["val"])
  vocab = None
  for (ms, ng) in zip(mask_sizes, ngs):
    print(f"Testing LISTOPS mask size: {ms}, # global tokens: {ng}")
    dset = ListOpsDataset(
      fname, vocab=vocab, mask_size=ms, ngtk=ng, split="train"
    )
    dl = DataLoader(dset, batch_size=5, shuffle=True)
    # vocab = dset.vocab
    for X, y in dl:
      print(f"Inputs: {X.shape}")
      print(f"Outputs: {y.shape}")
      print(X[:2, :5])
      print(y[:5])
      break
    print("-"*40)
  print("="*40)
