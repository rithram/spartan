import torch
from torch import nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()


def mask_cls_no_key(
    attn: torch.FloatTensor,
    mask_nnz: int,
    cls_token: bool,
    mask_cache = None,
):
  assert cls_token
  mask = torch.ones_like(attn).type(torch.bool)
  mask[:,:,1:,0] = 0
  return F.softmax(attn.masked_fill(mask == 0, -1e15), dim=-1), mask


def mask_window(
    attn: torch.FloatTensor,
    mask_nnz: int,
    cls_token: bool,
    mask_cache = None,
    ngtk: int = 0,
):
  assert ngtk == 0 or not cls_token
  mask = mask_cache
  if mask_cache is None:
    mask = torch.zeros_like(attn).type(torch.bool)
    if cls_token:
      mask[:,:,0,:] = 1
    nrows = mask.size(-2) - (ngtk+int(cls_token))
    ncols = mask.size(-1) - (ngtk+int(cls_token))
    assert nrows == ncols
    if ngtk > 0:
      # global tokens attend to all
      mask[:, :, nrows:nrows+ngtk, :] = 1
      # all attend to global tokens
      mask[:, :, :, ncols:ncols+ngtk] = 1
    l = (mask_nnz - 1) // 2
    u = mask_nnz - 1 - l
    rows = []
    cols = []
    for i in range(int(cls_token), int(cls_token) + nrows):
      lb = max(int(cls_token), i - l)
      ub = min(i + u + 1, int(cls_token)+ncols)
      if lb == int(cls_token):
        ub = lb + mask_nnz
      if ub == int(cls_token)+ncols:
        lb = ub - mask_nnz
      rows += [[i]]
      cols += [[jj for jj in range(lb,ub)]]
    mask[:, :, rows, cols] = 1
    mask_cache = mask
  return F.softmax(attn.masked_fill(mask == 0, -1e15), dim=-1), mask_cache


def mask_block_local(
    attn: torch.FloatTensor,
    mask_nnz: int,
    cls_token: bool,
    mask_cache = None,
    ngtk: int = 0,
):
  assert ngtk == 0 or not cls_token
  mask = mask_cache
  if mask_cache is None:
    mask = torch.zeros_like(attn).type(torch.bool)
    if cls_token:
      mask[:,:,0,:] = 1
    nrows = mask.size(-2) - (ngtk+int(cls_token))
    ncols = mask.size(-1) - (ngtk+int(cls_token))
    assert nrows == ncols
    if ngtk > 0:
      # global tokens attend to all
      mask[:, :, nrows:nrows+ngtk, :] = 1
      # all attend to global tokens
      mask[:, :, :, ncols:ncols+ngtk] = 1
    for i in range(int(cls_token), int(cls_token)+nrows, mask_nnz):
      j = min(i + mask_nnz, int(cls_token)+ncols)
      mask[:, :, i:j, i:j] = 1
    mask_cache = mask
  return F.softmax(attn.masked_fill(mask == 0, -1e15), dim=-1), mask_cache


def mask_topk(
    attn: torch.FloatTensor,
    mask_nnz: int,
    cls_token: bool,
    mask_cache = None,
    ngtk: int = 0,
):
  assert ngtk == 0
  mask = torch.zeros_like(attn).type(torch.bool)
  if cls_token:
    mask[:,:,0,:] = 1
  iidx = int(cls_token)
  _, ind = torch.topk(attn[:, :, iidx:, iidx:], mask_nnz, dim=-1)
  ridx = [[j + iidx] * mask_nnz for j in range(mask.size(-2)-iidx)]
  aridx = torch.tensor([ridx for _ in range(mask.size(1))])
  hidx = torch.ones_like(aridx).type(torch.int)
  for i in range(mask.size(1)):
    hidx[i, :, :] = i
  for mm, ii in zip(mask, ind):
    mm[hidx, aridx, ii + iidx] = 1
  return F.softmax(attn.masked_fill(mask == 0, -1e15), dim=-1), None


def mask_botk(
    attn: torch.FloatTensor,
    mask_nnz: int,
    cls_token: bool,
    mask_cache = None,
    ngtk: int = 0,
):
  assert ngtk == 0
  mask = torch.zeros_like(attn).type(torch.bool)
  if cls_token:
    mask[:,:,0,:] = 1
  iidx = int(cls_token)
  _, ind = torch.topk(-attn[:, :, iidx:, iidx:], mask_nnz, dim=-1)
  ridx = [[j + iidx] * mask_nnz for j in range(mask.size(-2)-iidx)]
  aridx = torch.tensor([ridx for _ in range(mask.size(1))])
  hidx = torch.ones_like(aridx).type(torch.int)
  for i in range(mask.size(1)):
    hidx[i, :, :] = i
  for mm, ii in zip(mask, ind):
    mm[hidx, aridx, ii + iidx] = 1
  return F.softmax(attn.masked_fill(mask == 0, -1e15), dim=-1), None

def mask_rand(
    attn: torch.FloatTensor,
    mask_nnz: int,
    cls_token: bool,
    mask_cache = None,
    ngtk: int = 0,
):
  assert ngtk == 0
  assert not cls_token
  assert len(attn.shape) == 4
  mask = torch.zeros_like(attn).type(torch.bool)
  iidx = 0
  ind = torch.randint(
    0, attn.size(-1), (attn.size(0), attn.size(1), attn.size(2), mask_nnz)
  )
  ridx = [[j + iidx] * mask_nnz for j in range(mask.size(-2)-iidx)]
  aridx = torch.tensor([ridx for _ in range(mask.size(1))])
  hidx = torch.ones_like(aridx).type(torch.int)
  for i in range(mask.size(1)):
    hidx[i, :, :] = i
  for mm, ii in zip(mask, ind):
    mm[hidx, aridx, ii + iidx] = 1
  return F.softmax(attn.masked_fill(mask == 0, -1e15), dim=-1), None


mask_attn_dict = {
  'none': None,
  'windowed': mask_window,
  'blklocal': mask_block_local,
  'topk': mask_topk,
  'botk': mask_botk,
  'rnd': mask_rand,
}
