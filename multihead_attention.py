import torch
from torch import nn
import torch.nn.functional as F
import math

from attn_masks import mask_attn_dict


def scaled_dot_prod(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    inv_temp: float = 1.0,
    dropout_p: int = 0,
    mask_attn = None,
    mask_size:int = -1,
    cls_token: bool = False,
    mask_cache = None,
    ngtk: int = 0,
):
  attn = inv_temp * (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
  mask = None
  if mask_attn is None:
    attn = F.softmax(attn, dim=-1)
  else:
    bsz = attn.size(0)
    cache = None
    if mask_cache is not None:
      if isinstance(mask_cache, dict):
        cache = mask_cache
      else:
        cache = mask_cache[:bsz]
    attn, mask = mask_attn(attn, mask_size, cls_token, cache, ngtk)
  if dropout_p > 0:
    attn = F.dropout(attn, p=dropout_p)
  attn_out = attn @ v
  return attn_out, (mask if mask_cache is None else mask_cache)


class MultiHeadAttention(nn.Module):
  def __init__(
      self,
      d_emb: int,
      n_heads: int,
      d_qkv: int,
      dropout_p: float,
      mask_type: str = 'none',
      mask_size: int = -1,
      cls_token: bool = False,
      inv_temp: float = 1.0,
      ngtk: int = 0,
  ):
    super().__init__()
    self.n_heads = n_heads
    self.dropout_p = dropout_p
    self.d_qkv = d_emb // n_heads if d_qkv == 0 or d_qkv is None else d_qkv
    self.d_emb = d_emb
    self.mask_type = mask_type
    self.mask_attn = mask_attn_dict[mask_type]
    self.mask_cache = None
    self.mask_size = mask_size
    self.cls_token = cls_token
    self.inv_temp = inv_temp
    self.ngtk = ngtk

    self.W_q = nn.Linear(d_emb, self.d_qkv * n_heads)
    self.W_k = nn.Linear(d_emb, self.d_qkv * n_heads)
    self.W_v = nn.Linear(d_emb, self.d_qkv * n_heads)
    self.W_o = nn.Linear(self.d_qkv * n_heads, d_emb)

  # reshape q, k or v to (d_batch, n_heads, d_seq, d_qkv)
  # for input to scaled_dot_prod
  def _reshape_qkv(self, qkv: torch.FloatTensor):
    assert (
      qkv.ndim == 3
    ), (
      "input to MHA must have 3 dimensions: (d_batch, d_seq, d_qkv) where "
      "d_qkv may be d_model or an arbirary value"
    )
    d_batch, d_seq = qkv.shape[:2]
    return qkv.view(
      d_batch, d_seq, self.n_heads, self.d_qkv
    ).transpose(1, 2)

  def forward(
      self,
      q_in: torch.FloatTensor,
      k_in: torch.FloatTensor,
      v_in: torch.FloatTensor,
      last_block = False,
  ):
    assert (
      q_in.shape[2] == self.d_emb and k_in.shape[2] == self.d_emb
      and v_in.shape[2] == self.d_emb
    )
    d_batch, d_seq = q_in.shape[:2]
    # q,k,v: (d_batch, d_seq, d_emb) -> (d_batch, n_heads, d_seq, d_qkv)
    q = self._reshape_qkv(self.W_q(q_in))
    k = self._reshape_qkv(self.W_k(k_in))
    v = self._reshape_qkv(self.W_v(v_in))

    # If using [CLS] token, last block does not need mask
    # Otherwise, last block uses mask
    MA = None if (self.cls_token and last_block) else self.mask_attn
    attn, self.mask_cache = scaled_dot_prod(
      q, k, v, self.inv_temp, self.dropout_p, MA, self.mask_size,
      self.cls_token, self.mask_cache, ngtk=self.ngtk,
    )

    # attn: (d_batch, d_seq, n_heads, d_qkv) -> (d_batch, d_seq, d_emb)
    attn = attn.transpose(2, 1).reshape(
      d_batch, d_seq, self.n_heads * self.d_qkv
    )
    return self.W_o(attn)
