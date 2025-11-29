import torch
from torch import nn
from multihead_attention import MultiHeadAttention


class TFEncoderLayer(nn.Module):
  def __init__(
      self,
      emb_dim: int,
      n_heads: int,
      attn_norm_eps: float,
      ffn_norm_eps: float,
      mlp_dim: int,
      d_qkv: int = None,
      dropout_p: float = None,
      norm_first: bool = True,
      mask_type: str = 'none',
      mask_size: int = -1,
      cls_token: bool = False,
      inv_temp: float = 1.0,
      ngtk: int = 0,
      mlpa: str = 'relu',
  ):
    super().__init__()
    self.norm_first = norm_first

    self.attention = MultiHeadAttention(
      d_emb=emb_dim,
      n_heads=n_heads,
      d_qkv=d_qkv,
      dropout_p=dropout_p,
      mask_type=mask_type,
      mask_size=mask_size,
      cls_token=cls_token,
      inv_temp=inv_temp,
      ngtk=ngtk,
    )

    ffn_activations = {
      'relu': nn.ReLU(inplace=True),
      'gelu': nn.GELU(),
      'mish': nn.Mish(inplace=True),
    }
      
    self.ffn = nn.Sequential(
      nn.Linear(emb_dim, mlp_dim),
      nn.Dropout(dropout_p),
      ffn_activations[mlpa],
      nn.Linear(mlp_dim, emb_dim),
    )

    self.norm_attn = nn.LayerNorm(
      emb_dim,
      eps=attn_norm_eps,
      elementwise_affine=False,
    )

    self.norm_ffn = nn.LayerNorm(
      emb_dim,
      eps=ffn_norm_eps,
      elementwise_affine=False,
    )

    self.drop = nn.Dropout(dropout_p)

  def forward(
      self,
      src: torch.FloatTensor,
      # this is for the trick used in FT.
      # Use only the CLS token as query in last block of tf.
      q_idx: int = None,
  ):
    if self.norm_first:
      x = self.norm_attn(src)

      if q_idx is None:
        attn = self.attention(x, x, x)
        x = src + self.drop(attn)
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
      else:
        q = x[:, q_idx : q_idx + 1, :]
        attn = self.attention(q, x, x, last_block=True)
        x = src[:, q_idx : q_idx + 1, :] + self.drop(attn)
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
    else:
      attn = self.attention(src, src, src)
      x = self.norm_attn(src + self.drop(attn))
      x = self.norm_ffn(x + self.drop(self.ffn(x)))
    return x
