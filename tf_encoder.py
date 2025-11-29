import torch
from torch import nn
from tf_encoder_layer import TFEncoderLayer


class TFEncoder(nn.Module):
  def __init__(
      self,
      emb_dim: int,
      d_qkv: int,
      n_layers: int,
      n_heads: int,
      dropout_p: float,
      mlp_dim: int,
      layer_norm_eps: float,
      mask_type: str = 'none',
      mask_size: int = -1,
      cls_token: bool = False,
      inv_temp: float = 1.0,
      ngtk: int = 0,
      mlpa: str = 'relu',
  ):
    super().__init__()
    self.emb_dim = emb_dim
    self.n_layer = n_layers
    self.cls_token = cls_token
    self.transformer = nn.ModuleList(
      [
        TFEncoderLayer(
          emb_dim=emb_dim,
          d_qkv=d_qkv,
          n_heads=n_heads,
          dropout_p=dropout_p,
          mlp_dim=mlp_dim,
          attn_norm_eps=layer_norm_eps if i != 0 else 0,
          ffn_norm_eps=layer_norm_eps,
          norm_first=True,
          mask_type=mask_type,
          mask_size=mask_size,
          cls_token=cls_token,
          inv_temp=inv_temp,
          ngtk=ngtk,
          mlpa=mlpa,
        )
        for i in range(n_layers)
      ]
    )

  def forward(self, x: torch.LongTensor, **kwargs):
    for block in self.transformer[:-1]:
      x = block(x)
    # special operation for last transformer layer.
    # Only use the cls embedding for query in attention
    last_block = self.transformer[-1]
    # If using [CLS] token, use only [CLS] token as query,
    if self.cls_token:
      x = last_block(x, q_idx=0)
    # otherwise, use all token as query, and average the reps across all tokens
    else:
      x = last_block(x).mean(dim=1)
    return x
