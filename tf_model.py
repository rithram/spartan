import math
import sys
import torch
from torch import nn, Tensor
from tf_encoder import TFEncoder

class TFClassifier(nn.Module):

  def __init__(
      self,
      ntoken: int,
      nclass: int,
      maxlen: int,
      d_emb: int,
      d_qkv: int,
      d_mlp: int,
      nlayers: int,
      nheads: int,
      dropout: float = 0.001,
      layer_norm_eps: float = 1e-05,
      mask_type: str = 'none',
      mask_size: int = -1,
      cls_token: bool = True,
      inv_temp: float = 1.0,
      ngtk: int = 0,
      mlpa: str = 'relu',
  ):
    super().__init__()
    self.model_type = 'TFClassifier'
    self.pos_encoder = PositionalEncoding(d_emb, dropout, max_len=maxlen)
    self.embedding = nn.Embedding(ntoken, d_emb)
    self.d_model = d_emb
    self.cls_token = cls_token
    self.encoder = TFEncoder(
      emb_dim=d_emb,
      d_qkv=d_qkv,
      n_layers=nlayers,
      n_heads=nheads,
      dropout_p=dropout,
      mlp_dim=d_mlp,
      layer_norm_eps=layer_norm_eps,
      mask_type=mask_type,
      mask_size=mask_size,
      cls_token=cls_token,
      inv_temp=inv_temp,
      ngtk=ngtk,
      mlpa=mlpa,
    )
    self.linear = nn.Linear(d_emb, nclass)
    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.linear.bias.data.zero_()
    self.linear.weight.data.uniform_(-initrange, initrange)

  def forward(self, src: Tensor) -> Tensor:
    """
    Arguments:
      src: Tensor, shape ``[batch_size, seq_len]``

    Returns:
      output Tensor of shape ``[batch_size, ntoken]``
    """
    src = self.embedding(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    enc_output = self.encoder(src)
    assert (
      (self.cls_token and enc_output.shape[1] == 1) or
      ((not self.cls_token) and len(enc_output.shape) == 2)
    ), (
      f"Encoder output size: {enc_output.shape}, input size: {src.shape}"
    )
    assert src.shape[0] == enc_output.shape[0], (
      f"Expected batch size: {src.shape[0]}, got {enc_output.shape[0]}"
    )
    assert src.shape[-1] == enc_output.shape[-1], (
      f"Expected d-model: {src.shape[-1]}, got {enc_output.shape[-1]}"
    )
    out = enc_output[:, 0, :] if self.cls_token else enc_output
    assert len(out.shape) == 2, (f"Out shape: {out.shape}")
    out = self.linear(out)
    return out


class PositionalEncoding(nn.Module):

  def __init__(
      self, d_model: int,
      dropout: float = 0.1,
      max_len: int = 5000
  ):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor) -> Tensor:
    """
    Arguments:
      x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
    """
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)
