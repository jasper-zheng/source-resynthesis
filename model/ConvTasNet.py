from asteroid.masknn.convolutional import TDConvNet, TDConvNetpp, DCUNetComplexEncoderBlock, DCUNetComplexDecoderBlock
from torch.nn import Conv1d, ConvTranspose1d, PReLU, Sequential
import torch

class Encoder(torch.nn.Module):
  def __init__(self, in_c, out_c, filters_len = 16, layer_num = 3) -> None:
      super().__init__()
      layers = []
      for _ in range(layer_num-1):
          layers.append(Conv1d(out_c, out_c, kernel_size=3, stride=1, padding=1))
          layers.append(PReLU())
      self.sequential = Sequential(
          Conv1d(in_c, out_c, filters_len, stride=filters_len//2), *layers
      )

  def forward(self, x):
      '''
        x: [Batch, in_channels, T]
        return: [Batch, filters_num, t]
      '''
      x = self.sequential(x)
      return x

class Decoder(torch.nn.Module):
  def __init__(self, in_c, out_c, filters_len = 16, layer_num = 3) -> None:
     super().__init__()
     layers = []
     for _ in range(layer_num-1):
         layers.append(ConvTranspose1d(in_c, in_c, kernel_size=3, stride=1, padding=1))
         layers.append(PReLU())
     self.sequential = Sequential(
         *layers,
         ConvTranspose1d(in_c, out_c, kernel_size=filters_len, stride=filters_len//2, bias=True)
     )

  def forward(self, x):
    '''
      x: [Batch, filters_num, t]
      return: [Batch, out_channels, T]
    '''
    x = self.sequential(x)
    return x

class ConvTasNet(torch.nn.Module):
  def __init__(self, in_channels, out_channels, masks_num, filters_num, filters_len = 16, tdcnpp = True, enc_layer_num=3, dec_layer_num=3,
               **tcn_kwargs) -> None:
    super().__init__()
    self.source_num = masks_num
    self.tdcnpp = tdcnpp
    self.filters_num = filters_num
    self.encoder = Encoder(in_channels, filters_num, filters_len, enc_layer_num)
    self.separation = TDConvNetpp(filters_num, masks_num, filters_num, **tcn_kwargs) if tdcnpp else TDConvNet(filters_num, masks_num, filters_num, **tcn_kwargs)
    self.decoder = Decoder(filters_num, out_channels, filters_len, dec_layer_num)
      
  def forward(self, x, return_mask = False):
    '''
      x: [Batch, Channels, T]
    '''
    assert x.dim() == 3
    
    w = self.encoder(x)
    # [Batch, filters_num, t]

    if self.tdcnpp:
        m, _ = self.separation(w)
    else:
        m = self.separation(w)
    # [Batch, mask_num, filters_num, t]

    assert m.dim() == 4
    assert m.shape[1] == self.source_num

    ws = w.unsqueeze(1).repeat([1,self.source_num,1,1])
    # [Batch, mask_num, filters_num, t]

    masked = m * ws
    # [Batch, mask_num, filters_num, t]

    masked = masked.unbind(dim=1)
    # [[Batch, filters_num, t], ...]: len = mask_num
    
    sources = [self.decoder(m) for m in masked]
    if return_mask:
        return sources, m
    else:
        return sources
