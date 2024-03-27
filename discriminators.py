import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(256,256),
               layout_dim=0):
    super(PatchDiscriminator, self).__init__()
    #input_dim = 1 + layout_dim
    input_dim = 3

    arch = 'I%d,%s' % (input_dim, arch)

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x)


class Pix2PixDiscriminator(nn.Module):

  def __init__(self, in_channels=3):
    super(Pix2PixDiscriminator, self).__init__()

    def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
      """Returns downsampling layers of each discriminator block"""
      layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
      if normalization:
        layers.append(nn.InstanceNorm2d(out_filters))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(

      #If model loading failed, try this discriminator and keep kernel size as 5
      # *discriminator_block(in_channels, 16, normalization=False),
      # *discriminator_block(16, 32),
      # *discriminator_block(32, 64),
      # *discriminator_block(64, 128),
      # *discriminator_block(128, 256),
      # #nn.ZeroPad2d((1, 0, 1, 0)),
      # nn.Conv2d(256, 1, 5, padding=1, bias=False),
      # #nn.ReLU()

      *discriminator_block(in_channels, 64, normalization=False),
      *discriminator_block(64, 128),
      *discriminator_block(128, 256),
      *discriminator_block(256, 512), #Added to safronize framework
      *discriminator_block(512, 512, stride=1),
      # nn.ZeroPad2d((1, 0, 1, 0)),
      nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False),
      # nn.Sigmoid() #It was not there when trained with residual generator so turn it off while its inference
    )

  def forward(self, mask, img):
    # Concatenate image and condition image by channels to produce input
    img_input = torch.cat((mask, img), 1)
    output = self.model(img_input)
    return output



