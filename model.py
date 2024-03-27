import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from generators import tissue_image_generator, pix2pix_generator

def extract_patches_2ds(x, kernel_size, padding=0, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    channels = x.shape[1]

    x = torch.nn.functional.pad(x, padding)

    # (B, C, H, W)
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])
    # x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1])
    # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])

    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()

    return x


def combine_patches_2d(x, kernel_size, output_shape, stride=1, padding=0, dilation=0):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[3]

    h_dim_out, w_dim_out = output_shape[2:]
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])

    # (B * h_dim_in * w_dim_in, C, kernel_size[0], kernel_size[1])
    x = x.permute(0, 3, 1, 2, 4, 5)
    # x = x.view(-1, channels, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    # (B, C, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    x = x.permute(0, 1, 4, 5, 2, 3)
    # (B, C, kernel_size[0], kernel_size[1], h_dim_in, w_dim_in)
    x = x.contiguous().view(-1, channels * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    x = torch.nn.functional.fold(x, (h_dim_out, w_dim_out), kernel_size=(kernel_size[0], kernel_size[1]),
                                 padding=padding, stride=stride, dilation=dilation)
    # (B, C, H, W)
    return x


class SAFRONModel(nn.Module):

    def __init__(self, generator='pix2pix',
                 mode='train',
                 normalization='instance', activation='leakyrelu-0.2',
                 **kwargs):
        super(SAFRONModel, self).__init__()
        self.mode = mode
        self.generator = generator
        self.context_encoder_block = self.build_context_encoder_block(3)
        if(self.generator == 'pix2pix'):
            self.image_generator = pix2pix_generator(in_channels=3)
        elif(self.generator == 'residual'):
            self.image_generator = tissue_image_generator(input_nc=3,
                                                         output_nc=3,
                                                         n_blocks_global=5,
                                                         n_downsample_global=3,
                                                         ngf=64,
                                                         norm=normalization)
        else:
            raise "Give valid generator name"

        self.context_encoder_block.cuda()
        self.image_generator.cuda()



    def build_context_encoder_block(self, dim):
        layers = []
        layers.append(nn.Conv2d(dim, dim, kernel_size=41))
        layers.append(nn.BatchNorm2d(dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)



    def forward(self, mask, generator_mode = "safron"):

        if(generator_mode == "pix2pix"):
            return self.image_generator(mask)

        if(self.mode == 'train'):
            mask = nn.ConstantPad2d(20, 0)(mask)
            patches = extract_patches_2ds(mask, kernel_size=296, stride=236)
            num_patches = patches.shape[1]
            output_patches = []
            for i in range(0,num_patches):
                output_patches_in = []
                for j in range(0,num_patches):
                    input_mask_patch = patches[0][i][j].unsqueeze(0)
                    patch_pred = self.context_encoder_block(input_mask_patch)
                    patch_pred = self.image_generator(patch_pred)
                    output_patches_in.append(patch_pred)
                output_patches_in = torch.cat(output_patches_in)
                output_patches.append(output_patches_in)
            output_patches = torch.stack(output_patches).unsqueeze(0)
            divider_tensor = torch.ones_like(output_patches).cuda()
            divider_tensor = combine_patches_2d(divider_tensor, kernel_size=256, stride=236, output_shape=(1, 3, 728, 728), dilation=1)
            image_pred = combine_patches_2d(output_patches, kernel_size=256, stride=236, output_shape=(1, 3, 728, 728), dilation=1)
            image_pred = image_pred / divider_tensor
        else:
            patch_pred = self.context_encoder_block(mask)
            image_pred = self.image_generator(patch_pred)

        return image_pred