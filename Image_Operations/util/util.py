from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os

from skimage.io import imread
from tiatoolbox.utils.transforms import imresize
import cv2
from skimage.morphology import remove_small_objects
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


kernel_size = 7
kernel = np.ones((kernel_size, kernel_size), np.uint8)
min_size = 500  # minimum size of objects to keep (in pixels)
def preprocess_mask(tissue_mask_path, misalignment_mask_path, mask_type, output_size=[], isdilate=False):
    tissue_mask = imread(tissue_mask_path)
    if len(tissue_mask.shape) > 2:
        tissue_mask = tissue_mask[:, :, 0]

    if mask_type == 'gland':
        tissue_mask = tissue_mask>0
    elif mask_type == 'tumor':
        tissue_mask = tissue_mask > 128
        tissue_mask = cv2.erode(tissue_mask.astype(np.uint8), kernel, iterations=1)>0
        tissue_mask = remove_small_objects(tissue_mask, min_size)
        tissue_mask = cv2.dilate(np.uint8(tissue_mask), kernel, iterations=1)
        # plt.imshow(tissue_mask)
        # plt.show()

    if(misalignment_mask_path):
        misalign_mask = imread(misalignment_mask_path)>0
        if len(misalign_mask.shape) > 2:
            misalign_mask = misalign_mask[:, :, 0]
    tissue_mask = imresize(tissue_mask, output_size=output_size)
    if (misalignment_mask_path):
        misalign_mask = imresize(misalign_mask, output_size=output_size)
        tissue_mask[misalign_mask > 0] = 0
    tissue_mask = np.array(tissue_mask > 0, dtype=np.uint8)

    if isdilate:
        tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=2)
    # plt.imshow(tissue_mask)
    # plt.imshow(misalign_mask, alpha=0.5)
    # plt.show()
    return tissue_mask

def get_coordinates(wsi_reader, mask_reader, patch_size=256):
    resolution = dict(units='baseline', resolution=1.0)
    wsi_proc_shape = wsi_reader.slide_dimensions(**resolution)

    (patch_inputs, patch_outputs) = PatchExtractor.get_coordinates(
        image_shape=wsi_proc_shape,
        patch_input_shape=[patch_size, patch_size],
        patch_output_shape=[patch_size, patch_size],
        stride_shape=[patch_size, patch_size],
    )
    print(patch_inputs)
    # if mask_reader is not None:
    #     sel = SemanticSegmentor.filter_coordinates(mask_reader, patch_outputs, **resolution)
    #     patch_outputs = patch_outputs[sel]
    #     patch_inputs = patch_inputs[sel]
    print(patch_inputs)
    print('----------------')
    return patch_inputs, patch_outputs
