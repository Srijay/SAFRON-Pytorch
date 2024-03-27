import glob
import os
from PIL import Image
import numpy as np
import time
import PIL
import matplotlib.pyplot as plt
import argparse

patchsize = 296
stride = 236

PIL.Image.MAX_IMAGE_PIXELS = 933120000

def CropImage(image_path,output_dir,pad):
    image_name = os.path.split(image_path)[1].split('.')[0]
    im = Image.open(image_path)

    size = im.size
    new_size = (size[0]+40,size[1]+40)

    if(pad):
        new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
        new_im.paste(im, ((new_size[0] - size[0]) // 2,
                              (new_size[1] - size[1]) // 2))
        #plt.imshow(new_im)
        #plt.show()
    else:
        new_im = im

    width, height = new_im.size

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):
        while (right < width):
            left = x
            top = y
            right = left + patchsize
            bottom = top + patchsize
            if (right > width):
                offset = right - width
                right -= offset
                left -= offset
            if (bottom > height):
                offset = bottom - height
                bottom -= offset
                top -= offset
            im_crop = new_im.crop((left, top, right, bottom))
            im_crop_name = image_name + "_" + str(left) + "_" + str(top) + ".png"
            output_path = os.path.join(output_dir, im_crop_name)
            im_crop.save(output_path)
            x += stride
        x = 0
        right = 0
        y += stride

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", help="path to image to crop",
                    default=r"F:\Datasets\himico\train_A\B-1986096_B3_106872_54168_338.png")
parser.add_argument("--output_dir", help="path to output folder",
                    default=r"F:\Datasets\himico\train_patches")
parser.add_argument("--pad", type=int, default=1, help="pad the image borders")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

CropImage(args.image_path,args.output_dir,args.pad)

print("--- %s seconds ---" % (time.time() - start_time))