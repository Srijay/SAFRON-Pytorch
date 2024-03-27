import os
import glob
import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import matplotlib
import time
import argparse

PIL.Image.MAX_IMAGE_PIXELS = 933120000

output_dir = r'F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches_2\cycleGAN\himico_tumor\output\pix2pixhd'
input_dir = r'C:\Users\Srijay\Desktop\Projects\safron_pytorch\ruqayya\pix2pixHD-HIMICO\results\himico\test_20\test_B_syn'
image_names = [filename for filename in os.listdir(input_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
image_names = ['_'.join(st.split('_')[:5]) for st in image_names]
image_names = list(set(image_names))
patch = 1024

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def join_images(imname,height,width):
    start_time = time.time()
    paths = glob.glob(os.path.join(input_dir,imname+"*.jpg"))
    image = np.zeros((height,width,3))
    count_masks = np.zeros((height,width,3))
    k=0
    for path in paths:
        imname = os.path.split(path)[1].split(".")[0]
        imname_splt = imname.split("_")
        y,x = int(imname_splt[-4]),int(imname_splt[-3])
        if (x+patch) > width:
            x = width - patch
        if (y+patch) > height:
            y = height - patch
        img = Image.open(path)
        img = np.asarray(img)
        image[x:x+patch,y:y+patch,:] += img
        count_masks[x:x+patch,y:y+patch,:]+=1.0
        k+=1

    count_masks = count_masks.clip(min=1)

    image = image/count_masks

    image = image/255.0

    matplotlib.image.imsave(os.path.join(output_dir,imname+".png"), image)

    print("--- %s seconds ---" % (time.time() - start_time))

    print("Done")

parser = argparse.ArgumentParser()
parser.add_argument("--im_height", default=4096, type=int, help="image height")
parser.add_argument("--im_width", default=4096, type=int, help="image width")

args = parser.parse_args()


for imname in image_names:
    join_images(imname,args.im_height,args.im_width)