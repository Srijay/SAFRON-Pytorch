from math import log10, sqrt
from PIL import Image
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

import glob
import random
from skimage.color import rgb2hed, hed2rgb
import matplotlib.pyplot as plt

#winner: conic_residual_19_nogconvoraffinetransform_hovernet, score = 0.9556
#selected due to high fid: conic_residual_19_nogconvoraffinetransform_hovernet_fid81, score = 0.95527

#on with and without graph conv:
#winner: conic_residual_22, score: 0.95528
#another: conic_residual_22_hovernet, score: 0.95287
#another: conic_residual_17, score: 0.95621
#another: conic_residual_17_hovernet, score: 0.95461
#another: conic_residual_2, score: 0.9563
#another: conic_residual_2_hovernet_30epochs, score: 0.9545
#another: conic_residual_2_hovernet_fullepochs, score: 0.955639
#another: conic_residual_5, score: 0.95485
#another: conic_residual_5_hovernet, score:
#another: conic_residual_5_hovernet_45epochs, score: 0.9551

real_folder = r"F:\Datasets\lizard\CoNIC_Challenge\challenge\valid\images_971"
syn_folder = r"F:\Datasets\lizard\CoNIC_Challenge\challenge\valid\results\conic_residual_19_nogconvoraffinetransform_hovernet\pred_image"

def remove_alpha_channel(img):
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def display_stains(ihc_hed):
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(hed2rgb(ihc_hed))
    ax[0].set_title("Original image")

    ax[1].imshow(ihc_h)
    ax[1].set_title("Hematoxylin")

    ax[2].imshow(ihc_e)
    ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

    ax[3].imshow(ihc_d)
    ax[3].set_title("DAB")

    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()
    plt.show()

def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(img1,img2):

    img1 = img1/255.0
    img2 = img2/255.0

    # Image.fromarray(img1).save("img1.jpeg")
    # Image.fromarray(img2).save("img2.jpeg")

    img1 = rgb2hed(img1)
    img2 = rgb2hed(img2)

    # display_stains(img1)
    # display_stains(img2)

    score_h = ssim(img1[:, :, 0],img2[:, :, 0]) #h channel ssim
    score_e = ssim(img1[:, :, 1],img2[:, :, 1]) #e channel ssim
    score_d = ssim(img1[:, :, 2],img2[:, :, 2]) #d channel ssim

    average = (score_h + score_e + score_d)/3

    return score_h, score_e, score_d,average


image_paths = glob.glob(os.path.join(real_folder, "*.png"))
max_image_num = 971
psnr = 0
total_ssim_h = 0
total_ssim_e = 0
total_ssim_d = 0
total_ssim_avg = 0
image_paths = random.sample(image_paths, max_image_num)

for path in image_paths:
    imname = os.path.split(path)[1]
    syn_imname = imname
    # syn_imname = imname.split(".")[0]+"_synthesized_image.jpg" #for pix2pixhd
    im1 = remove_alpha_channel(np.asarray(Image.open(os.path.join(real_folder,imname))))
    im2 = remove_alpha_channel(np.asarray(Image.open(os.path.join(syn_folder,syn_imname))))
    # psnr_c = PSNR(im1, im2)
    score_h, score_e, score_d, average = SSIM(im1, im2)
    # ssim_c,_ = compare_ssim(im1, im2, full=True)
    total_ssim_h+=score_h
    total_ssim_e+=score_e
    total_ssim_d+=score_d
    total_ssim_avg+=average
    # psnr+=psnr_c

l = len(image_paths)
print("Total images ",l)
print("synthetic data path: ",syn_folder)
# print("Average PSNR value is dB ",psnr/l)
print("Average SSIM H score is ",total_ssim_h/l)
print("Average SSIM E score is ",total_ssim_e/l)
print("Average SSIM D score is ",total_ssim_d/l)
print("Average SSIM AVG score is ",total_ssim_avg/l)