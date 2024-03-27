# This script computes average dice index scores between two
# sets of class outputs or images in 2-d with each pixel
# representing the class
import numpy
import glob
import random
import os
from PIL import Image
import numpy as np
import scipy.io


def get_images(folder,file_names, replacepng=False):
    images = []
    for fname in file_names:
        if(replacepng):
            fname = fname.replace(".png","")
        img = np.load(os.path.join(folder,fname))
        if len(img.shape)==3 and img.shape[2]==2:
            img = img[:,:,1]
        images.append(img)
    return images


def compute_dice_score(x,y):
    k=1
    denom = (np.sum(x) + np.sum(y))
    if(denom==0):
        return 1
    return np.sum(x[y==k])*2.0 / denom


def evaluate_dice_score(images1,images2,class_id='all'):

    dice_score = 0
    l = len(images1)

    for i in range(0,len(images1)):

        img1 = images1[i].copy()
        img2 = images2[i].copy()

        # img1_color = colored_images_from_classes(img1,256)
        # img2_color = colored_images_from_classes(img2,256)
        # save_numpy_image_FLOAT(img1_color, os.path.join(gt_color_output, str(i) + ".png"))
        # save_numpy_image_FLOAT(img2_color, os.path.join(pred_color_output, str(i) + ".png"))

        if(class_id=='all'):
            img1[img1>0] = 1.0
            img2[img2>0] = 1.0
        else:
            img1[img1 != class_id] = 0
            img1[img1 == class_id] = 1.0
            img2[img2 != class_id] = 0
            img2[img2 == class_id] = 1.0
            # save_numpy_image_FLOAT(img1, os.path.join(gt_bw_output, str(i) + ".png"))
            # save_numpy_image_FLOAT(img2, os.path.join(pred_bw_output, str(i) + ".png"))

        dice_score += compute_dice_score(img1,img2)

    dice_score=dice_score/l*1.0
    return dice_score


def colored_images_from_classes(x,image_size):
    color_dict = {
        1: [0, 0, 0],  # neutrophil  : black
        2: [0, 255, 0],  # epithelial : green
        3: [255, 255, 0],  # lymphocyte : Yellow
        4: [255, 0, 0],  # plasma : red
        5: [0, 0, 255],  # eosinophil : Blue
        6: [255, 0, 255],  # connectivetissue : fuchsia
        0: [255, 255, 255]  # Background : white
    }
    k = np.zeros((image_size, image_size, 3))
    l = image_size
    for i in range(0, l):
        for j in range(0, l):
            k[i][j] = color_dict[x[i][j]]
    return k/255.0


folder_1 = "F:/Datasets/conic/CoNIC_Challenge/challenge/valid/labels"
folder_2 = "F:/Datasets/conic/CoNIC_Challenge/challenge/valid/results/conic_residual_2_hovernet_30epochs/pred_image"

image_names = [file for file in os.listdir(folder_2) if file.endswith('.npy')]

length = len(image_names)

print("Number of files to be processed: ",length)

images_real = get_images(folder_1,image_names,replacepng=True)
images_gen = get_images(folder_2,image_names)

class_dict = {
        1: 'neutrophil',  # neutrophil  : black
        2: 'epithelial',  # epithelial : green
        3: 'lymphocyte',  # lymphocyte : Yellow
        4: 'plasma',  # plasma : red
        5: 'eosinophil',  # eosinophil : Blue
        6: 'connectivetissue',  # connectivetissue : fuchsia
        0: 'background'  # Background : white
    }

# class_dict = {
#         1: 'Neoplastic',  # neutrophil  : black
#         2: 'Inflammatory',  # epithelial : green
#         3: 'Soft',  # lymphocyte : Yellow
#         4: 'Dead',  # plasma : red
#         5: 'Epithelial',  # eosinophil : Blue
#         0: 'background'  # Background : white
#     }

classes = [1,2,3,4,5, 6]

for c in classes:
    dice_score = evaluate_dice_score(images_real,images_gen,c)
    print("Dice score of class ",class_dict[c]," is ",dice_score)

print("Overall Dice score is ",evaluate_dice_score(images_real,images_gen,'all'))