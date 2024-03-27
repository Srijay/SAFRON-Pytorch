# example of calculating the frechet inception distance in Keras
import numpy
import glob
import random
import os
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize
import numpy as np

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    num_images = images1.shape[0]
    avg_fid = 0
    for i in range(0,num_images):
        # calculate activations
        act1 = model.predict(np.expand_dims(images1[i], axis=0))
        act2 = model.predict(np.expand_dims(images2[i], axis=0))
        # act1 = numpy.concatenate((act1,act1),axis=0)
        # calculate mean and covariance statistics
        # mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        # mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        mu1 = act1.mean(axis=0)
        mu2 = act2.mean(axis=0)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        # covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        # if iscomplexobj(covmean):
        #     covmean = covmean.real
        # calculate score
        fid = ssdiff #+ trace(sigma1 + sigma2 - 2.0 * covmean)
        avg_fid += fid
    avg_fid = avg_fid/num_images
    return avg_fid

folder_1 = "F:/Datasets/conic/CoNIC_Challenge/challenge/valid/images"
folder_2 = "F:/Datasets/conic/CoNIC_Challenge/challenge/valid/results/conic_residual_17_hovernet_30epochs/pred_image"

scale = 0
size = 256
max_file_num = 500

paths1 = os.listdir(folder_1)
paths2 = os.listdir(folder_2)

common_file_names = list(set(paths1) & set(paths2))

if(len(common_file_names)>max_file_num):
    common_file_names = random.sample(common_file_names,max_file_num)

length = len(common_file_names)

print("Number of files to be processed: ",length)

def get_images(folder,file_names):
    images = []
    for fname in file_names:
        img = Image.open(os.path.join(folder,fname))
        img = numpy.asarray(img)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        images.append(img)
    return images

images_random = randint(0, 255, length*size*size*3)
images_random = images_random.reshape((length,size,size,3))
images_random = images_random.astype('float32')

images_real = get_images(folder_1,common_file_names)
images_gen = get_images(folder_2,common_file_names)

images_real = numpy.array(images_real)
images_gen = numpy.array(images_gen)

# prepare the inception v3 model
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(size,size,3))

# pre-process images
images_real = preprocess_input(images_real)
images_gen = preprocess_input(images_gen)
images_random = preprocess_input(images_random)

# fid = calculate_fid(model, images_real, images_real)
# print('FID (same): %.3f' % fid)

fid = calculate_fid(model, images_real, images_random)
print('FID (random): %.3f' % fid)

fid = calculate_fid(model, images_real, images_gen)
print('FID (predicted) : %.3f' % fid)