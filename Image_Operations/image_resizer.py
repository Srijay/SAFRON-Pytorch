from PIL import Image
import os, sys
import glob
import matplotlib
import matplotlib.pyplot as plt

path = r"F:\Datasets\CRAG_LabServer\train\cropped\1024\images"
outdir = r"F:\Datasets\CRAG_LabServer\train\cropped\1024\images_resized_728"
resize_len = 728

if not os.path.exists(outdir):
        os.makedirs(outdir)

dirs = os.listdir(path)

image_paths = glob.glob(os.path.join(path,"*.png"))

for path in image_paths:
    imname = os.path.split(path)[1]
    savepath = os.path.join(outdir,imname)
    im = Image.open(path)
    imResize = im.resize((resize_len,resize_len))
    imResize.save(savepath)
    # matplotlib.image.imsave(savepath, imResize)