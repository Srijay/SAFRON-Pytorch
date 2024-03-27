import os
import glob
import argparse
import shutil
from PIL import Image
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--mask_dir", help="path to mask directory",
                    default=r"F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches_2\cycleGAN\himico_tumor\testA")
parser.add_argument("--output_dir", help="path to output folder",
                    default=r"F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches_2\cycleGAN\himico_tumor\output\safron")
parser.add_argument("--model_name", help="path to model",
                    default="./output/ruqayya/model/model.pt")

args = parser.parse_args()
PIL.Image.MAX_IMAGE_PIXELS = 933120000

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def create_image(input_path):
    #create temporary directory
    tmp_dir = "./tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        shutil.rmtree(tmp_dir)

    #determine name and size of image
    im = Image.open(input_path)
    width,height = im.size
    imname = os.path.split(input_path)[1]

    # Create patches from input component mask
    mask_patches_path = os.path.join(tmp_dir, "mask_patches")
    os.system('python ./assistant/SingleImageCropper.py --image_path ' + input_path + ' --output_dir ' + mask_patches_path)

    #Compute output patches using generator
    results_dir = os.path.join(tmp_dir, "results")
    os.system("python main.py --mode test --test_mask_dir " + mask_patches_path + " --test_output_dir " + results_dir)

    #Join patches into single file
    output_path = os.path.join(args.output_dir,imname)
    os.system("python ./assistant/join_images.py --patches_dir " + os.path.join(results_dir,"images") + " --output_file " + output_path + " --im_height " + str(height) + " --im_width " + str(width))

    #Cleanup files
    shutil.rmtree(tmp_dir)
    print("Done")

image_paths = glob.glob(os.path.join(args.mask_dir, "*.png"))
for path in image_paths:
    create_image(path)

