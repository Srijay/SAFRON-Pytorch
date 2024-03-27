'''This script is used to create training dataset for pix2pix, cycleGAN and pix2pixHD.
Mask can be used to extract patches from ROIs only and it could be for glandular or tumor regions.
Example: python cloud_workspace/RF_Projects/General/pytorch-CycleGAN-and-pix2pix/RA/create_data/create_train_dataset.py --patch_size=728 --mask_type=tumor --split=train --num_patch_per_wsi=500 --dataset_type=pix2pixHD --num_workers=10
'''

import os
# OPENSLIDE_PATH = r"E:\\Dropbox\\PhD_Work\\PythonVE\\openslide-win64-20171122\\bin"
# os.add_dll_directory(OPENSLIDE_PATH)
import glob
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader
import matplotlib.pyplot as plt
from util.util import preprocess_mask, get_coordinates
import numpy as np
import argparse
from multiprocessing import Pool
import cv2
from scipy.ndimage import binary_fill_holes

def save_image(image, tissue_thresh):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image < 240
    mask = binary_fill_holes(mask)
    if np.sum(mask) / (mask.shape[0] * mask.shape[1]) > tissue_thresh:
        return True

def process_file(file_path, args):
    # setting save path
    if args.dataset_type == 'pix2pix':
        save_path = args.save_path + '/' + args.dataset_type + '/himico_' + args.mask_type + '/' + args.split
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif args.dataset_type == 'cycleGAN':
        split = args.split
        if split == 'valid':
            split = 'test'
        save_pathA = args.save_path + '/' + args.dataset_type + '/himico_' + args.mask_type + '/' + split + 'A'
        if not os.path.exists(save_pathA):
            os.makedirs(save_pathA)
        save_pathB = args.save_path + '/' + args.dataset_type + '/himico_' + args.mask_type + '/' + split + 'B'
        if not os.path.exists(save_pathB):
            os.makedirs(save_pathB)
    elif args.dataset_type == 'pix2pixHD':
        split = args.split
        save_pathA = args.save_path + '/' + args.dataset_type + '/himico_' + args.mask_type + '/' + split + '_A'
        if not os.path.exists(save_pathA):
            os.makedirs(save_pathA)
        save_pathB = args.save_path + '/' + args.dataset_type + '/himico_' + args.mask_type + '/' + split + '_B'
        if not os.path.exists(save_pathB):
            os.makedirs(save_pathB)

    gland_mask_dir = args.gland_mask_dir + '/' + args.dataset_name
    misalignment_mask_dir = args.misalignment_mask_dir + '/' + args.dataset_name

    basename = os.path.basename(file_path).split('_HE')
    basename = '_'.join(basename[:-1])
    print(basename)

    # read mIHC wsi data
    wsi_reader_he = WSIReader.open(input_img=r'F:\Datasets\himico\test\B-2032566_B5_HE\B-2032566_B5_HE.mrxs')
    # temp = glob.glob(os.path.join(args.mIHC_wsi_dir, basename + '*.tiff'))
    # print(basename + ' ' + str(len(temp)) + ' mIHC images found!')
    # wsi_reader_mihc = WSIReader.open(input_img=glob.glob(os.path.join(args.mIHC_wsi_dir, basename + '*.tiff'))[0])
    wsi_reader_mihc = WSIReader.open(input_img=r'F:\Datasets\himico\test\B-2032566_B5_HE\B-2032566_B5_CDX2p_MUC2y_MUC5g_CD8dab.tiff')

    gland_mask_path = r'F:\Datasets\himico\test\B-2032566_B5_HE\B-2032566_B5_HE.png'
    # misalignment_mask_path = glob.glob(os.path.join(misalignment_mask_dir, basename + '*.png'))[0]
    misalignment_mask_path = None
    mask = preprocess_mask(gland_mask_path, misalignment_mask_path, args.mask_type, output_size=wsi_reader_he.info.level_dimensions[6], isdilate=True)
    mask_reader = VirtualWSIReader(mask)
    mask_reader.info = wsi_reader_he.info

    # # visualize mask overlaid on HE image
    # # he_thumbnail = wsi_reader_he.read_region(location=(0, 0), level=6, size=wsi_reader_he.info.level_dimensions[6])
    # # mask_thumbnail = mask_reader.read_region(location=(0, 0), level=6, size=wsi_reader_he.info.level_dimensions[6])
    # # plt.imshow(he_thumbnail)
    # # plt.imshow(mask_thumbnail*255, alpha=0.5)
    # # plt.show()

    # get coordinates
    patch_inputs, patch_outputs = get_coordinates(wsi_reader_he, mask_reader, patch_size=(args.patch_size * 2) + 8)
    print(basename + ' ' + str(patch_inputs.shape[0]) + ' patches identified!')

    # randomly select patches
    if patch_inputs.shape[0] > args.num_patch_per_wsi:
        print('Randomly selecting ' + str(args.num_patch_per_wsi) + ' patches...')
        idx = np.random.choice(patch_inputs.shape[0], args.num_patch_per_wsi, replace=False)
        patch_inputs = patch_inputs[idx, :]

    for iPatch in range(0, patch_inputs.shape[0]):
        bbox = patch_inputs[iPatch]

        # extract H&E and mIHC patches
        he_patch = wsi_reader_he.read_rect(location=bbox[:2], size=(args.patch_size, args.patch_size),
                                           resolution=args.resolution,
                                           units=args.unit)

        is_save = save_image(he_patch, args.tissue_thresh)
        if not is_save:
            continue

        bbox = [int(x / 2) for x in bbox]
        mihc_patch = wsi_reader_mihc.read_rect(location=bbox[:2], size=(args.patch_size, args.patch_size),
                                               resolution=args.resolution,
                                               units=args.unit)

        # visualize patches in subplots
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(mihc_patch)
        # ax[1].imshow(stain_mask_patch)
        # plt.show()

        # save patches
        location = str(bbox[0]) + '_' + str(bbox[1])
        if args.dataset_type == 'pix2pix':
            combine_img = np.zeros((args.patch_size, args.patch_size * 2, 3), dtype=np.uint8)
            combine_img[:, 0:args.patch_size, :] = he_patch
            combine_img[:, args.patch_size:, :] = mihc_patch
            plt.imsave(os.path.join(save_path, basename + '_' + location + '_' + str(iPatch) + '.png'), combine_img)
        elif args.dataset_type == 'cycleGAN':
            plt.imsave(os.path.join(save_pathA, basename + '_' + location + '_' + str(iPatch) + '.png'), he_patch)
            plt.imsave(os.path.join(save_pathB, basename + '_' + location + '_' + str(iPatch) + '.png'), mihc_patch)
        elif args.dataset_type == 'pix2pixHD':
            plt.imsave(os.path.join(save_pathA, basename + '_' + location + '_' + str(iPatch) + '.png'), he_patch)
            plt.imsave(os.path.join(save_pathB, basename + '_' + location + '_' + str(iPatch) + '.png'), mihc_patch)

        # print()

def get_arg_val():
    # setting local directories
    # dataset_name = 'Janssen'         # 'Early_data' or 'Janssen'
    # HE_wsi_dir = os.path.join(r'D:\WSI_Dataset\HIMICO', dataset_name)
    # mIHC_wsi_dir = os.path.join(r'D:\WSI_Dataset\HIMICO', dataset_name, 'Registered_mIHC')
    # gland_mask_dir = r'E:\RF_Work\HIMICO\Outputs\TumourMask_Cerberus'
    # misalignment_mask_dir = r'E:\Dropbox\RF_Warwick\Projects\HIMICO\Dataset\Misalignment_mask_WSI'
    # save_path = r'D:\Training_Data\Generative_Modelling_delete'

    # setting lsf directories
    dataset_name = 'Janssen'
    HE_wsi_dir = os.path.join(r'lab-private/it-services/leuven/all_tumor_image_data/KUL Tumor IHC&H&E', dataset_name)
    mIHC_wsi_dir = os.path.join(r'cloud_workspace/RF_Projects/HIMICO/data/Registered_Images', dataset_name)
    gland_mask_dir = r'cloud_workspace/RF_Projects/HIMICO/data/TumourMask_Cerberus'
    misalignment_mask_dir = r'cloud_workspace/RF_Projects/HIMICO/data/Misalignment_mask_WSI'
    save_path = r'F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches_2'

    parser = argparse.ArgumentParser(description=('Passing argument for split info.'))
    parser.add_argument('--dataset_type', type=str, default='cycleGAN', help='pix2pix or cycleGAN or pix2pixHD')
    parser.add_argument('--dataset_name', type=str, default=dataset_name)
    parser.add_argument('--HE_wsi_dir', type=str, default=r'F:\Datasets\himico\test\B-2032566_B5_HE')
    parser.add_argument('--mIHC_wsi_dir', type=str, default=r'F:\Datasets\himico\test\B-2032566_B5_HE')
    parser.add_argument('--gland_mask_dir', type=str, default=r'F:\Datasets\himico\test\B-2032566_B5_HE')
    parser.add_argument('--mask_type', type=str, default='tumor', help='tumor or gland')
    parser.add_argument('--misalignment_mask_dir', type=str, default=misalignment_mask_dir)
    parser.add_argument('--save_path', type=str, default=save_path)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_patch_per_wsi', type=int, default=500)
    parser.add_argument('--patch_size', type=int, default=4096)
    parser.add_argument('--resolution', type=int, default=0.5)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--unit', type=str, default='mpp')
    parser.add_argument('--tissue_thresh', type=int, default=0.8)
    args = parser.parse_args()
    print('Printing args dictionary', args)
    return args


if __name__ == "__main__":
    args = get_arg_val()
    list_files = glob.glob(os.path.join(args.HE_wsi_dir, '*HE*.mrxs'))

    list_exclude_cases = []
    if args.dataset_name == 'Janssen':
        list_exclude_cases = ['B-2029594_B6', 'B-2027329_B1-8', 'B-2036576_B1-6', 'B-2031106_B1-11']     # Janssen
        list_valid_cases = ['B-2032566', 'B-2041977']  # 'B-2032566'  'B-2041977'   Early_data or Janssen
    elif args.dataset_name == 'MicrobiomeCRC':
        list_exclude_cases = ['B-1927609_01-04', 'B-1915597_01-05', 'B-1866992_01-06', 'B-1867401_01-05',
                              'B-1881780_01-03', 'B-1914637_01-02', 'B-1917870_01-03', 'B-1917977_01-05']  # Microbiome CRC
        list_valid_cases = ['B-1895617_01-04', 'B-1895617_01-05', 'B-1927609_01-03', 'B-1905259_01-04', 'B-1905259_01-05',
                            'B-1863984_01-03', 'B-1863984_01-04', 'B-1907207_01-04', 'B-1907207_01-06',
                            'B-1910430_01-02', 'B-1910430_01-03']       # Microbiome CRC
    elif args.dataset_name == 'KUL35':
        list_exclude_cases = []
        list_valid_cases = ['B-1928279_B05','B-1899716_B01-04','B-1936734_B05','B-1936734_B06']       # KUL35

    # %% these are actually train cases from Janssen
    # list_test_cases = ['B-2042252_B1-7', 'B-2042252_B1-8', 'B-2042252_B1-9', 'B-2042252_B1-10', 'B-2042252_B1-11', 'B-2042252_B1-12']  # Early data both cdx2 and muc5 present
    # list_test_cases = ['B-1989502_B3', 'B-1989502_B5', 'B-1989502_B6', 'B-1989502_B7', 'B-1989502_B8', 'B-1989502_B9','B-1989502_B11']    # Early data both cdx2 and muc5 present
    # list_test_cases = ['B-2036576_B1-5', 'B-2036576_B1-7', 'B-2036576_B1-8', 'B-2038298_B1-4', 'B-2038298_B1-5',
    #                    'B-2038298_B1-6', 'B-2038298_B1-7', 'B-2038298_B1-8', 'B-2038298_B1-9']
    # list_test_cases = ['B-1986096_B6', 'B-1986943_B6', 'B-1993721_B5', 'B-1993721_B9']
    # list_test_cases = ['B-1986183_B6', 'B-1986183_B7', 'B-1986183_B8', 'B-2031106_B1-3', 'B-2031106_B1-5', 'B-2031106_B1-10', 'B-2031106_B1-11',
    #                    'B-2032566_B5', 'B-2032566_B7']
    # list_test_cases = ['B-2027329_B1-4', 'B-2027329_B1-5', 'B-2027329_B1-6', 'B-2027329_B1-7', 'B-2027329_B1-8', 'B-2027329_B1-9',
    #                    'B-2027329_B1-10', 'B-2027329_B1-11']
    # list_test_cases = ['B-2041977_B3', 'B-2041977_B5', 'B-2041977_B6', 'B-2041977_B7', 'B-2041977_B8', 'B-2041977_B9', 'B-2041977_B10', 'B-2041977_B11']
    # list_test_cases = ['B-2034859_B1-5', 'B-2034859_B1-6', 'B-2034859_B1-7', 'B-2034859_B1-8']
    # list_test_cases = ['B-2055115_B6', 'B-2055115_B8', 'B-2055115_B9', 'B-2055115_B10', 'B-2055115_B11', 'B-2055115_B12',
    #                    'B-2057814_B4', 'B-2057814_B5', 'B-2057814_B6', 'B-2057814_B7', 'B-2057814_B8', 'B-2057814_B9',
    #                    'B-2057814_B10', 'B-2057814_B11', 'B-2057814_B12', 'B-2057814_B13']
    # list_test_cases = ['B-2055115_B7']

    # %% these are actually train cases from MicrobiomeCRC
    # list_test_cases = ['B-1868667_01-03', 'B-1868667_01-04', 'B-1871288_01-05',
    #                    'B-1907302_02-23', 'B-1907302_02-26', 'B-1907616_01-06',
    #                    'B-1907616_01-09', 'B-1911464_01-07', 'B-1911464_01-10', 'B-1914093_01-02', 'B-1914093_01-03',
    #                    'B-1917070_01-05','B-1917121_01-03', 'B-1917121_01-04', 'B-1917870_01-03',
    #                    'B-1917977_01-05', 'B-1927074_01-07']

    # %% these are actually train cases from KUL35
    # list_test_cases = ['B-1913865_B2-5', 'B-1913865_B2-6', 'B-1932012_B12', 'B-1935905_B05', 'B-1935905_B07', 'B-1938167_B1-6', 'B-1938167_B1-7']
    list_test_cases = ['B-2032566_B5']

    if args.split == 'valid':
        list_files = [glob.glob(os.path.join(args.HE_wsi_dir, x + '*HE*.mrxs')) for x in list_valid_cases]
        list_files = list_files[0]
    elif args.split == 'test':
        list_files = [glob.glob(os.path.join(args.HE_wsi_dir, x + '*HE*.mrxs'))[0] for x in list_test_cases]
    elif args.split == 'train':
        list_files = [x for x in list_files if x not in [glob.glob(os.path.join(args.HE_wsi_dir, x + '*HE*.mrxs'))[0] for x in list_valid_cases]]
        list_files = [x for x in list_files if x not in [glob.glob(os.path.join(args.HE_wsi_dir, x + '*HE*.mrxs'))[0] for x in list_exclude_cases]]
        list_files = [x for x in list_files if x not in [glob.glob(os.path.join(args.HE_wsi_dir, x + '*HE*.mrxs'))[0] for x in list_test_cases]]

    with Pool(args.num_workers) as p:
        p.starmap(process_file, [(file, args) for file in list_files])