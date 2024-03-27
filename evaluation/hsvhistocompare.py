# -*- coding: utf-8 -*-
"""HSVHistoCompare.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/foxtrotmike/musings/blob/main/HSVHistoCompare.ipynb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 07:46:14 2023

@author: fayya
"""

import numpy as np
from skimage.color import rgb2hsv
from scipy.stats import pearsonr

def circular_histogram(values, num_bins,edge = 0.0):
    """
    Compute a circular histogram for a set of values assumed to be in a circular range.

    This function is particularly useful for data that wraps around, such as angles or hue values in color spaces, where the beginning and end of the range are equivalent.

    Parameters:
    values (array-like): A sequence of values in the range [0, 1], where 0 and 1 are considered equivalent.
    num_bins (int): The number of bins to use for the histogram.
    edge (float, optional): A small extension beyond 0 and 1 to properly bin edge values. Default is 0.0.

    Returns:
    tuple: A tuple containing two elements:
           - wrapped_counts (numpy.ndarray): The count of values in each bin, with the first and last bins combined.
           - bin_edges (numpy.ndarray): The edges of the bins.

    Usage:
    Pass a sequence of values, along with the desired number of bins and an optional edge value.
    For example, to create a circular histogram of hue values with 10 bins:
        hues = [0.1, 0.2, 0.95, 0.99]  # Example hue values
        hist_counts, bin_edges = circular_histogram(hues, 10)
    This will return the counts in each bin and the corresponding bin edges, considering the circular nature of the data.

    Note:
    The 'edge' parameter allows for a slight extension of the range to accurately bin values near 0 and 1. For example, with edge=0.1, the range is effectively extended to -0.1 to 1.1.
    """
    # assumes range of data to be 0-1 in input and considers them to be circular, i.e., 0 is 1 as in the case of hue
    extended_values = np.mod(values - edge, 1.0) + edge
    bin_edges = np.linspace(-edge, 1+edge, num_bins + 1)
    counts, _ = np.histogram(extended_values, bins=bin_edges,density = True)
    # Wrap the histogram: combine the first and last bins
    counts[0] += counts[-1]
    wrapped_counts = counts[:-1]
    return wrapped_counts, bin_edges[:-1]
"""
#ignore
def hue_histogram_similarity(X, Y, bins = 16):
    L, W, C = X.shape
    M = (np.random.choice([0, 1], (L, W), p=[0.0, 1.0])*255).astype(np.uint8)
    # Convert to HSV using skimage
    Xhsv = rgb2hsv(X)
    Yhsv = rgb2hsv(Y)
    # Set ranges and bins for histograms
    histX, _ = circular_histogram(Xhsv[:,:,0][M==255], num_bins = bins)
    histY, _ = circular_histogram(Yhsv[:,:,0][M==255], num_bins = bins)
    # Normalize histograms
    histX /= histX.sum()
    histY /= histY.sum()
    # Calculate correlation
    corr, _ = pearsonr(histX, histY)
    return corr
"""
def HSV_histogram_similarity(X, Y, M_x = None, M_y = None, bins = (16,16)):
    """
    Calculate the histogram similarity of Hue and Saturation between two images.
    Note that this function does nont compare structures.

    Parameters:
    X (numpy.ndarray): The first image in RGB format.
    Y (numpy.ndarray): The second image in RGB format.
    M_x (numpy.ndarray, optional): A binary mask to specify the region of interest in the image X.
                                 If None, a mask with all ones is used.
    M_y (numpy.ndarray, optional): A binary mask to specify the region of interest in the image Y.
                                 If None, a mask with all ones is used.
    bins (tuple of integers): The number of bins to use for histogram calculation (hue, saturation)

    Returns:
    tuple: A tuple containing the correlation coefficients for Hue and Saturation histograms.

    Usage:
    Pass two RGB images X and Y, along with an optional mask M and the number of histogram bins.
    The function returns a tuple (hue_corr, sat_corr) where:
    - hue_corr is the Pearson correlation coefficient of the Hue histograms.
    - sat_corr is the Pearson correlation coefficient of the Saturation histograms.
    """

    if M_x is None: #create a mask of all ones
        M_x = (np.random.choice([0, 1], X.shape[:2], p=[0.0, 1.0])*255).astype(np.uint8)
    if M_y is None: #create a mask of all ones
        M_y = (np.random.choice([0, 1], Y.shape[:2], p=[0.0, 1.0])*255).astype(np.uint8)
    M_x = M_x==255
    M_y = M_y==255
    # Convert to HSV using skimage
    Xhsv = rgb2hsv(X)
    Yhsv = rgb2hsv(Y)
    # hue similarity
    histX, _ = circular_histogram(Xhsv[:,:,0][M_x], num_bins = bins[0], edge = 0.1)
    histY, _ = circular_histogram(Yhsv[:,:,0][M_y], num_bins = bins[0], edge = 0.1)
    # Normalize histograms
    histX /= histX.sum()
    histY /= histY.sum()
    # Calculate correlation
    hue_corr, _ = pearsonr(histX, histY)
    # saturation similarity
    histX, _ = np.histogram(Xhsv[:,:,1][M_x], bins=bins[1], range=[0,1], density=True)
    histY, _ = np.histogram(Yhsv[:,:,1][M_y], bins=bins[1], range=[0,1], density=True)
    # Normalize histograms
    histX /= histX.sum()
    histY /= histY.sum()
    # Calculate correlation
    sat_corr, _ = pearsonr(histX, histY)
    return hue_corr,sat_corr


from skimage import data
base_img = data.astronaut()  # Example RGB image
#%%
from skimage.io import imread
import os

real_dir = r'F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches_2\cycleGAN\himico_tumor\test_B'
syn_dir = r'F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches_2\cycleGAN\himico_tumor\output\pix2pixhd'
hue_corr = 0
sat_corr = 0
num_images = len(os.listdir(real_dir))

for file in os.listdir(real_dir):
    print('-------------------', file, '------------------------')
    X = imread(os.path.join(real_dir, file))
    file = file.split(".")[0]+'_984_984_synthesized_image.png'
    Y = imread(os.path.join(syn_dir, file))
    X = X[:,:,:3]
    Y = Y[:,:,:3]
    hue,sat = HSV_histogram_similarity(Y,X)
    print(hue)
    print(sat)
    print('---------------------------------------------------')
    hue_corr+=hue
    sat_corr+=sat

print("Hue Histogram Correlation: ",hue_corr/num_images)
print("Saturation Histogram Correlation: ",sat_corr/num_images)