# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:51:46 2022

@author: FireT
"""


from skimage import measure, io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray
import numpy as np
import cv2 
import scipy 
from scipy.stats import norm
from skimage import morphology


# The input image.
image = cv2.imread("Co-Cu-19_05.tif",0)
image=image[0:696, :]
scale = 0.1116 #microns/pixel
Pixelcut= 100 #Minimum pixel Size
 
#plt.hist(image.flat, bins=100, range=(0,150))  #.flat returns the flattened numpy array (1D)



from skimage.filters import threshold_otsu #binary image
ret, thresh=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Generate thresholded image
#thresholded_img = image > threshold
#plt.imshow(thresholded_img,cmap='gray')


#Remove edge touching regions
from skimage.segmentation import clear_border
edge_touching_removed = clear_border(thresh)
#plt.imshow(edge_touching_removed)

kernel=np.ones ((3,3),np.uint8)
eroded=cv2.erode(edge_touching_removed,kernel,iterations=1)
dilated=cv2.dilate(eroded,kernel,iterations=1)

#Label connected regions of an integer array using measure.label
#Labels each connected entity as one object
#Connectivity = Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
#If None, a full connectivity of input.ndim is used, number of dimensions of the image
#For 2D image it would be 2

label_image = measure.label(dilated, connectivity=image.ndim)


#Removal of small sized particles
label_image = morphology.remove_small_objects(label_image, Pixelcut)
#plt.imshow(label_image,cmap='Blue',interpolation='none')

#plt.imshow(label_image)
#Return an RGB image where color-coded labels are painted over the image.
#Using label2rgb

image_label_overlay = label2rgb(label_image, image=image)
plt.imshow(image_label_overlay)

plt.imsave("labeled_Co-cu_05.jpg", image_label_overlay) 


#################################################
#Calculate properties
#Using regionprops or regionprops_table
all_props=measure.regionprops(label_image, image)
#Can print various parameters for all objects
for prop in all_props:
    print('Label: {} Area: {}'.format(prop.label, prop.area))

#Compute image properties and return them as a pandas-compatible table.
#Available regionprops: area, bbox, centroid, convex_area, coords, eccentricity,
# equivalent diameter, euler number, label, intensity image, major axis length, 
#max intensity, mean intensity, moments, orientation, perimeter, solidity, and many more

props = measure.regionprops_table(label_image, image, 
                          properties=['label',
                                      'area', 'equivalent_diameter',
                                      'mean_intensity', 'solidity'])

import pandas as pd
df = pd.DataFrame(props)
print(df.head())

#To delete small regions...
#df = df[df['area'] > 0]

print(df.head())

#######################################################
#Convert to micron scale
df['area_sq_microns'] = df['area'] * (scale**2)
df['equivalent_diameter_microns'] = df['equivalent_diameter'] * (scale)
print(df.head())





df.to_csv('measurements-100pixdel.csv')


#size distrib
import seaborn as sns

sns.set_style('white')

sns.distplot(df.area_sq_microns)



