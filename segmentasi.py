# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:37:36 2022

@author: jg

pip install openpyxl
pip install colorama
"""

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time, datetime
import os, sys, getopt
import shutil
from colorama import init
from colorama import Fore, Back, Style

import numpy as np
from scipy.spatial.distance import jaccard
from sklearn.metrics import jaccard_score

from skimage.filters import threshold_otsu
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening, square)
from skimage.filters import unsharp_mask
from skimage import exposure

#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
    
#load image
def loadImage(path):
    img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#load images
def loadImages(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images

#Function used to show a single image
def showIm(im, name="image"):
    plt.imshow(im, cmap="gray"), plt.title(name)
    show = plt.show()
    return show

def entp(x):
    temp = np.multiply(x, np.log(x))
    temp[np.isnan(temp)] = 0
    return temp

def adaptiveThresholdingEntropy(img):
    H = cv2.calcHist([img],[0],None,[256],[0,256])
    H = H / np.sum(H)
    theta = np.zeros(256)
    Hf = np.zeros(256)
    Hb = np.zeros(256)
    for T in range(1,255):
        Hf[T] = - np.sum( entp(H[:T-1] / np.sum(H[1:T-1])) )
        Hb[T] = - np.sum( entp(H[T:] / np.sum(H[T:])) )
        theta[T] = Hf[T] + Hb[T]
    theta_max = np.argmax(theta)
    img_out = img > theta_max
    threshold_v = theta_max
    if (threshold_v%2==0):
        threshold_v+=1
    else:
        pass
    adaptive_g = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 401, 8)
    return adaptive_g


def cv2Binary(base_img):
    ret,thresh2 = cv2.threshold(base_img,0,1,cv2.THRESH_BINARY)
    return thresh2

#grayscale single
def cv2Gray(img):
    image  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return image

#grayscale to all images
def cv2Grays(img):
    images= []
    for image in img:
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
    return images

#preprocess all images
def preprocessing(ima):
    images = []
    for image in ima:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image)
        if image is not None:
            images.append(image)
    return images

def cv2rgb(img):
    image  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image

def rgb2gray(rgb):
    return np.round((np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])),0)

# def bilateral(img):
#     bilateral = cv2.bilateralFilter(img, 6, 15, 30)
#     return bilateral

# Contrast stretching
def contrast(img):
    p2, p98 = np.percentile(img, (2, 90))
    image = exposure.rescale_intensity(img, in_range=(p2, p98))
    return image

# Image sharpening
# def sharpen(img):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5,-1],
#                        [0, -1, 0]])
#     image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
#     return image_sharp

def unsharp(img):
    image = unsharp_mask(img, radius=5, amount=2)
    return image

def billateralBlur(img, kernel=6):
    img = cv2.bilateralFilter(img, kernel,15,90)
    return img

def gaussianBlur(img, kernel=5):
    img = cv2.GaussianBlur(img, (kernel,kernel))
    return img

def seg_otsu(img):
    thresh = threshold_otsu(img)
    imbw = closing(img > thresh, square(1))
    return (~imbw)*1

def otsuThresholding(img):
    threshold_v, otsu = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold_v)
    return otsu


def seg_kmeans(image, ch=1):
    pixel_vals = image.reshape((-1,ch))
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # then perform k-means clustering with number of clusters defined as 2
    #also random centres are initially choosed for k-means clustering
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
#     segmented_image = segmented_data.reshape((image.shape)).astype(np.uint8)
    segmented_image = segmented_data.reshape((image.shape))
    th, segmented_image = cv2.threshold(segmented_image, 120, 255, cv2.THRESH_BINARY)
#     th, segmented_image = cv2.threshold(segmented_image, 0, 1, cv2.THRESH_BINARY)
#     segmented_image = 255 - segmented_image
    return segmented_image

def tipis(im, k_w=1, k_h=1):
    kernel = np.ones((k_w,k_h),np.uint8)
    im = cv2.erode(im, kernel, iterations=1 )
    return im

def tebal(im, k_w=1, k_h=1):
    kernel = np.ones((k_w,k_h),np.uint8)
    im = cv2.dilate(im, kernel, iterations=1 )
    return im

def connCompSingle(im, min_size=1600):
    im = im
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(im)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    # minimum size of particles
    min_size = min_size
    #check if component size to small
    im_result = np.zeros((im.shape))
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255
    return im_result  

def connComp(img):
    image = connCompSingle(img)
    if (countWhitePixel(image)==0):
        image = connCompSingle(img,200)
    else:
        pass
    return image


#count white pixels for all images
def countWhitePixels(img):
    data = []
    for image in img:
        pxl = cv2.countNonZero(image)
        data.append(pxl)
    return data

#count white pixel for single image
def countWhitePixel(img):
    pxl = cv2.countNonZero(img)
    return pxl

def holeFilling(im):
    im = im.astype(np.uint8)    
    im_floodfill = im.copy()
    h, w = im.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im | im_floodfill_inv
    return im_out

def simJaccard(gt_image, th_image):
    similarity = jaccard_score(gt_image.flatten(), th_image.flatten())
    return round(similarity,4)

def dice_score(true, pred, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def simDice(gt_image, th_image):
    similarity = dice_score(gt_image.flatten(), th_image.flatten())
    return round(similarity,4)


#assigning name to dataframe
def getName(folderin):
    data = []
    for filename in os.listdir(folderin):
        data.append(filename)
    return data

def showAllIm(img):
    for images in img:
        showIm(images)
        
def showHist(img):
    dst = cv2.calcHist(img, [0], None, [256], [0,256])
    plt.hist(img.ravel(),256,[0,256])
    plt.show()        
        
def saveImages(folderin, folderOut, images):
    i=0
    for filename in os.listdir(folderin):  
        img = os.path.join(folderOut, filename)
        cv2.imwrite(img, images[i])
        i+=1    
        
def myfunc(argv):
    arg_input = ""
    arg_output = ""
    arg_help = "{0} -i <input> -o <output>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "input=", 
        "user=", "output="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            if(os.path.isdir(arg)==True):
                arg_input = arg
            else:
                print(f"Folder {arg} not found! please make sure the folder exist", file=sys.stderr)
                sys.exit(2)
        elif opt in ("-o", "--output"):
            arg_output = arg
    
    init()
    line = '------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
    print(line)
    print('$$$$$$\                                                    $$$$$$\                                                         $$\                $$\     $$\                     ')
    print('\_$$  _|                                                  $$  __$$\                                                        $$ |               $$ |    \__|                    ')
    print('  $$ |  $$$$$$\$$$$\   $$$$$$\   $$$$$$\   $$$$$$\        $$ /  \__| $$$$$$\   $$$$$$\  $$$$$$\$$$$\   $$$$$$\  $$$$$$$\ $$$$$$\    $$$$$$\ $$$$$$\   $$\  $$$$$$\  $$$$$$$\  ') 
    print('  $$ |  $$  _$$  _$$\  \____$$\ $$  __$$\ $$  __$$\       \$$$$$$\  $$  __$$\ $$  __$$\ $$  _$$  _$$\ $$  __$$\ $$  __$$\\_$$  _|   \____$$\\_$$  _|  $$ |$$  __$$\ $$  __$$\ ')
    print('  $$ |  $$ / $$ / $$ | $$$$$$$ |$$ /  $$ |$$$$$$$$ |       \____$$\ $$$$$$$$ |$$ /  $$ |$$ / $$ / $$ |$$$$$$$$ |$$ |  $$ | $$ |     $$$$$$$ | $$ |    $$ |$$ /  $$ |$$ |  $$ |')
    print('  $$ |  $$ | $$ | $$ |$$  __$$ |$$ |  $$ |$$   ____|      $$\   $$ |$$   ____|$$ |  $$ |$$ | $$ | $$ |$$   ____|$$ |  $$ | $$ |$$\ $$  __$$ | $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |')
    print('$$$$$$\ $$ | $$ | $$ |\$$$$$$$ |\$$$$$$$ |\$$$$$$$\       \$$$$$$  |\$$$$$$$\ \$$$$$$$ |$$ | $$ | $$ |\$$$$$$$\ $$ |  $$ | \$$$$  |\$$$$$$$ | \$$$$  |$$ |\$$$$$$  |$$ |  $$ |')
    print('\______|\__| \__| \__| \_______| \____$$ | \_______|       \______/  \_______| \____$$ |\__| \__| \__| \_______|\__|  \__|  \____/  \_______|  \____/ \__| \______/ \__|  \__|')
    print('                                $$\   $$ |                                    $$\   $$ |                                                                                      ')
    print('                                \$$$$$$  |                                    \$$$$$$  |                                                                                      ')
    print('                                 \______/                                      \______/                                                                                       \n')

    print('       __  _   _  _   __ ___            __     ___ _  ___  _      _   _   _  _                         _     _          _   _   __  _   _   _  ___             _           __ ___ ') 
    print('  /\  (_  |_) |_ |_) /__  |  |  |  | | (_       | |_)  |  /  |_| / \ | \ |_ |_) |\/|  /\     /\  |\ | | \   /  |   /\  | \ / \ (_  |_) / \ |_)  |  | | |\/|   |_ | | |\ | /__  |  ')
    print(' /--\ __) |   |_ | \ \_| _|_ |_ |_ |_| __) o    | | \ _|_ \_ | | \_/ |_/ |_ | \ |  | /--\   /--\ | \| |_/   \_ |_ /--\ |_/ \_/ __) |   \_/ | \ _|_ |_| |  |   |  |_| | \| \_| _|_ ')
    print('                                           /')                                                  
    print(Fore.RED + 'version: 1.0.0 \n' + Style.RESET_ALL )
    print(line)
    a = str(datetime.datetime.now())
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Program running, started at: '+ Fore.CYAN + a + Style.RESET_ALL)
    folder_base = arg_input+'/dataset/base/'
    folder_gt = arg_input+'/dataset/gt/'
    base_images = loadImages(folder_base)
    gt_images = loadImages(folder_gt)
    #base_image = loadImage(base_path)
    #gt_image = loadImage(gt_path)
    #do otsu
    name = getName(folder_base)
    name_gt = getName(folder_gt)
    #get the names, validate
    data = pd.DataFrame()
    data["name"] = name
    data["name_gt"] = name_gt
    otsu_images = [] 
    dire = 'output'
    ots = 'otsu'
    adp = 'adaptive'
    clu = 'clustering'
    fileout = os.path.join(arg_input, dire)
    ots2 = os.path.join(fileout, ots)
    adp2 = os.path.join(fileout, adp)
    clu2 = os.path.join(fileout, clu)
    print(line+'\n')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' CHECKING FILE CONFLICT')    
    if os.path.isdir(fileout)==True:
        print(Back.RED + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder output found,'+Fore.RED+' deleting'+Style.RESET_ALL+ ' file conflict...')
        shutil.rmtree(fileout)
    else:
        pass
    os.mkdir(fileout)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder output has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    os.mkdir(ots2)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder otsu has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    os.mkdir(adp2)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder adaptive has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    os.mkdir(clu2)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder clustering has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' CHECKING FILE CONFLICT DONE \n')
    print(line)
    #do otsu
    otsu_only = os.path.join(ots2, 'otsu_images') 
    os.mkdir(otsu_only)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' RUNNING SEGMENTATION ALGORITHM \n')
    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder otsu_images has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'Otsu Thresholding algorithm...'+Style.RESET_ALL)    
    for image in base_images:
    #     img = morp(seg_otsu(rgb2gray(np.float32(unsharp(contrast(bilateral(cv2rgb(image))))))))
        img = seg_otsu(rgb2gray(np.float32(unsharp(contrast(billateralBlur(cv2rgb(image),6))))))
        otsu_images.append(img)
    saveImages(folder_base, otsu_only, otsu_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' otsu_images has been' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')        
        
    #do adaptive images with billateral blur
    adap_only = os.path.join(adp2, 'adaptive_images') 
    os.mkdir(adap_only)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder adaptive_images has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'Adaptive Thresholding algorithm...'+Style.RESET_ALL)    
    adaptive_images = []
    for image in cv2Grays(base_images):
        img = adaptiveThresholdingEntropy(billateralBlur(image,6))
        adaptive_images.append(img)
    saveImages(folder_base, adap_only, adaptive_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' adaptive_images has been' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
        
    #do clustering
    clus_only = os.path.join(clu2, 'clustering_images') 
    os.mkdir(clus_only)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder adaptive_images has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'K-Means Clustering algorithm...'+Style.RESET_ALL)    
    clus_images = []
    for image in base_images:
    #     img = cv2Binary(morp(connCompSingle(rgb2gray(255-seg_kmeans(sharpen(contrast(billateralBlur(cv2rgb(image),5))),3)).astype(np.uint8))))
        img = cv2Binary(rgb2gray(255-seg_kmeans(contrast(billateralBlur(cv2rgb(image),6)),3)))
        clus_images.append(img)
    saveImages(folder_base, clus_only, clus_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' clustering_images have been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
 
    
    #Adaptive - CCA
    adap_cca = os.path.join(adp2, 'adaptive_images_cca') 
    os.mkdir(adap_cca)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder adaptive_images_cca has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'Adaptive Thresholding algorithm with CCA...'+Style.RESET_ALL)
    adap_cca_images = []
    for image in adaptive_images:
        img = connComp(image)
        adap_cca_images.append(img)
    saveImages(folder_base, adap_cca, adap_cca_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Adaptive Images with CCA have been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
        
    #Otsu - CCA
    otsu_cca = os.path.join(ots2, 'otsu_images_cca') 
    os.mkdir(otsu_cca)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder otsu_images_cca has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'Otsu Thresholding algorithm with CCA...'+Style.RESET_ALL)
    otsu_cca_images = []
    for image in otsu_images:
        img = connComp(image.astype(np.uint8))
        otsu_cca_images.append(img)
    saveImages(folder_base, otsu_cca, otsu_cca_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Otsu Images with CCA have been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
          
        
    #Cluster - CCA
    clus_cca = os.path.join(clu2, 'clustering_images_cca') 
    os.mkdir(clus_cca)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder clustering_images_cca has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'K-Means Clustering algorithm with CCA...'+Style.RESET_ALL)
    clus_cca_images = []
    for image in clus_images:
        img = connComp(image.astype(np.uint8))
        clus_cca_images.append(img)
    saveImages(folder_base, clus_cca, clus_cca_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Clustering Images with CCA have been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
        
    #Adaptive - FILL
    adap_fill = os.path.join(adp2, 'adaptive_images_cca_fill') 
    os.mkdir(adap_fill)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder adaptive_images_cca_fill has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'Adaptive Thresholding algorithm with CCA and Hole Filling...'+Style.RESET_ALL)  
    adap_fill_images = []
    # connected_images2 = connected_images
    # for image in connected_images2:
    for image in adap_cca_images:
        img = holeFilling(image)
        if (countWhitePixel(img) <= countWhitePixel(image)*3):
            adap_fill_images.append(img)
        else:
            adap_fill_images.append(image)
    saveImages(folder_base, adap_fill, adap_fill_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Adaptive Thresholding with CCA and Hole Filling Images have been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
                         
    #Otsu - FILL
    otsu_fill = os.path.join(ots2, 'otsu_images_cca_fill') 
    os.mkdir(otsu_fill)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder otsu_images_cca_fill has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'Otsu Thresholding algorithm with CCA and Hole Filling...'+Style.RESET_ALL)
    otsu_fill_images = []
    for image in otsu_cca_images:
        img = holeFilling(image)
        if (countWhitePixel(img) <= countWhitePixel(image)*3):
            otsu_fill_images.append(img)
        else:
            otsu_fill_images.append(image)
    saveImages(folder_base, otsu_fill, otsu_fill_images)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Otsu Thresholding with CCA and Hole Filling Images have been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
            
    #Cluster - FILL
    clus_fill = os.path.join(clu2, 'clus_images_cca_fill') 
    os.mkdir(clus_fill)    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' folder clustering_images_cca_fill has been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'created...')
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Executing '+Fore.LIGHTGREEN_EX+'K-Means Clustering algorithm with CCA and Hole Filling...'+Style.RESET_ALL)
    clus_fill_images = []
    for image in clus_cca_images:
        img = holeFilling(image)
        if (countWhitePixel(img) <= countWhitePixel(image)*3):
            clus_fill_images.append(img)
        else:
            clus_fill_images.append(image)
    saveImages(folder_base, clus_fill, clus_fill_images)
    
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' K-Means Clustering with CCA and Hole Filling images have been ' + Fore.CYAN +'successfully '+Style.RESET_ALL +'generated...')
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + 'All of the Segmentation Algorithms ' + Fore.CYAN +' have been successfully '+Style.RESET_ALL +'executed')
    print(line)   
    #print('---------------------------------------------------------------------------------------')   
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:                 :'+Fore.LIGHTRED_EX+'(0%)'+Style.RESET_ALL)     
    #do DICE similarity ADAPTIVE
    dic_adaptive = []
    for index in range (0, len(gt_images)):
        sim_ad = simDice(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(adaptive_images[index]))
        dic_adaptive.append(sim_ad)
    data["dic_adap"] = dic_adaptive
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:-                :'+Fore.LIGHTYELLOW_EX+'(05,88%)'+Style.RESET_ALL) 
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:--               :'+Fore.LIGHTYELLOW_EX+'(11,76%)'+Style.RESET_ALL)        
    #Adaptive - CCA [DICE]
    dic_adap_cca = []
    for index in range (0, len(gt_images)):
        sim = simDice(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(adap_cca_images[index]))
        dic_adap_cca.append(sim)
    data["dic_adap_cca"] = dic_adap_cca

    #Adaptive - FILL [DICE]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:---              :'+Fore.LIGHTYELLOW_EX+'(17,64%)'+Style.RESET_ALL)        
    dic_adap_fill = []
    for index in range (0, len(gt_images)):
        sim = simDice(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(adap_fill_images[index]))
        dic_adap_fill.append(sim)
    data["dic_adap_fill"] = dic_adap_fill


    #do DICE similarity OTSU
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:----             :'+Fore.LIGHTYELLOW_EX+'(23,52%)'+Style.RESET_ALL)        
    dic_otsu = []
    for index in range (0, len(gt_images)):
        sim_otsu = simDice(cv2Binary(cv2Gray(gt_images[index])),otsu_images[index])
        dic_otsu.append(sim_otsu)
    data["dic_otsu"] = dic_otsu

    #Otsu - CCA [DICE]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:-----            :'+Fore.LIGHTYELLOW_EX+'(29,41%)'+Style.RESET_ALL)        
    dic_otsu_cca = []
    for index in range (0, len(gt_images)):
        sim = simDice(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(otsu_cca_images[index]))
        dic_otsu_cca.append(sim)
    data["dic_otsu_cca"] = dic_otsu_cca

    #Otsu - FILL [DICE]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:------           :'+Fore.LIGHTYELLOW_EX+'(35,29%)'+Style.RESET_ALL)        
    dic_otsu_fill = []
    for index in range (0, len(gt_images)):
        sim = simDice(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(otsu_fill_images[index]))
        dic_otsu_fill.append(sim)
    data["dic_otsu_fill"] = dic_otsu_fill


    #do DICE similarity CLUS
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:-------          :'+Fore.LIGHTYELLOW_EX+'(41,17%)'+Style.RESET_ALL)        
    dic_clus = []
    for index in range (0, len(gt_images)):
        sim_clus = simDice(cv2Binary(cv2Gray(gt_images[index])),clus_images[index])
        dic_clus.append(sim_clus)
    data["dic_clus"] = dic_clus

    #Cluster - CCA [DICE]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:--------         :'+Fore.LIGHTYELLOW_EX+'(47,05%)'+Style.RESET_ALL)        
    dic_clus_cca = []
    for index in range (0, len(gt_images)):
        sim = simDice(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(clus_cca_images[index]))
        dic_clus_cca.append(sim)
    data["dic_clus_cca"] = dic_clus_cca

    #Cluster - FILL [DICE]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:---------        :'+Fore.LIGHTYELLOW_EX+'(52,94%)'+Style.RESET_ALL)       
    dic_clus_fill = []
    for index in range (0, len(gt_images)):
        sim = simDice(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(clus_fill_images[index]))
        dic_clus_fill.append(sim)
    data["dic_clus_fill"] = dic_clus_fill
    
    #do jacard similarity
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:----------       :'+Fore.LIGHTYELLOW_EX+'(58,82%)'+Style.RESET_ALL)       
    jac_adaptive = []
    for index in range (0, len(gt_images)):
        sim_ad = simJaccard(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(adaptive_images[index]))
        jac_adaptive.append(sim_ad)
    data["jac_adap"] = jac_adaptive

    #Adaptive - CCA [JACCARD]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:-----------      :'+Fore.LIGHTYELLOW_EX+'(64,70%)'+Style.RESET_ALL)       
    jac_adap_cca = []
    for index in range (0, len(gt_images)):
        sim_ad = simJaccard(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(adap_cca_images[index]))
        jac_adap_cca.append(sim_ad)
    data["jac_adap_cca"] = jac_adap_cca

    #Adaptive - FILL [JACCARD]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:------------     :'+Fore.LIGHTYELLOW_EX+'(70,58%)'+Style.RESET_ALL)       
    jac_adap_fill = []
    for index in range (0, len(gt_images)):
        sim_ad = simJaccard(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(adap_fill_images[index]))
        jac_adap_fill.append(sim_ad)
    data["jac_adap_fill"] = jac_adap_fill

    #do jacard similarity
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:-------------    :'+Fore.LIGHTYELLOW_EX+'(76,47%)'+Style.RESET_ALL)       
    jac_otsu = []
    for index in range (0, len(gt_images)):
        sim_otsu = simJaccard(cv2Binary(cv2Gray(gt_images[index])),otsu_images[index])
        jac_otsu.append(sim_otsu)
    data["jac_otsu"] = jac_otsu

    #Otsu - CCA [JACCARD]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:--------------   :'+Fore.LIGHTYELLOW_EX+'(82,35%)'+Style.RESET_ALL ) 
    jac_otsu_cca = []
    for index in range (0, len(gt_images)):
        sim_ad = simJaccard(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(otsu_cca_images[index]))
        jac_otsu_cca.append(sim_ad)
    data["jac_otsu_cca"] = jac_otsu_cca

    #Otsu - FILL [JACCARD]
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:---------------  :'+Fore.LIGHTYELLOW_EX+'(88,23%)'+Style.RESET_ALL)      
    jac_otsu_fill = []
    for index in range (0, len(gt_images)):
        sim_ad = simJaccard(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(otsu_fill_images[index]))
        jac_otsu_fill.append(sim_ad)
    data["jac_otsu_fill"] = jac_otsu_fill


    #do jacard similarity
    print(Back.LIGHTBLACK_EX + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:---------------- :'+Fore.LIGHTYELLOW_EX+'(94,11%)'+Style.RESET_ALL)      
    jac_clus = []
    for index in range (0, len(gt_images)):
        sim_clus = simJaccard(cv2Binary(cv2Gray(gt_images[index])),clus_images[index])
        jac_clus.append(sim_clus)
    data["jac_clus"] = jac_clus

    #Cluster - CCA [JACCARD]
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + Fore.LIGHTYELLOW_EX+' Calculating'+Style.RESET_ALL+ ' images similarities:-----------------:'+Fore.LIGHTGREEN_EX+'(100%)'+Style.RESET_ALL)      
    jac_clus_cca = []
    for index in range (0, len(gt_images)):
        sim_ad = simJaccard(cv2Binary(cv2Gray(gt_images[index])),cv2Binary(clus_cca_images[index]))
        jac_clus_cca.append(sim_ad)
    data["jac_clus_cca"] = jac_clus_cca
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Images similarities calculation'+Fore.CYAN+' completed'+Style.RESET_ALL+', assigning the results') 

    print(line)    
    data.to_excel(os.path.join(fileout, "detail_result.xlsx"))
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Image segmentation results has been '+Fore.CYAN+'successfully'+Style.RESET_ALL+' generated as '+Fore.CYAN+'.xlsx file\n'+Style.RESET_ALL)
    data.describe().to_excel(os.path.join(fileout, "segmentation_result.xlsx"))  
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Segmentation results has been '+Fore.CYAN+'successfully'+Style.RESET_ALL+' generated as '+Fore.CYAN+'.xlsx file\n'+Style.RESET_ALL)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL + ' Completed at ' + Fore.CYAN+ str(datetime.datetime.now())+Style.RESET_ALL)
    print(Back.GREEN + '!SYSTEM NOTIFICATION:' + Style.RESET_ALL +  Fore.LIGHTRED_EX+ " Exiting...")
        

if __name__ == "__main__":
    myfunc(sys.argv)