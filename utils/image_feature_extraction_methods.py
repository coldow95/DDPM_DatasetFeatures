import numpy as np
import seaborn as sns
import pandas as pd
import sklearn
import os
import cv2
import torch
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, skew
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from scipy import stats

from skimage import data
from skimage.measure import shannon_entropy


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#Deep-learning based saliency prediction
from torchvision import models, transforms
from PIL import Image
from matplotlib import cm


# Removed num_to_groups
from IPython.display import display, HTML


'''We start by loading the deep learning models, namely, segment anything and a standard resnet'''

sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#mask_generator = SamAutomaticMaskGenerator(sam)

#8, 0.9, 0.8
#12, 0.95, 0.8
#10, 0.95, 0.9 -- recent
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=12,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.5,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=1000,  # Requires open-cv to run post-processing
)


#https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
# Load the pre-trained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()


def round_pad(value, round_digits=3):
    return format(round(value,round_digits), "."+str(round_digits)+"f")

def crop_background(image):
    '''Function that crops when the image has a purely black or white background (shells, skulls and pokemon), as most measures (i.e. texture or symmetry) give biased results when the background is too large. This function choses cropped are based on hsv-values.'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
    k = [0,1,2,3,4,5,6,7,8,9,10]
            
    y_top, y_bot, x_left, x_right = 0,0,0,0
    done = False
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y,x][2] < 115 or image[y,x][2] >220:
                continue
            y_top = y
            done = True
            break
        if done:
            break
                    
    done = False
    for y in range(image.shape[0]-1,0,-1):
        for x in range(image.shape[1]):
            if image[y,x][2] < 115 or image[y,x][2] >220:
                continue
                
            y_bot = y
            done = True
            break
        if done:
            break
                    
    done = False
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if image[y,x][2] < 115 or image[y,x][2] >220:
                continue
            x_left = x
            done = True
            break
        if done:
            break
                    
    done = False
    for x in range(image.shape[1]-1,0,-1):
        for y in range(image.shape[0]):
            if image[y,x][2] < 115 or image[y,x][2] >220:
                continue
            x_right = x
            done = True
            break
        if done:
            break
            
    image = image[y_top:y_bot, x_left:x_right]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = cv2.resize(image,(96,96),interpolation = cv2.INTER_CUBIC)
    return image

def get_representative_index(feature_list):
    '''Helper function that returns the image position based on the average value of that feature. This means that the method returns the position of the image in the list, which has a value closest to the average value of the feature. This image is called representative image.'''
    return np.argmin([abs(feature - np.mean(feature_list)) for feature in feature_list]), np.mean(feature_list)


def plot_images(pixel_arrays, n=100):  
    if torch.is_tensor(pixel_arrays):
        pixel_arrays = cv2.normalize(src=pixel_arrays.cpu().detach().numpy().transpose(0,2,3,1), dst=None, alpha=0, beta=255,              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(pixel_arrays) < n:
        n = len(pixel_arrays)
    fig = plt.figure(figsize=(30,int(n*0.5)))
    img_list = list()
    img_file_list = list()
    mask_list = list()
    for i in range (n):
        ax = fig.add_subplot(math.ceil(n/10),10,i+1)
        plt.imshow(pixel_arrays[i])

    plt.show()

    
def iterate(datasets, dataset_folder):
    img_dict = {}
    for folder_name in datasets:
        img_dict[folder_name] = []
        path = os.path.join(dataset_folder, folder_name)
        image_paths = os.listdir(path)
        if 'img' in image_paths:
            path = os.path.join(path,'img')
            image_paths = os.listdir(path)
            folder = folder_name+'/img/'
        else:
            folder = folder_name

        for image_path in image_paths:
            #print(image_paths)
            image = cv2.imread(os.path.join(dataset_folder, folder, image_path)) 
            #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            img_dict[folder_name].append(image)
        
    return img_dict

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zero(len(d))
    return data[s<m]

def fill_dict(dict_list, fill_list):
    for i, item in enumerate(fill_list):
        dict_list[i] = item
    
def gray_level_entropy(image):
    return round(shannon_entropy(image), 2)


#Deep-learning based segmentation
#Segmentation code based on https://github.com/facebookresearch/segment-anything
def show_anns(anns):
    '''This method helps to make the annotations of the segmenet anything model visible'''
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.8)))
        
        
def plot_masked_image(image, anns):
    '''Helper function to plot the normal and the masked images for demonstration purposes'''
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    show_anns(anns)
    plt.axis('off')
    plt.show()    
    
    

def segment_anything(data_image_dict, mask_generator, visualize=False, center=True):
    '''Method that applies the segmentation algorithm and returns a dataframe with the segmentation-based features.'''
    #data_dict = iterate(datasets, dataset_folder)
    image_list = list()
    image_feat_dict, image_compl_dict, image_div_dict = dict(), dict(), dict()
    for dataset, image_list in data_image_dict:
        print(dataset)
        image_count = 0
        image_compl_dict[dataset], image_div_dict[dataset] = list(), list()
        max_num_masks, min_num_masks, max_max_size_masks, min_max_size_masks = 0, 1000, 0, 1000
        
        min_max_image_list, min_max_mask_list = [None]*4, [None]*4
        num_mask_list, avg_size_list, std_size_list, max_area_list = list(), list(), list(), list()
        
        masks_list = list()

        for image in image_list:
            if dataset in ['Shells','Skulls', 'Poke'] and center == True:
                image = crop_background(image)
            image_representation = list()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            masks_list.append(masks)
                      
            num_mask_list.append(len(masks))
            avg_size_list.append(np.mean([mask['area'] for mask in masks]))
            std_size_list.append(np.std([mask['area'] for mask in masks]))
            max_area_list.append(np.max([mask['area'] for mask in masks]))
        
            image_count += 1
                        
            
            if len(masks) > max_num_masks:
                max_num_masks = len(masks)
                min_max_image_list[0] = image
                min_max_mask_list[0] = masks
                
            if len(masks) < min_num_masks:
                min_num_masks = len(masks)
                min_max_image_list[1] = image
                min_max_mask_list[1] = masks
                
            if np.mean([mask['area'] for mask in masks]) > max_max_size_masks:
                max_max_size_masks = np.mean([mask['area'] for mask in masks])
                min_max_image_list[2] = image
                min_max_mask_list[2] = masks
                
            if np.mean([mask['area'] for mask in masks]) < min_max_size_masks:
                min_max_size_masks = np.mean([mask['area'] for mask in masks])
                min_max_image_list[3] = image
                min_max_mask_list[3] = masks
                
            
        if visualize == True:
            print("\nMINMAX_comparsion")
            print("max_num_mask:\t", round(max_num_masks,2), "\tmin_num_masks:\t",min_num_masks)
            print("max_max_size:\t", round(max_max_size_masks,2), "\tmin_max_size:\t",min_max_size_masks)
            for i, image in enumerate(min_max_image_list):
                plot_masked_image(image, min_max_mask_list[i])

            print("\nREPRESENTATIVE IMAGES")
            index1, mean1 = get_representative_index(num_mask_list)
            index2, mean2 = get_representative_index(max_area_list)
            
            print("Number of maskse - mean: ",np.round(mean1,2))
            plot_masked_image(image_list[index1], masks_list[index1])
            
            print("Size of largest mask - mean: ",np.round(mean2,2))
            plot_masked_image(image_list[index2], masks_list[index2])

            
        image_compl_dict[dataset] = [np.mean(num_mask_list),np.mean(avg_size_list),np.mean(max_area_list),np.std(num_mask_list),np.std(avg_size_list),np.std(max_area_list)]
        
    image_compl_dict['Features'] = ['Avg_NumberOfSegments','Avg_Avg_SizeOfSegments','Avg_SizeOfLargest_Segment','Std_Num_Segments','Std_Avg_SizeOfSegments','Std_SizeOfLargest_Segment']
        
    segall_df=pd.DataFrame.from_dict(image_compl_dict,orient='index').transpose().set_index('Features').transpose()
    #segall_div_df=pd.DataFrame.from_dict(image_div_dict,orient='index').transpose().set_index('Features').transpose()
    
    return segall_df   



#Composition-based features

#Asymmetry
def calc_symmetrie(data_image_dict, center=False, visualize=False):
    result_dict = dict()
    result_dict_simple = dict()
    for dataset, image_list in data_image_dict:
        ssi_ver, ssi_hor, ssi_comb = list(), list(), list()
        result_dict[dataset] = list()
        result_dict_simple[dataset] = list()
        min_struct = 1
        max_struct = 0
        max_images, min_images, hor_images, ver_images = [], [], [], []

        for i, image in enumerate(image_list):
            if dataset in ['Shells','Skulls', 'Poke'] and center == True:
                image = crop_background(image)
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hor_flip = cv2.flip(image,0)
            ver_flip = cv2.flip(image,1)

            hor_images.append(hor_flip)
            ver_images.append(ver_flip)

            ssi_ver.append((ssim(image[:,:,0], ver_flip[:,:,0]) + ssim(image[:,:,1], ver_flip[:,:,1]) + ssim(image[:,:,2], ver_flip[:,:,2]) )/3)
            ssi_hor.append((ssim(image[:,:,0], hor_flip[:,:,0]) + ssim(image[:,:,1], hor_flip[:,:,1]) + ssim(image[:,:,2], hor_flip[:,:,2]) )/3)
            
            ssi_comb.append((ssi_ver[i]+ssi_hor[i])/2)

                           

            if ssi_comb[i] < min_struct:
                min_struct = ssi_comb[i]
                min_images = [image, ver_flip, hor_flip]

            if max_struct < ssi_comb[i]:
                max_struct = ssi_comb[i]
                max_images = [image, ver_flip, hor_flip]
                
            
        if visualize == True:
            print("Min:")
            for min_image in min_images:
                plt.imshow(min_image)
                plt.axis('off')
                plt.show()
            
            print("Max:")
            for max_image in max_images:
                plt.imshow(max_image)
                plt.axis('off')
                plt.show()
            
            index1, mean1 = get_representative_index(ssi_comb)
            plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
            plt.imshow(hor_images[i])
            plt.axis('off')
            plt.show()
            
            plt.imshow(ver_images[i])
            plt.axis('off')
            plt.show()
            
            
        result_dict[dataset] = [np.mean(ssi_ver), np.mean(ssi_hor), (np.mean(ssi_comb))/2, np.std(ssi_ver), np.std(ssi_hor), (np.std(ssi_comb))/2, kurtosis(ssi_ver), kurtosis(ssi_hor), (kurtosis(ssi_comb))/2]
        result_dict[dataset] = [round(x, 3) for x in  result_dict[dataset]]
        result_dict_simple[dataset] = [(np.mean(ssi_comb))/2, (np.std(ssi_comb))/2]
        result_dict_simple[dataset] = [round(x, 3) for x in  result_dict_simple[dataset]]
           
    

    result_dict['Features'] =['Mean_SSI_ver','Mean_SSI_hor','Mean_SSI_comb','Std_SSI_ver','Std_SSI_hor','Std_SSI_comb','kurt_SSI_ver','kurt_SSI_hor','kurt_SSI_comb' ]
    result_dict_simple['Features'] = ['Mean_Asymmetry','Std_Asymmetry']
    result_df= pd.DataFrame.from_dict(result_dict,orient='index').transpose().set_index('Features').transpose()
    result_df_simple = pd.DataFrame.from_dict(result_dict_simple,orient='index').transpose().set_index('Features').transpose()

  
    
    return result_df, result_df_simple

#Multiscale Entropy and helper Methods
def coarse_grain_image(image, scale):
    coarse_grained_image=cv2.pyrDown(image)
    return coarse_grained_image

def shannon(image):
    h_ent = shannon_entropy(image[:,:,0])
    s_ent = shannon_entropy(image[:,:,1])
    v_ent = shannon_entropy(image[:,:,2])
    return h_ent, s_ent, v_ent, (h_ent+s_ent+v_ent)/3

def multiscale_entropy(image, max_scale):
    # Calculate sample entropy for each coarse-grained time series
    hue_ent, sat_ent, bri_ent = [], [], []
    coarse_grained_image = image
    for scale in range(1, max_scale + 1):
        hue_ent.append(shannon_entropy(coarse_grained_image[:,:,0]))
        sat_ent.append(shannon_entropy(coarse_grained_image[:,:,1]))
        bri_ent.append(shannon_entropy(coarse_grained_image[:,:,2]))
        coarse_grained_image = coarse_grain_image(coarse_grained_image, scale)
            
    # Calculate multiscale entropy as the average of the sample entropies
    mse_hue = np.mean(hue_ent)
    mse_sat = np.mean(sat_ent)
    mse_bri = np.mean(bri_ent)
    
    mse = (mse_hue + mse_sat + mse_bri)/3
    return mse_hue, mse_sat, mse_bri, mse

def compute_mnf(channel):
    # Fourier Transform
    f_transform = np.fft.fft(channel)
    
    # Power Spectrum
    power_spectrum = np.abs(f_transform) ** 2
    
    # Frequencies
    frequencies = np.fft.fftfreq(len(channel))
    
    # Consider only positive frequencies
    positive_indices = np.where(frequencies > 0)
    
    # Mean Frequency (MNF)
    numerator = np.sum(frequencies[positive_indices] * power_spectrum[positive_indices])
    denominator = np.sum(power_spectrum[positive_indices])
    
    # Adding epsilon to avoid division by zero
    epsilon = 1e-8
    mnf = numerator / (denominator + epsilon)
    
    return mnf

def compute_mnf_2d(image):
    # Assuming the input image is 2D (grayscale)
    # First apply MNF to each column
    mnf_columns = np.apply_along_axis(compute_mnf, 0, image)
    
    # Then apply MNF to the resulting vector of mean frequencies
    mnf_final = compute_mnf(mnf_columns)
    
    return mnf_final

def mnf_rgb(rgb_image):
    # Assuming the input is an RGB image with shape (height, width, 3)
    # Split the channels
    red_channel, green_channel, blue_channel = np.split(rgb_image, 3, axis=-1)
    
    # Squeeze the singleton dimension
    red_channel = np.squeeze(red_channel)
    green_channel = np.squeeze(green_channel)
    blue_channel = np.squeeze(blue_channel)
    
    # Compute MNF for each channel
    mnf_red = compute_mnf_2d(red_channel)
    mnf_green = compute_mnf_2d(green_channel)
    mnf_blue = compute_mnf_2d(blue_channel)
    
    return mnf_red, mnf_green, mnf_blue, (mnf_red + mnf_green + mnf_blue)/3

import numpy as np
from skimage import measure

def perimetric_complexity(binary_image):
    # Ensure the image is a binary numpy array
    binary_image = np.asarray(binary_image).astype(bool)
    
    # Compute the perimeter using the contours
    contours = measure.find_contours(binary_image, level=0.8)
    perimeter = sum([len(contour) for contour in contours])
    
    # Compute the area by simply summing up the foreground pixels
    area = np.sum(binary_image)
    
    # Compute Perimetric Complexity
    PC = (perimeter ** 2) / (4 * np.pi * area)
    
    return PC


def delentropy(img):
    # Ensure the image is a numpy array
    #image = np.asarray(image)
    DE_h, DE_s, DE_v = 0, 0, 0
    image_list = [img[:,:,0],img[:,:,1],img[:,:,2]]
    
    # Get the width and height of the image
    W, H = image_list[0].shape
    
    # Define derivative kernels
    dx_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    dy_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    
    for i, image in enumerate(image_list):
        # Compute the gradients in x and y direction
        dx = np.gradient(image, axis=1)
        dy = np.gradient(image, axis=0)

        # Discretize the gradient values (quantization)
        dx = np.round(dx).astype(int)
        dy = np.round(dy).astype(int)

        # Determine the range of gradient values for binning
        I = np.ptp(dx) + 1  # Range of dx values
        J = np.ptp(dy) + 1  # Range of dy values

        # Compute the normalized joint histogram (deldensity)
        p_ij = np.zeros((I, J))
        for w in range(W):
            for h in range(H):
                i = dx[w, h]
                j = dy[w, h]
                p_ij[i, j] += 1
        p_ij /= (0.25 * W * H)

        # Compute the Delentropy (DE)
        DE = -0.5 * np.sum(p_ij * np.log2(p_ij + 1e-9))  # Adding small value to avoid log(0)
        
        if i == 0:
            DE_h = DE
        elif i == 1:
            DE_s = DE
        elif i == 2:
            DE_v = DE
            
    avg_DE = (DE_h+DE_s+DE_v)/3
    
    return  DE_h, DE_s, DE_v, avg_DE

def calculate_delentropy(image, num_bins=256):
    # Calculate the gradient of the image
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

    # Compute the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(gradient_x.ravel(), gradient_y.ravel(), bins=num_bins)

    # Normalize the histogram to create a probability density function
    hist /= hist.sum()

    # Compute the delentropy
    delentropy = -0.5 * np.nansum(hist*np.log2(hist + np.finfo(float).eps))

    return delentropy


def calc_ent(data_image_dict, channels='rgb',method='shannon', center=False, plot=False):
    mul_ent_dict = {}
    minmax_list = []
    for dataset, image_list in data_image_dict:
        mul_ent_dict[dataset] = [0]*15
        hue_list, sat_list, bri_list, mse_list= [], [], [], []
        min_mse = 100
        max_mse = -100
        for image in image_list:
            if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                image = crop_background(image)
                
            if channels=='rgb':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            if method == 'multiscale':
                hue, sat, bri, mse = multiscale_entropy(image, 5) 
            elif method == 'delentropy':
                channels = cv2.split(image)
                # Calculate the delentropy for each channel
                delentropy = [calculate_delentropy(channel) for channel in channels]
                hue, sat, bri, mse = delentropy[0], delentropy[1], delentropy[2], (delentropy[0] + delentropy[1] + delentropy[2])/3 
            elif method == 'shannon':
                hue, sat, bri, mse = shannon(image) 
            elif method == 'mean_frequency':
                hue, sat, bri, mse = mnf_rgb(image) 
            elif method == 'peri' and channels =='rgb':
                mse = perimetric_complexity(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                hue, sat, bri = mse, mse, mse
            else:
                print("no available method chosen")
            hue_list.append(hue)
            sat_list.append(sat)
            bri_list.append(bri)
            mse_list.append(mse)

            if mse < min_mse:
                min_mse = mse
                min_img = image
            if max_mse < mse:
                max_mse = mse
                max_img = image

        image_feature_list = [np.mean(hue_list), np.median(hue_list), np.std(hue_list), np.mean(sat_list), np.median(sat_list), np.std(sat_list), np.mean(bri_list), np.median(bri_list), np.std(bri_list), np.mean(mse_list), np.median(mse_list), np.std(mse_list), np.max(mse_list), np.min(mse_list), np.max(mse_list)-np.min(mse_list)]
        fill_dict(mul_ent_dict[dataset], [round(x,2) for x in image_feature_list])
        minmax_list.append(min_img)
        minmax_list.append(max_img)
        
        if plot:
            index1, mean1 = get_representative_index(mse_list)
            plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    mul_ent_dict['Features'] = ['Mean_hue','Median_hue','Std_hue','Mean_sat','Median_sat','Std_sat','Mean_bri','Median_bri','Std_bri','Mean_Entropy','Median_mse','Std_Entropy','Max_mse','Min_mse','Range_mse']
    mul_ent_df=pd.DataFrame.from_dict(mul_ent_dict,orient='index').transpose().set_index('Features').transpose()

    #mul_ent_df.to_csv('./results/multiscale_entropy_hsv.csv') 
    if plot:
        for image in minmax_list:
            plt.imshow(image)
            plt.axis('off')
            plt.show()
    if method == 'multiscale':
        return mul_ent_df, mul_ent_df[['Mean_Entropy','Std_Entropy']].rename(columns={'Mean_Entropy':'Avg_MultiscaleEntropy','Std_Entropy':'Std_MultiscaleEntropy'})
    elif method == 'delentropy':
        return mul_ent_df, mul_ent_df[['Mean_Entropy','Std_Entropy']].rename(columns={'Mean_Entropy':'Avg_Deltropy','Std_Entropy':'Std_DelEntropy'})
    elif method == 'mean_frequency':
        return mul_ent_df, mul_ent_df[['Mean_Entropy','Std_Entropy']].rename(columns={'Mean_Entropy':'Avg_MeanFrequency','Std_Entropy':'Std_MeanFrequency'})
    else:
        return mul_ent_df, mul_ent_df[['Mean_Entropy','Std_Entropy']]


#EDGE DETECTION
def edge_det(data_image_dict, center=False, plot=False):
    edge_dict = {}
    minmax_list = []

    for dataset, image_list in data_image_dict:
        edge_dict[dataset] = [0]*7
        edge_list, edge_image_list = [], []
        min_edges = 1000000000
        max_edges = 0
        for image in image_list:
            if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                image = crop_background(image)
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 150)
            num_edges = cv2.countNonZero(edges)
            edge_list.append(num_edges)
            edge_image_list.append(edges)

            if num_edges < min_edges:
                min_edges = num_edges
                min_img = image
                min_edge_img = edges
            if max_edges < num_edges:
                max_edges = num_edges
                max_img = image
                max_edge_img = edges

        pre_len = len(edge_list)
        #edge_list = reject_outliers(np.asarray(edge_list),5.4)
        image_feature_list = [np.mean(edge_list),np.median(edge_list),np.std(edge_list),min_edges, max_edges, max_edges-min_edges, pre_len-len(edge_list)]

        fill_dict(edge_dict[dataset], [round(x,2) for x in image_feature_list])

        minmax_list.extend([min_img, min_edge_img])
        minmax_list.extend([max_img, max_edge_img])
        
        if plot:
            index1, mean1 = get_representative_index(edge_list)
            plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.imshow(cv2.cvtColor(edge_image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    edge_dict['Features'] = ['Mean_edges','Median_edges','Std_edges','Min_edges','Max_edges','Range_edges','Outliers']
    edge_df=pd.DataFrame.from_dict(edge_dict,orient='index').transpose().set_index('Features').transpose()

    if plot:
        for image in minmax_list:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
    return edge_df, edge_df[['Mean_edges','Std_edges']]


#https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6603194
def calculate_spatial_information(image):
    # Read the image in grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the Sobel operator in the x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the magnitude of spatial information at each pixel
    SIr = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calculate the total number of pixels in the image
    P = image.size
    
    # Calculate the mean spatial information
    SImean = np.sum(SIr) / P
    
    # Calculate the root-mean-square spatial information
    SIrms = np.sqrt(np.sum(SIr**2) / P)
    
    # Calculate the standard deviation of spatial information
    SIstdev = np.sqrt(np.sum(SIr**2) / P - SImean**2)
    
    return SImean, SIrms, SIstdev


#Spatial Information
def spatial(data_image_dict, center=False, plot=False):
    result_dict = dict()
    for dataset, image_list in data_image_dict:
        simean_list, sirms_list, sistd_list = list(), list(), list()
        min_SI, max_SI = 10000, 0
        minmax_list = list()
        
        for image in image_list:
            if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                image = crop_background(image)
            # Calculate spatial information
            SImean, SIrms, SIstdev = calculate_spatial_information(image)
            simean_list.append(SImean)
            sirms_list.append(SIrms)
            sistd_list.append(SIstdev)
            
            if SImean < min_SI:
                min_SI = SImean
                min_img = image
            
            if max_SI < SImean:
                max_SI = SImean
                max_img = image

        #result_dict[dataset] = [np.mean(simean_list), np.std(simean_list), np.mean(sirms_list), np.std(sirms_list), np.mean(sistd_list),np.std(sistd_list)]
        result_dict[dataset] = [np.mean(simean_list), np.std(simean_list)]
      
        if plot:
            index1, mean1 = get_representative_index(simean_list)
            plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
        minmax_list = [max_img, min_img]
        if plot:
            for image in minmax_list:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

    result_dict['Features'] = ['Avg_SpatialInformation','Std_SpatialInformation']
    result_df= pd.DataFrame.from_dict(result_dict,orient='index').transpose().set_index('Features').transpose()

    display(result_df)
    return result_df


def fractal_dimension(image):   
    # Only for 2d image
    assert(len(image.shape) == 2)

    # Box sizes
    sizes = 2**np.arange(8, 0, -1)

    # Box counting for each size
    counts = []
    for size in sizes:
        count = 0
        for i in range(0, image.shape[0], size):
            for j in range(0, image.shape[1], size):
                patch = image[i:i+size, j:j+size]
                if np.max(patch) > 0:
                    count += 1

        counts.append(count)

    # Fit the sizes and counts to a linear regression model
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]

def fractal(data_image_dict, center=True, plot=False):
    result_dict = dict()
    
    for dataset, image_list in data_image_dict:
        result_list = list()
        min_fd, max_fd = 10000, 0
        minmax_list=list()

        for image in image_list:
            if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                    image = crop_background(image)

            # Convert to grayscale for binarization
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Binarize the image for simplicity
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

            # Calculate fractal dimension for each color channel
            fds = []
            for i in range(3):
                # Apply binary mask to color channel
                channel = cv2.bitwise_and(image[:,:,i], image[:,:,i], mask=binary)

                fd = fractal_dimension(channel)
                fds.append(fd)

            # Average fractal dimensions of all channels
            fd = np.mean(fds)
        
            result_list.append(fd)
            
            if fd < min_fd:
                min_fd = fd
                min_img = image
            
            if max_fd < fd:
                max_fd = fd
                max_img = image
        
       
        result_dict[dataset] = [np.mean(result_list), np.std(result_list)] 
        minmax_list = [max_img, min_img]
        if plot:
            index1, mean1 = get_representative_index(result_list)
            plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
        if plot:
            for image in minmax_list:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
        
        
    result_dict['Features'] = ['Avg_Fractal','Std_Fractal']
    result_df = pd.DataFrame.from_dict(result_dict,orient='index').transpose().set_index('Features').transpose()
    
    return  result_df


#TEXTURE-BASED
#1) GLCM
def discrete_degree(glcm):
    glcm = glcm[...,np.newaxis]
    rows, cols, channels = glcm.shape
    row_indices, col_indices = np.indices((rows, cols))
    abs_diff = np.abs(row_indices - col_indices)
    weighted_glcm = glcm.squeeze(2) * abs_diff
    return np.sum(weighted_glcm)


def calculate_complexity_values(glcm):
    #First calculate the sum of values in each diagonal:
    diagonal_list, discrete_degree_list, non_zero_list = [[]], [[]], [[]]

    for i in range(glcm.shape[3]):
        for j in range(glcm.shape[2]):
            diagonal_list[0].append(np.sum([glcm[k,k,j,i] for k in range(glcm.shape[0])]))
            discrete_degree_list[0].append(discrete_degree(glcm[:,:,j,i]))
            non_zero_list[0].append(np.count_nonzero(glcm[:,:,j,i]))

    
    return np.asarray(diagonal_list, dtype=np.float64), np.asarray(discrete_degree_list, dtype=np.float64), np.asarray(non_zero_list, dtype=np.float64)
    
#TODO: really symmetric and normed?
def glcm_features(image, distances=[1], angles=[0,45,90,145], levels=32, only = False):
    if only is False:
        image = (image * (levels / 256)).astype(np.uint8)
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    else:
        glcm = image

    contrast = graycoprops(glcm, 'contrast')
    correlation = graycoprops(glcm, 'correlation')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    entropy = -np.sum(glcm * np.log2(glcm + 1e-6), axis=(0, 1))
    
    #glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=False)
    dia, disc_degree, non_zero = calculate_complexity_values(glcm)
    
    
    return [contrast, correlation, energy, homogeneity, entropy, dia, disc_degree, non_zero]

def glcm_color_features(image, distance, angles, levels):
    feature_list = [0,0,0,0,0,0,0,0]
    for i in range(image.shape[2]):
        feature_list = [x + y for x, y in zip(feature_list, glcm_features(image[:,:,i], distance, angles, levels))]
    return list(np.asarray(feature_list)/3)



def multispectral_method(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image / levels).astype(np.uint8)
    channels = [0, 1, 2]
    channel_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    feature_list = [0,0,0,0,0,0,0,0] 

    co_occurrence_matrices = []
    for cu, cv in channel_pairs:
        co_occurrence_matrix = np.zeros((levels, levels, len(angles)), dtype=np.uint32)
        for angle in angles:
            co_occurrence_matrix_uv = graycomatrix(image[:, :, cu], distances, angles, levels=levels, symmetric=False, normed=False).squeeze(2)
            if cu == cv:
                co_occurrence_matrix += co_occurrence_matrix_uv
            else:
                co_occurrence_matrix_vu = graycomatrix(image[:, :, cv], distances, angles, levels=levels, symmetric=False, normed=False).squeeze(2)
                co_occurrence_matrix += co_occurrence_matrix_uv + np.transpose(co_occurrence_matrix_vu, axes=(1, 0, 2))
        co_occurrence_matrices.append(np.expand_dims(co_occurrence_matrix,2) / np.sum(co_occurrence_matrix))
    
    for co_occ_mat in co_occurrence_matrices:
        feature_list = [x + y for x, y in zip(feature_list, glcm_features(co_occ_mat, distances, angles, levels, only=True))]
    
    return list(np.asarray(feature_list)) 


def det_max(degree_list):
    return det_degree(np.argmax(degree_list))

def det_min(degree_list):
    return det_degree(np.argmin(degree_list))
    
def det_degree(pos):
    if pos == 0:
        return "0"
    elif pos == 1:
        return "45"
    elif pos == 2:
        return "90"
    else:
        return "145"
    

#TODO: Save and print min and max images
#TODO: better cheng and haralick scores (over normalized columns)
from skimage.feature import graycomatrix, graycoprops
def calculate_glcm_features(data_image_dict, color_levels = 128, method='GLCM', distance = [1,2,4], center =True, plot=False):
    coocc_dict = {}
    for dataset, image_list in data_image_dict:
        print("\n",dataset)
        coocc_dict[dataset] = [0]*54
        #contrast_sum, correlation_sum, energy_sum, homogeneity_sum, entropy_sum, diagonal_sum, discrete_degree_sum, non_zero_sum 
        feature_list = [[], [], [], [], [], [], [], []]
        for image in image_list:
            #image = cv2.resize(image,(256,256))
            if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                image = crop_background(image)
            
            if method == 'GLCM':
                calculated_features = glcm_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), distance, [0, 45, 90, 135], levels=color_levels)
            elif method == 'COLOR':
                calculated_features = glcm_color_features(image, distance, [0, 45, 90, 135], levels=color_levels)
            elif method == 'MULT':
                calculated_features = multispectral_method(image, distance, [0, 45, 90, 135], levels=color_levels)
            else:
                #calculated_features = extract_micicm(image, 12, 16, 0.125, 0.125, 0.125, 0.125)
                calculated_features = extract_micicm(image, 30, 32, 0.125, 0.125, 0.125, 0.125)
            
            for i in range(len(feature_list)):
                feature_list[i].append((calculated_features[i]))                    
                 
            
        mean_list = [round(np.mean([np.mean(feature) for feature in fl]),2) for  i, fl in enumerate(feature_list)]
        #median_list = [round(np.median([np.mean(feature) for feature in fl]),2) for  i, fl in enumerate(feature_list)]
        std_list = [round(np.std([np.mean(feature) for feature in fl]),2) for  i, fl in enumerate(feature_list)]
        skew_list = [round(skew([np.mean(feature) for feature in fl]),2) for  i, fl in enumerate(feature_list)]
        
        #First, we itearte through the different features (contrast, correlation...)#
        if plot :
            for i, fl in enumerate(feature_list):
                #For each of those features we have a list of lists
                feat_list = [np.mean(feature) for feature in fl]
                feat_list_max = np.argmax(feat_list)
                feat_list_min = np.argmin(feat_list)
                index1, mean1 = get_representative_index(feat_list)

                print("REPRESENTATIVE - MAX - MIN")
                plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                plt.imshow(cv2.cvtColor(image_list[feat_list_max], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                plt.imshow(cv2.cvtColor(image_list[feat_list_min], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            

                                 
            
        image_feature_list = sum([mean_list, std_list],[])
        #image_feature_list.extend([haralick_avg_complexity, haralick_std_score, haralick_skew_score, cheng_complexity, cheng_std_score, cheng_skew_score])
        
        #fill_dict(coocc_dict[dataset], image_feature_list)
        coocc_dict[dataset] = image_feature_list
    
    coocc_dict['Features'] = ['Mean_contrast','Mean_corr','Mean_energy','Mean_homo','Mean_entropy','Mean_diag','Mean_disc','Mean_zero',
                        'Std_contrast','Std_corr','Std_energy','Std_homo','Std_entropy','Std_diag','Std_disc','Std_zero']
    coocc_df=pd.DataFrame.from_dict(coocc_dict,orient='index').transpose().set_index('Features').transpose()
    #display(coocc_df)
    
    
    #coocc_df.to_csv('./results/coocc_{}_d{}_diversity.csv'.format(method, distance))
    return coocc_df


#2) LBP
'''from skimage.feature import local_binary_pattern

def hist(axis, lbp):
    
    #Create a histogram
    #:param axis: matplotlib axes
    #:param lbp: ndarray local binary pattern representation of an image
    #:return: matplotliob histogram
    
    n_bins = int(lbp.max() + 1) # number of bins based on number of different values in lbp
    return axis.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5') # np.ravel() returns a flattened 1D array

def calculate_entropy(hist):
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def calculate_energy(hist):
    energy = np.sum(hist ** 2)
    return energy
def is_uniform_pattern(bin_number, n_points):
    """
    Check if a pattern is uniform.
    bin_number: integer, the decimal number representing the pattern.
    n_points: integer, number of points in the pattern.
    """
    binary_str = format(bin_number, '0' + str(n_points) + 'b')
    transitions = sum(binary_str[i] != binary_str[i-1] for i in range(1, n_points)) + (binary_str[-1] != binary_str[0])
    return transitions <= 2

def calculate_uniform_patterns(hist, n_points):
    """
    Count the number of uniform patterns in the histogram.
    hist: array-like, the LBP histogram.
    n_points: integer, number of points in the pattern.
    """
    uniform_count = 0
    for bin_number, bin_value in enumerate(hist):
        if is_uniform_pattern(bin_number, n_points):
            uniform_count += bin_value
    return uniform_count

def calculate_uniform_patterns_ratio(hist, n_points):
    """
    Calculate the ratio of uniform patterns to the total number of patterns in the histogram.
    hist: array-like, the LBP histogram.
    n_points: integer, number of points in the pattern.
    """
    uniform_count = 0
    total_count = 0
    for bin_number, bin_value in enumerate(hist):
        total_count += bin_value
        if is_uniform_pattern(bin_number, n_points):
            uniform_count += bin_value
    
    if total_count == 0:
        return 0
    else:
        return uniform_count / total_count

# Modify the apply_lbp function
def apply_lbp2(data_image_dict, r=2, center=True):
    method='uniform'
    radius = r
    n_points = 8 * radius
    
    result_dict = dict()
    result_hist_dict = dict()
    
    datasets = list()
    
    # assuming data_image_dict is defined elsewhere
    for dataset, image_list in data_image_dict:
        datasets.append(dataset)
        img_count = 0
        result_hist_dict[dataset] = list()
        
        entropy_list, energy_list, uniformity_list = list(), list(), list()
        
        for i,image in enumerate(image_list):
            # Convert to grayscale
            
            if dataset in ['shells96','skulls96', 'pokemon96_100'] and center ==True:
                image = crop_background(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Compute LBP
            lbp_features = local_binary_pattern(image, n_points, radius, method)
            
            # Compute histogram
            n_bins = int(lbp_features.max() + 1)
            hist, _ = np.histogram(lbp_features, bins=n_bins, density=True, range=(0, n_bins))
            
            # Calculate and print entropy, energy and uniform patterns
            entropy = calculate_entropy(hist)
            energy = calculate_energy(hist)
            uniform_patterns = calculate_uniform_patterns_ratio(hist, n_points)
            
            entropy_list.append(entropy)
            energy_list.append(energy)
            uniformity_list.append(uniform_patterns)
            
            result_hist_dict[dataset].append(hist)
            
        result_dict[dataset] = [np.mean(entropy_list), np.std(entropy_list),np.mean(energy_list), np.std(energy_list), np.mean(uniformity_list), np.std(uniformity_list),]
        
    result_dist_dict = dict()
    for dataset in datasets:
        jensen_list = list()
        chi_list = list()
        for i in range(len(result_hist_dict[dataset])):
            for j in range(i+1,len(result_hist_dict[dataset])):
                jensen_list.append(jensen_shannon(result_hist_dict[dataset][i],result_hist_dict[dataset][j]))
                chi_list.append(chi_squared_distance(result_hist_dict[dataset][i],result_hist_dict[dataset][j]))
            
        result_dist_dict[dataset] = [np.mean(jensen_list), np.std(jensen_list), np.mean(chi_list), np.std(chi_list)]
        
    result_dict['Features'] = ['AvgEnt','StdEnt','AvgEnergy','StdEnergy','AvgUni','StdUni']
    result_df=pd.DataFrame.from_dict(result_dict,orient='index').transpose().set_index('Features').transpose()
    
    result_dist_dict['Features'] = ['AvgJenDist','StdJenDist','AvgChiDist','StdChiDist']
    dist_df=pd.DataFrame.from_dict(result_dist_dict,orient='index').transpose().set_index('Features').transpose()
    
    display(result_df)
    display(dist_df)
    
    return result_df, dist_df'''

#3) Garbor filter responses
def build_filters_with_increasing_bandwidth():
    filters = []
    ksize = 31
    theta = np.pi / 4  # fixed orientation
    for sigma in np.linspace(1.0, 10.0, 10):  # increasing spatial bandwidth
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, 1.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def apply_filters_and_calculate_slope(img, filters):
    responses = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        responses.append(np.mean(fimg))
    
    # Calculate the slope of the responses
    slope, _, _, _, _ = stats.linregress(range(len(responses)), responses)
    
    return slope


def build_filters():
    filters = []
    ksize = 21
    for theta in np.arange(0, np.pi, np.pi / 4):  # different orientations
        for sigma in np.linspace(1.0, 10.0, 10):  # different scales
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, 1.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

def apply_filters_and_calculate_variances(img, filters):
    variances = []
    for filter in filters:
        filtered_image = cv2.filter2D(img, cv2.CV_8UC3, filter)
        variances.append(np.var(filtered_image))
    return variances

def garbor_features(data_image_dict, center=True, plot=False):
    # Define the orientations and scales for the Gabor filters
    filters_fixed_orientation = build_filters_with_increasing_bandwidth()
    filters_variable = build_filters()
    result_dict = dict()
    
    for dataset, image_list in data_image_dict:
        response_consistency_list, local_variance_list = list(), list()
        
        for image in image_list:
            if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                image = crop_background(image)
            response_consistency_list.append(apply_filters_and_calculate_slope(image, filters_fixed_orientation))
            local_variance_list.append(np.mean(apply_filters_and_calculate_variances(image, filters_variable)))
            
        if plot :
            for feat_list in [response_consistency_list, local_variance_list]:
                feat_list_max = np.argmax(feat_list)
                feat_list_min = np.argmin(feat_list)
                index1, mean1 = get_representative_index(feat_list)

                print("REPRESENTATIVE - MAX - MIN")
                plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                plt.imshow(cv2.cvtColor(image_list[feat_list_max], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                plt.imshow(cv2.cvtColor(image_list[feat_list_min], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
        result_dict[dataset] = [np.mean(response_consistency_list), np.std(response_consistency_list), np.mean(local_variance_list), np.std(local_variance_list)]
    
    result_dict['Features'] = ['Avg_garbor_response_consistency','Std_garbor_response_consistency','Avg_garbor_local_variance','Std_garbor_local_variance']
    result_df = pd.DataFrame.from_dict(result_dict,orient='index').transpose().set_index('Features').transpose()
    return result_df



#Manifold-based
#Garbor filter variance
from skimage import io, filters, color
import numpy as np



from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def intrinsic_dim_sample_wise(X, k=5):
    neighb = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]  # Exclude the point itself
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    
    # Compute the maximum likelihood estimate of intrinsic dimensionality
    log_ratios = np.log(dist[:, -1, np.newaxis] / dist[:, :-1])
    intdim_sample = (1.0 / (k - 1)) * np.sum(log_ratios, axis=1)
    intdim_sample = 1.0 / intdim_sample
    
    return intdim_sample

def intrinsic_dim_scale_interval(X, k1=10, k2=20):
    X = pd.DataFrame(X).drop_duplicates().values  # remove duplicates in case you use bootstrapping
    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k

def repeated(func, X, nb_iter=100, random_state=None, verbose=0, mode='bootstrap', **func_kw):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results

def intrinsic_dimension(data_image_dict, channels='rgb', center=False, plot=False):
    result_dict = dict()
    # Load your images
    for dataset, image_list in data_image_dict:
        # Flatten images and stack them into a matrix X
        k1 = 10  # start of interval(included)
        k2 = 20  # end of interval(included)
        if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                X = np.array([crop_background(image).flatten() for image in image_list])
        else:
            X = np.array([image.flatten() for image in image_list])

        intdim_k_repeated = repeated(intrinsic_dim_scale_interval,
                                 X,
                                 mode='same',
                                 nb_iter=10,  # number of bootstrap iterations
                                 verbose=1,
                                 k1=k1, k2=k2)

        # Convert to numpy array for convenience
        intdim_k_repeated = np.array(intdim_k_repeated)
        # Use intrinsic_dim_scale_interval with your data matrix X

        #intdim_k = intrinsic_dim_scale_interval(X, k1=k1, k2=k2)

        # Printing the results
        print(dataset)
        print(np.mean(intdim_k_repeated))
        if plot :
            feat_list_max = np.argmax(intdim_k_repeated)
            feat_list_min = np.argmin(intdim_k_repeated)
            index1, mean1 = get_representative_index(intdim_k_repeated)

            print("REPRESENTATIVE - MAX - MIN")
            plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.imshow(cv2.cvtColor(image_list[feat_list_max], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.imshow(cv2.cvtColor(image_list[feat_list_min], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        result_dict[dataset] = [np.mean(intdim_k_repeated), np.std(intdim_k_repeated)]

    result_dict['Features'] = ['Mean_IntrinsicDimension','STD_IntrinsicDimension']
    result_df= pd.DataFrame.from_dict(result_dict,orient='index').transpose().set_index('Features').transpose()

    #display(result_df)
    #result_df.to_csv('./results/intrinsic_dimension.csv')
    return result_df





def compute_saliency_map(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # add batch dimension

    # Make sure the image requires gradient
    image.requires_grad = True

    # Forward pass
    output = model(image)

    # Get the index of the max log-probability
    _, predicted_class = torch.max(output, 1)

    # Compute the saliency map
    output[0, predicted_class].backward()
    saliency_map, _ = torch.max(image.grad.data.abs(), dim=1)
    saliency_map = saliency_map.squeeze().cpu().numpy()

    return saliency_map

def saliency_map(image_data_dict, center=True, plot=False):
    result_dict = dict()
    
    for dataset, image_list in image_data_dict:
        overall_size_list, mean_size_list, std_size_list = list(), list(), list()
        saliency_map_list = list()
        image_count = 0
        for image in image_list:
            if dataset in ['Shells','Skulls', 'Poke'] and center ==True:
                    image = crop_background(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Get the saliency map
            saliency_map = compute_saliency_map(image, model)
            # Threshold the saliency map to create a binary mask
            _, mask = cv2.threshold(saliency_map, 0.05, 1, cv2.THRESH_BINARY)
            

            # Label connected components in the mask
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

            # Compute the size of each connected component (salient region)
            sizes = stats[1:, -1]

            # Compute the mean and standard deviation of the sizes
            overall_size = np.sum(mask)
            mean_size = np.mean(sizes)
            std_dev_size = np.std(sizes)
            
            overall_size_list.append(overall_size)
            mean_size_list.append(mean_size)
            std_size_list.append(len(sizes))
            
            image_count += 1
            
            saliency_map_list.append(mask)
                                   
            
            
        #print(np.mean(intdim_k_repeated))
        if plot :
            feat_list_max = np.argmax(std_size_list)
            feat_list_min = np.argmin(std_size_list)
            index1, mean1 = get_representative_index(std_size_list)

            print("REPRESENTATIVE - MAX - MIN")
            plt.imshow(cv2.cvtColor(image_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.imshow(cv2.cvtColor(image_list[feat_list_max], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.imshow(cv2.cvtColor(image_list[feat_list_min], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
            print("REPRESENTATIVE - MAX - MIN")
            plt.imshow(cv2.cvtColor(saliency_map_list[index1], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.imshow(cv2.cvtColor(saliency_map_list[feat_list_max], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.imshow(cv2.cvtColor(saliency_map_list[feat_list_min], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
        result_dict[dataset] = [np.mean(overall_size_list), np.mean(std_size_list), np.std(overall_size_list), np.std(std_size_list)]
    
    result_dict['Features'] = ['Avg_saliency_size','Avg_num_connectedSaliency','Std_saliency_size','Std_num_connectedSaliency']
    result_df=pd.DataFrame.from_dict(result_dict,orient='index').transpose().set_index('Features').transpose()
    return result_df

            
            
def compute_saliency_map(image, model):
    # Load and preprocess the image
    image = image = Image.fromarray(image.astype('uint8'), 'RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # add batch dimension

    # Make sure the image requires gradient
    image.requires_grad = True

    # Forward pass
    output = model(image)

    # Get the index of the max log-probability
    _, predicted_class = torch.max(output, 1)

    # Compute the saliency map
    output[0, predicted_class].backward()
    saliency_map, _ = torch.max(image.grad.data.abs(), dim=1)
    saliency_map = saliency_map.squeeze().cpu().numpy()

    return saliency_map








            

        
        
        



    






