import numpy as np
import seaborn as sns
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import cv2

from scipy.stats import kurtosis, skew
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from scipy import stats

from utils.image_feature_extraction_methods import *

import os


from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import figure
import seaborn as sns


# Removed num_to_groups
from IPython.display import display, HTML

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["axes.grid"] = False


def calculate_regression_values(feat_df, score_df, visualize=False, p_value=0.1, standardize=False, prediction_dict=None, show_corr=True, dataset_folder='./train_data'):
    #realign_features(feat_df)
    feat_df.rename(columns={'index':'Dataset'}, inplace=True)
    datasets = os.listdir(dataset_folder)
    if prediction_dict is not None:
        #datasets =  ['Grumpy','Cat','Dog','Panda','Influ','Moon','Anime','Poke','Art','Fauvism','Shells','Skulls','FFHQ','CelebA','Dining','Garbage']
        datasets.extend(list(prediction_dict.keys()))
        feat_df['Dataset'] = datasets
    else:
        feat_df['Dataset'] = datasets
        
        
    score_cols = score_df.columns[1:]
    feat_cols = feat_df.columns[:]
    result = pd.merge(feat_df, score_df, on="Dataset")
    corr_df =feat_df.drop(['Dataset'],axis=1)

    #Display feature correlation
    if show_corr:
        display(corr_df.corr(numeric_only=False).round(2))
    
    mean_list, std_list = list(), list()
    
    
    pred_rows = list()
    if prediction_dict is not None:
        print(list(prediction_dict.keys()))
        prediction_df = result[(result.Dataset.isin(list(prediction_dict.keys())))]
        regression_df = result
        for key in list(prediction_dict.keys()):
            regression_df = regression_df[regression_df.Dataset != key]
        
    else:
        regression_df = result
        
    print("DATASET")
    #display(prediction_df)
    #display(regression_df)
        
       
    #display(prediction_df)
    if standardize:
        for column in regression_df.columns:
            if column != 'Dataset':
                regression_df[column] = pd.to_numeric(regression_df[column])
                mean =  np.mean(regression_df[column].values)
                std = np.std(regression_df[column].values)
                regression_df.loc[:,column] = (regression_df.loc[:,column] - mean) / std
                if prediction_dict is not None and column in feat_df.columns:
                    feat_df[column] = pd.to_numeric(feat_df[column])
                    feat_df.loc[:,column] = (feat_df.loc[:,column] - mean) / std
                    
    else:
        for column in regression_df.columns:
            if column != 'Dataset':
                regression_df[column] = pd.to_numeric(regression_df[column])
                mean =  np.mean(regression_df[column].values)
                std = np.std(regression_df[column].values)
                regression_df.loc[:,column] = regression_df.loc[:,column] 
                if prediction_dict is not None and column in feat_df.columns:
                    feat_df[column] = pd.to_numeric(feat_df[column])
                    feat_df.loc[:,column] = feat_df.loc[:,column]
            
   
    quantiles = regression_df.quantile(q=[0.25, 0.5, 0.75], axis=0, numeric_only=True)  
    
    #display(regression_df)
    significant_features_list = list()
    
    #print("regre/feat:")
    #display(regression_df)
    #display(feat_df)
    #display(score_df)
        
    for feat_col in feat_cols:
        for score_col in score_cols:            
            string = score_col+"~"+feat_col
            regression_model = ols(string, data=regression_df).fit()
            
            if regression_model.pvalues[1] < 0.1:
                significant_features_list.append(feat_col)
            
           
            if regression_model.pvalues[1] < p_value:
                
                print('\n\n----------------------------------------------------------------------\n{} has significant impact on {}:\n'.format(feat_col,score_col))
                #print("correlation:\t",regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]))
                #print("p_value:\t", regression_model.pvalues[1])
                
                if regression_model.pvalues[1] < 0.01:
                    print(str(round(regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]),2))+'***')
                elif regression_model.pvalues[1] < 0.05:
                    print(str(round(regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]),2))+'**')
                elif regression_model.pvalues[1] < 0.1:
                    print(str(round(regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]),2))+'*')
                else:
                    print(round(regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]),2))
                print('('+str(round(regression_model.bse[1],2))+')')
                
                #print(regression_model.summary())
                
                if prediction_dict is not None:
                    for key in prediction_dict.keys():

                            
                        #Add results to the dictionary
                        difficulty = None

                        if feat_df.loc[feat_df.Dataset==key,feat_col].values[0] < quantiles.loc[0.25,feat_col]:
                            if regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]) < 0:
                                difficulty = "high"
                            else:
                                difficulty = "low"
                        
                        elif feat_df.loc[feat_df.Dataset==key,feat_col].values[0] < quantiles.loc[0.5,feat_col]:
                            if regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]) < 0:
                                difficulty = "moderate-high"
                            else:
                                difficulty = "low-moderate"
                        
                        elif feat_df.loc[feat_df.Dataset==key,feat_col].values[0] < quantiles.loc[0.75,feat_col]:
                            if regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]) < 0:
                                difficulty = "low-moderate"
                            else:
                                difficulty = "moderate-high"
                        else:
                            if regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]) < 0:
                                difficulty = "low"
                            else:
                                difficulty = "high"
                        prediction_dict[key][feat_col] = [regression_df.select_dtypes(include=[np.number])[feat_col].corr(regression_df.select_dtypes(include=[np.number])[score_col]), regression_model.pvalues[1], 
                                                                            regression_model.rsquared, quantiles.loc[0.25,feat_col], quantiles.loc[0.5,feat_col], quantiles.loc[0.75,feat_col],feat_df.loc[feat_df.Dataset==key,feat_col].values[0], difficulty]
                
                else:
                    if visualize:
                        fig = sm.graphics.plot_fit(regression_model, feat_col)
                        fig.tight_layout(pad=1.0)
                        plt.show()

                        figure(figsize=(15, 8), dpi=80)
                        ax = sns.regplot(x=regression_df[feat_col], y=regression_df[score_col], data=regression_df, ci=None)
                        ax.set_xlabel(feat_col,fontsize=28)
                        ax.set_ylabel(score_col,fontsize=28)
                        for i, x in enumerate(regression_df[[feat_col,score_col]].values):
                            ax.text(x=x[0]+x[0]/100, y=x[1]+0.05, s=regression_df['Dataset'].values[i], fontsize=20)



                        ax.axline((quantiles.loc[0.25,feat_col],quantiles.loc[0.25,score_col]),slope=1000000000,ls='--',color='y')
                        ax.axline((quantiles.loc[0.5,feat_col],quantiles.loc[0.5,score_col]),slope=1000000000,ls='--',color='y')
                        ax.axline((quantiles.loc[0.75,feat_col],quantiles.loc[0.75,score_col]),slope=1000000000,ls='--',color='y')
                        plt.show()

    return prediction_dict, significant_features_list
                    
                    
        
        
        
        
def calculate_features_correlation_regression(result_folder='feature_results', mode='kid'):
    #100-shot datasets
    dataset_folder = "./train_data"
    datasets = os.listdir(dataset_folder)
    data_image_dict = iterate(datasets,dataset_folder).items()
    
    if mode in ['kid','clip','fid']:
        best_scores_df = pd.read_csv('./score_tables/best_{}.csv'.format(mode))
        if mode == 'kid':
            best_scores_df = best_scores_df[['Dataset','KID']]
        elif mode == 'fid':
            best_scores_df = best_scores_df[['Dataset','FID']]
        else:
            best_scores_df = best_scores_df[['Dataset','Clip_FID']]
    else:
        print('No valid mode chosen. Choose from: kid, clip, fid')
        
    display(best_scores_df)
    
    
    #Composition-based Features
    #1) Asymmetry
    print("\n\nSYMMETRIE\n---------------------------------------")
    df_sym, df_sym_simple = calc_symmetrie(data_image_dict, center=True)
    display(df_sym_simple)
    display(df_sym_simple.corr(numeric_only=False))
    df_sym_simple['Mean_Asymmetry'] = 1- df_sym_simple['Mean_Asymmetry']
    df_sym_simple.to_csv('./{}/asymmetry.csv'.format(result_folder))
    _, sig_feat = calculate_regression_values(df_sym_simple, best_scores_df,p_value=1, standardize=True, visualize=True, prediction_dict = None)
    display(df_sym_simple)
    
    df_comp = df_sym_simple[sig_feat]
    
    #2) Entropy
    print("\n\nEntropy\n---------------------------------------")
    df_shannon_rgb, df_shannon_rgb_simple = calc_ent(data_image_dict, method='multiscale', channels='rgb', center=True, plot=False)
    _, sig_feat = calculate_regression_values(df_shannon_rgb_simple, best_scores_df,p_value=1, standardize=True, visualize=True, prediction_dict = None)
    df_shannon_rgb_simple.to_csv('./{}/entropy.csv'.format(result_folder))
    
    df_comp = pd.concat([df_comp, df_shannon_rgb_simple[sig_feat]], axis=1)
    
    #3) Mean Frequency
    print("\n\nFrequency\n---------------------------------------")
    df_frequ, df_freq_simple = calc_ent(data_image_dict, method='mean_frequency', channels='rgb', center=True, plot=False)
    df_freq_simple = df_freq_simple.rename(columns={'Mean_mse':'Mean_freq','Std_mse':'Std_freq'})
    _, sig_feat = calculate_regression_values(df_freq_simple, best_scores_df,p_value=1, standardize=True, visualize=True, prediction_dict = None)
    df_freq_simple.to_csv('./{}/frequency.csv'.format(result_folder))
    
    df_comp = pd.concat([df_comp, df_freq_simple[sig_feat]], axis=1)

    
    #4) Edge detection
    print("\n\nEdge Detection\n---------------------------------------")
    df_edge, df_edge_simple = edge_det(data_image_dict, center=False, plot=False)
    _, sig_feat = calculate_regression_values(df_edge_simple, best_scores_df,p_value=1, standardize=True, visualize=True, prediction_dict = None)
    df_edge_simple.to_csv('./{}/edge.csv'.format(result_folder))
    
    df_comp = pd.concat([df_comp, df_edge_simple[sig_feat]], axis=1)
    
    #5) Spatial information
    print("\n\nSpatial Information\n---------------------------------------")
    spatial_df = spatial(data_image_dict, center=True)
    spatial_df = spatial_df
    _, sig_feat = calculate_regression_values(spatial_df, best_scores_df, p_value=1, standardize=True, visualize=True, prediction_dict = None)
    spatial_df.to_csv('./{}/spatial.csv'.format(result_folder))
    
    df_comp = pd.concat([df_comp, spatial_df[sig_feat]], axis=1)
    
    #6) Fractal dimension
    print("\n\nFractal Dimension\n---------------------------------------")
    fractal_df = fractal(data_image_dict)
    _, sig_feat = calculate_regression_values(fractal_df, best_scores_df, p_value=1, standardize=True, visualize=True, prediction_dict = None)
    fractal_df.to_csv('./{}/fractal.csv'.format(result_folder))
    
    df_comp = pd.concat([df_comp, fractal_df[sig_feat]], axis=1)
    
    #7) delentropy
    print("\n\nEntropy\n---------------------------------------")
    df_shannon_rgb, df_shannon_rgb_simple = calc_ent(data_image_dict, method='delentropy', channels='rgb', center=True, plot=False)
    _, sig_feat = calculate_regression_values(df_shannon_rgb_simple, best_scores_df,p_value=1, standardize=True, visualize=False, prediction_dict = None)
    df_shannon_rgb_simple.to_csv('./{}/delentropy.csv'.format(result_folder))
    
    df_comp = pd.concat([df_comp, df_shannon_rgb_simple[sig_feat]], axis=1)
    
    print(print("\n---------------------------------\Correlation_Composition\n---------------------------------------"))
    display(df_comp.corr(numeric_only=False).round(2))
    
    
    #Texture-based features
    #1) Haralick GLCM-features
    print("\n\nGLCM Features \n---------------------------------------")
    df_glcm = calculate_glcm_features(data_image_dict, color_levels=64, method='COLOR',distance=[1], center=True)
    _, sig_feat = calculate_regression_values(df_glcm, best_scores_df, p_value=1, standardize=True, visualize=True, prediction_dict=None)
    df_glcm.to_csv('./{}/glcm.csv'.format(result_folder))
    
    df_text = df_glcm[sig_feat]
    
    #2) Garbor Features
    #print("\n\nGarbor Features \n---------------------------------------")
    garbor_df = garbor_features(data_image_dict)
    _, sig_feat = calculate_regression_values(garbor_df, best_scores_df, p_value=1, standardize=True, visualize=True, prediction_dict=None)
    garbor_df.to_csv('./{}/garbor.csv'.format(result_folder))
    
    df_text = pd.concat([df_text, garbor_df[sig_feat]], axis=1)
    
    print(print("\n---------------------------------\Correlation_Texture\n---------------------------------------"))
    display(df_text.corr(numeric_only=False).round(2))
    
    
    #Manifold-based features
    #1) Intrinsic Dimension
    print("\n\nIntrinsic Dimension\n---------------------------------------")
    id_df = intrinsic_dimension(data_image_dict, center=True)
    calculate_regression_values(id_df, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    id_df.to_csv('./{}/intrinsic_dimension.csv'.format(result_folder))
    
    
    #Deep-learning-based features
    #1) Segmentation
    segall_df = segment_anything(data_image_dict, mask_generator_2, visualize=False)
    segall_df.to_csv('./{}/segall.csv'.format(result_folder))
    #segall_div.to_csv('./results/segall_div.csv')
    _, sig_feat = calculate_regression_values(segall_df, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    df_deep = segall_df[sig_feat]
    
    #2) Saliency Detection
    saliency_df = saliency_map(data_image_dict, center=True)
    _, sig_feat = calculate_regression_values(saliency_df, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    saliency_df.to_csv('./{}/saliency.csv'.format(result_folder))
    
    df_deep = pd.concat([df_deep, saliency_df[sig_feat]], axis=1)
    
    
    print(print("\n---------------------------------\Correlation_DeepLearning\n---------------------------------------"))
    display(df_deep.corr(numeric_only=False).round(2))
    
    

    
def robustness_check(results_path='feature_results',mode='fid'):
    if mode in ['kid','clip','fid']:
        best_scores_df = pd.read_csv('./score_tables/best_{}.csv'.format(mode))
        if mode == 'kid':
            best_scores_df = best_scores_df[['Dataset','KID']]
        elif mode == 'fid':
            best_scores_df = best_scores_df[['Dataset','FID']]
        else:
            best_scores_df = best_scores_df[['Dataset','Clip_FID']]
    else:
        print('No valid mode chosen. Choose from: kid, clip, fid')
        
    display(best_scores_df)
    #1) Asymmetry
    df_sym = pd.read_csv('./{}/asymmetry.csv'.format(results_path)).rename(columns={'Unnamed: 0':'Dataset'})
    result_dict,_ = calculate_regression_values(df_sym, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #2) Entropy
    df_ent = pd.read_csv('./{}/entropy.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_ent, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #3) Frequency
    df_freq = pd.read_csv('./{}/frequency.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_freq, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #4) Number of edges
    df_edge = pd.read_csv('./{}/edge.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_edge, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #5) Spatial Information
    df_si =  pd.read_csv('./{}/spatial.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_si, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #6) Fractal Dimension
    df_frac = pd.read_csv('./{}/fractal.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_frac, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #1) GLCM Features
    df_glcm = pd.read_csv('./{}/glcm.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_glcm, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #3) Garbor Features
    df_garbor = pd.read_csv('./{}/garbor.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_garbor, best_scores_df, p_value=1, standardize=True, visualize=False, prediction_dict = None)
    
    #1) Intrinsic Dimension
    df_id = pd.read_csv('./{}/intrinsic_dimension.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,_ = calculate_regression_values(df_id, best_scores_df, p_value=1, standardize=True, visualize=True, prediction_dict = None)
    
    #1) segmentation
    df_segall = pd.read_csv('./{}/segall.csv'.format(results_path)).drop('Unnamed: 0', axis=1)
    result_dict,sig_feat = calculate_regression_values(df_segall, best_scores_df, p_value=1, standardize=True, visualize=True, prediction_dict = None)
    
    df_deep = df_segall[sig_feat]
    
    #2) Saliency detection
    df_sal = pd.read_csv('./{}/saliency.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    result_dict,sig_feat = calculate_regression_values(df_sal, best_scores_df, p_value=1, standardize=True, visualize=True, prediction_dict = None)
    
    df_deep = pd.concat([df_deep, df_sal[sig_feat]], axis=1)
    display(df_deep.corr(numeric_only=False).round(2))
    

def prepare_test_datasets(path):
    dataset_list = list()
    result_dict = dict()
    for dataset in os.listdir(path):
        dataset_list.append(dataset)
        result_dict[dataset] = dict()
    print(dataset_list)
    return iterate(dataset_list, path).items(), result_dict


def test_framework(results_path='feature_results',mode='kid'):
    #100-shot datasets
    dataset_folder = "./test_data"
    data_test_dict, result_dict = prepare_test_datasets(dataset_folder)
    
    if mode in ['kid','clip','fid']:
        best_scores_df = pd.read_csv('./score_tables/best_{}.csv'.format(mode))
        if mode == 'kid':
            best_scores_df = best_scores_df[['Dataset','KID']]
        elif mode == 'fid':
            best_scores_df = best_scores_df[['Dataset','FID']]
        else:
            best_scores_df = best_scores_df[['Dataset','Clip_FID']]
    else:
        print('No valid mode chosen. Choose from: kid, clip, fid')
    
    #-------------------------------------------
    #Composition-based features
    #1) Asymmetry
    df_sym = pd.read_csv('./{}/asymmetry.csv'.format(results_path)).rename(columns={'Unnamed: 0':'Dataset'})
    _, df_sym_pred = calc_symmetrie(data_test_dict, center=True, visualize=False)
    
    df_sym_pred = df_sym_pred.reset_index(drop=False).rename(columns={'index':'Dataset'})
    df_sym_pred['Mean_Asymmetry'] = 1-df_sym_pred['Mean_Asymmetry']
    
    df_sym_comb = pd.concat([df_sym, df_sym_pred]).reset_index(drop=True)
    
    result_dict,_ = calculate_regression_values(df_sym_comb, best_scores_df, p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_sym_comb)

    
    #2) Entropy
    df_ent = pd.read_csv('./{}/entropy.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    _, df_shannon_rgb_simple = calc_ent(data_test_dict, method='multiscale', channels='rgb', center=True, plot=False)
    
    df_shannon_rgb_simple = df_shannon_rgb_simple.reset_index(drop=False).rename(columns={'index':'Dataset'})
        
    df_ent_comb = pd.concat([df_ent, df_shannon_rgb_simple]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_ent_comb, best_scores_df, p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_ent_comb)
    
    #3) Frequency
    df_freq = pd.read_csv('./{}/frequency.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    _, df_freq_simple = calc_ent(data_test_dict, method='mean_frequency', channels='rgb', center=True, plot=False)
    
    df_freq_simple = df_freq_simple.reset_index(drop=False).rename(columns={'index':'Dataset'}).rename(columns={'Mean_mse':'Mean_freq','Std_mse':'Std_freq'})
    
    df_freq_comb = pd.concat([df_freq, df_freq_simple]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_freq_comb, best_scores_df, p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_freq_comb)
    
    #4) Number of edges
    print("\n\nEdge Detection\n---------------------------------------")
    df_edge = pd.read_csv('./{}/edge.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    _, df_edge_simple = edge_det(data_test_dict, center=False, plot=False)
    
    df_edge_simple = df_edge_simple.reset_index(drop=False).rename(columns={'index':'Dataset'})
    
    df_edge_combined = pd.concat([df_edge, df_edge_simple]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_edge_combined, best_scores_df,p_value=0.1, standardize=False, visualize=True, prediction_dict = result_dict)
    display(df_edge_combined)
    
    
    #5) Spatial Information
    df_si =  pd.read_csv('./{}/spatial.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    df_si_pred = spatial(data_test_dict, center=True, plot=False)
    
    df_si_pred = df_si_pred.reset_index(drop=False).rename(columns={'index':'Dataset'})
    
    df_si_combined = pd.concat([df_si, df_si_pred]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_si_combined, best_scores_df,p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_si_combined.round(2))
    
    #6) Fractal Dimension
    df_frac = pd.read_csv('./{}/fractal.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    df_frac_pred = fractal(data_test_dict, plot=False)
    
    df_frac_pred = df_frac_pred.reset_index(drop=False).rename(columns={'index':'Dataset'})
    
    df_frac_combined = pd.concat([df_frac, df_frac_pred]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_frac_combined, best_scores_df,p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_frac_combined.round(2))
    
    #Texture-based features
    #1) GLCM Features
    df_glcm = pd.read_csv('./{}/glcm.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    df_glcm_pred = calculate_glcm_features(data_test_dict, color_levels=64, method='COLOR',distance=[1], center=True, plot=False)
    
    df_glcm_pred = df_glcm_pred.reset_index(drop=False).rename(columns={'index':'Dataset'})
    
    df_glcm_combined = pd.concat([df_glcm, df_glcm_pred]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_glcm_combined, best_scores_df,p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_glcm_combined)
    
    #2) LBP Features
    #df_lbp = pd.read_csv('./{}/lbp.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    #df_lbp_pred,_ = apply_lbp2(data_test_dict)
    
    #df_lbp_pred = df_lbp_pred.reset_index(drop=False).rename(columns={'index':'Dataset'}).rename(columns={'AvgEnt':'Mean_LBP_Ent','StdEnt':'Std_LBP_Ent','AvgEnergy':'Mean_LBP_Energy','StdEnergy':'Std_LBP_Energy','AvgUni':'Mean_LBP_Uniformity','StdUni':'Std_LBP_Uniformity'})
    #display(df_lbp)
    #display(df_lbp_pred)
    #df_lbp_combined = pd.concat([df_lbp, df_lbp_pred]).reset_index(drop=True)
    #display(df_lbp_combined)
    #result_dict,_ = calculate_regression_values(df_lbp_combined, best_scores_df,p_value=0.1, standardize=True, visualize=False, prediction_dict = result_dict)
    
    #3) Garbor Features
    df_garbor = pd.read_csv('./{}/garbor.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    df_garbor_pred = garbor_features(data_test_dict, center=True, plot=False)
    
    df_garbor_pred = df_garbor_pred.reset_index(drop=False).rename(columns={'index':'Dataset'})
    
    df_garbor_combined = pd.concat([df_garbor, df_garbor_pred]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_garbor_combined, best_scores_df,p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_garbor_combined.round(2))
    
    
    #Manifold-based features
    #1) Intrinsic Dimension
    df_id = pd.read_csv('./{}/intrinsic_dimension.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    df_id_pred = intrinsic_dimension(data_test_dict, center=True, plot=False)
    
    df_id_pred = df_id_pred.reset_index(drop=False).rename(columns={'index':'Dataset'})
    
    df_id_combined = pd.concat([df_id, df_id_pred]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_id_combined, best_scores_df,p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)
    display(df_id_combined.round(2))
    
    #Deep-learning based features
    #1) segmentation
    df_segall = pd.read_csv('./{}/segall.csv'.format(results_path)).drop('Unnamed: 0', axis=1)
    
    df_segall_pred = segment_anything(data_test_dict, mask_generator_2, visualize=True)
    
    df_seg_comb = pd.concat([df_segall, df_segall_pred]).reset_index(drop=True)
    

    display(df_seg_comb.round(2))
    
    result_dict,_ = calculate_regression_values(df_seg_comb, best_scores_df, p_value=0.1, standardize=False, visualize=False, prediction_dict = result_dict)

    
    
    #2) Saliency detection
    df_sal = pd.read_csv('./{}/saliency.csv'.format(results_path)).drop('Unnamed: 0',axis=1)
    df_sal_pred = saliency_map(data_test_dict, center=True, plot=False)
    
    df_sal_pred = df_sal_pred.reset_index(drop=False).rename(columns={'index':'Dataset'})
    df_sal_combined = pd.concat([df_sal, df_sal_pred]).reset_index(drop=True)
    result_dict,_ = calculate_regression_values(df_sal_combined, best_scores_df,p_value=0.1, standardize=False, visualize=True, prediction_dict = result_dict)
    display(df_sal_combined.round(2))
    
    
   

    
    #Generate the result dataframe for all the test datasets
    result_df= pd.concat({k: pd.DataFrame(v).T for k, v in result_dict.items()}, axis=0)
    result_df.columns = ['beta_value','p_value','r_squared','0.25_quartile','0.5_quartile','0.75_quartile','feature_value','difficulty']
    
    res_df = result_df.reset_index()

    for dataset in result_dict.keys():
        #Generate diversity and complexity frameworks
        result_div = res_df[(res_df.level_1.str.startswith('std')) | (res_df.level_1.str.startswith('Std'))| (res_df.level_1.str.startswith('STD')) | (res_df.level_1 == 'SI_std')]
        result_comp = pd.concat([res_df,result_div]).drop_duplicates(keep=False)

        #Filter for dataset
        result_div = result_div[result_div.level_0 == dataset]
        result_comp = result_comp[result_comp.level_0 == dataset]

        #Display
        print("\nDiversity Framework for ",dataset)
        display(result_div.sort_values('p_value').reset_index(drop=True))
        print("\nComplexity Framework for ", dataset)
        display(result_comp.sort_values('p_value').reset_index(drop=True))
        
    return result_df
    

    
    