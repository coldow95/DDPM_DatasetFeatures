# DDPM_DatasetFeatures

This is the official repository of the thesis: ASSESSING SPARSE IMAGE DATASETS FOR GENERATIVE MODELING: A FEATURE-DRIVEN APPROACH.

To start, clone the repository and download the remaining data (datasets and model weights) from one of the following link:

- https://drive.google.com/file/d/1qHagoStazQKI0CnL_6bMyCVYTRhco7a1/view?usp=drive_link

Unzip the data into the working directory, and install the requirements based on the requirements.txt file.





Much credit to the following references (Citations are included in the thesis, if the code is based on a paper):

"Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis" (Liu et al., 2021):
https://github.com/odegeasslbc/FastGAN-pytorch 

Efficient Attention: Attention with Linear Complexities (Shen et al., 2018) ... was not used in the final implementation.
https://github.com/cmsflash/efficient-attention  

Base implementation of DDPM
https://github.com/lucidrains/denoising-diffusion-pytorch 
https://github.com/lucidrains/denoising-diffusion-pytorch   

Segment Anything
https://github.com/facebookresearch/segment-anything    


@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
