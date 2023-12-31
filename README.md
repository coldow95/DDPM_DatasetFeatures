# DDPM_DatasetFeatures

This is the official repository of the thesis: ASSESSING SPARSE IMAGE DATASETS FOR GENERATIVE MODELING: A FEATURE-DRIVEN APPROACH.

To start, clone the repository and download the remaining data (datasets and model weights) from one of the following link:

- https://drive.google.com/file/d/1qHagoStazQKI0CnL_6bMyCVYTRhco7a1/view?usp=drive_link

Unzip the data into the working directory, and install the requirements based on the requirements.txt file.

To apply the dataset assessment framework on unseen data, put your datasets into the "test_data" folder. It is important to keep following folder structure: ./test_data/img/
Obama and LSUNBed are already placed in the test folder as examples.

```python
  !python Main.py --apply_framework True
```

To train the model or evaluate across all datasets in the "model_weights" folder, please execute the corresponding jupyter notebooks (model_evaluation for testing and model_and_train_loop for training)





## Implementation references

"Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis":<br />
- https://github.com/odegeasslbc/FastGAN-pytorch  <br />
- Liu, B., Zhu, Y., Song, K., & Elgammal, A. (2021). Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis. <br /> <br />

Efficient Attention: Attention with Linear Complexities (Shen et al., 2018) ... was not used in the final implementation. <br />
- https://github.com/cmsflash/efficient-attention  <br /> <br />

Elucidating the Design Space of Diffusion-Based Generative Models (EDM) 
- https://github.com/NVlabs/edm
- Karras, T., Miika Aittala, Timo Aila, & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. Advances in Neural Information Processing Systems 35 (NeurIPS 2022) . https://doi.org/10.48550/arxiv.2206.00364 <br /> <br /> 

Base implementation of DDPM  <br />
- https://github.com/lucidrains/denoising-diffusion-pytorch  <br /> <br />

Denoising Diffusion Implicit Models
- https://github.com/ermongroup/ddim <br />
- Song, J., Meng, C., & Stefano Ermon. (2020). Denoising Diffusion Implicit Models. ArxiV. https://doi.org/10.48550/arxiv.2010.02502 <br /><br />

Efficient Diffusion Training via Min-SNR Weighting Strategy
- https://github.com/TiankaiHang/Min-SNR-Diffusion-Training <br />
- Hang, T., Gu, S., Li, C., Bao, J., Chen, D., Hu, H., … Guo, B. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. https://doi.org/10.48550/arxiv.2303.09556. ARriV.  <br /> <br />

Segment Anything
- https://github.com/facebookresearch/segment-anything     <br />
- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., … Girshick, R. (2023). Segment Anything. ArXiv (Cornell University). https://doi.org/10.48550/arxiv.2304.02643 <br /> <br />

Maximum Likelihood Estimation of Intrinsic Dimension
- https://gist.github.com/mehdidc/8a0bb21a31c43b0cbbdd31d75929b5e4/  <br />
- Levina, E., & Bickel, P. J. (2004). Maximum Likelihood Estimation of Intrinsic Dimension. Advances in Neural Information Processing Systems 17 (NIPS 2004), 17, 777–784.
