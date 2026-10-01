# 📸 Image Denoising using Convolutional Autoencoders (CAE)

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

An end-to-end Deep Learning application that restores corrupted, noisy images back to clean ground-truth quality using a **Symmetric Convolutional Autoencoder** trained on the **Fashion-MNIST** benchmark dataset.

🎯 What This Project Does
When digital images get corrupted by severe additive Gaussian noise ($\sigma = 0.3$), standard filters blur essential edges. This Deep Learning pipeline
Compresses noisy input images into a low-dimensional latent bottleneck feature space (Encoder).
Filters out random noise while retaining core structural patterns.
Reconstructs high-fidelity, clean $28 \times 28$ images pixel-by-pixel (Decoder).


🏗 Model Architecture BreakdownEncoder: 
Conv2D(32) $\rightarrow$ MaxPooling2D $\rightarrow$ Conv2D(16) $\rightarrow$ MaxPooling2D
Latent Bottleneck: Compressed structural feature representation.
Decoder: Conv2D(16) $\rightarrow$ UpSampling2D $\rightarrow$ Conv2D(32) $\rightarrow$ UpSampling2D $\rightarrow$ Conv2D(1, Sigmoid)


## 📊 Quantitative Benchmark Results
| Metric | Noisy Image | Denoised Reconstruction | Performance Gain |
| :--- | :---: | :---: | :---: |
| **Peak Signal-to-Noise Ratio (PSNR)** | 12.81 dB | **20.18 dB** | **+7.37 dB** |
| **Structural Similarity (SSIM)** | 0.4121 | **0.6888** | **+0.2767** |

## 🖼️ Dataset Results Evaluation
![Batch Denoising Results](fashion_denoising_results.png)


## 🚀 How to Run
1. Clone this repository.
2. Open `Image_Denoising_CAE.ipynb` in Google Colab or Jupyter Notebook.
3. Run all cells to train or perform evaluation using `autoencoder_fashion.h5`.
