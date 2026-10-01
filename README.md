# 📸 Image Denoising using Convolutional Autoencoders (CAE)

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

An end-to-end Deep Learning application that restores corrupted, noisy images back to clean ground-truth quality using a **Symmetric Convolutional Autoencoder** trained on the **Fashion-MNIST** benchmark dataset.

## 📌 Project Overview
This project implements a Deep Convolutional Autoencoder (CAE) trained to recover clean, high-fidelity images from severe synthetic Gaussian noise (\(\sigma = 0.3\)).

## 🏗️ Model Architecture
- **Input Layer:** \((28, 28, 1)\)
- **Encoder:** 
  - `Conv2D(32, (3,3))` + ReLU -> `MaxPooling2D((2,2))`
  - `Conv2D(16, (3,3))` + ReLU -> `MaxPooling2D((2,2))`
- **Bottleneck:** Compressed structural feature representations.
- **Decoder:**
  - `Conv2D(16, (3,3))` + ReLU -> `UpSampling2D((2,2))`
  - `Conv2D(32, (3,3))` + ReLU -> `UpSampling2D((2,2))`
  - `Conv2D(1, (3,3))` + Sigmoid output.

## 📊 Quantitative Benchmark Results
| Metric | Noisy Image | Denoised Reconstruction | Performance Gain |
| :--- | :---: | :---: | :---: |
| **Peak Signal-to-Noise Ratio (PSNR)** | 12.81 dB | **20.18 dB** | **+7.37 dB** |
| **Structural Similarity (SSIM)** | 0.4121 | **0.6888** | **+0.2767** |

## 🖼️ Dataset Results Evaluation
![Batch Denoising Results](fashion_denoising_results.png)


## 🛠️ Requirements & Tech Stack
Frameworks: TensorFlow 2.x, Keras

Libraries: NumPy, Matplotlib

Environment: Google Colab GPU (T4)
