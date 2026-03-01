# Week 5 – Encoder–Decoder CNN for Image-to-Image Translation

## Course
CSET419 – Introduction to Generative AI

## Lab Title
Baseline CNN for Image-to-Image Translation (Encoder–Decoder without GAN)

## Objective
The objective of this lab is to implement a basic encoder–decoder convolutional neural network (CNN) for paired image-to-image translation and analyze its performance using reconstruction loss.

## Dataset
CIFAR-10 Dataset

## Methodology
- Loaded paired images from the CIFAR-10 dataset
- Normalized images to the range [-1, 1]
- Designed a baseline encoder–decoder CNN architecture
- Trained the model using Mean Squared Error (MSE) loss
- Evaluated reconstruction quality by visual comparison of original and reconstructed images

## Model Description
- Encoder: Two convolutional layers with ReLU activation
- Decoder: Two transposed convolution layers with ReLU and Tanh activation
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

## Results
The reconstructed images preserve the overall structure of the original images but appear blurry.  
This behavior is expected due to pixel-wise reconstruction loss and the absence of adversarial or perceptual loss.

## Conclusion
This lab demonstrates the limitations of baseline encoder–decoder CNNs for image-to-image translation.  
While the model learns coarse visual features, it fails to recover fine details, resulting in blurred outputs.

## Files
- `genai_lab5/lab5.py` – Implementation of encoder–decoder CNN
- `genai_lab5/result.png` – Visualization of original and reconstructed images
