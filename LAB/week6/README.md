# Pix2Pix GAN – Image-to-Image Translation (Lab 6)

## Overview

This project implements **Pix2Pix**, a conditional Generative Adversarial Network (cGAN) used for **image-to-image translation**. The model learns a mapping from an input image to a target image using paired training data.

In this implementation, edge images are used as input and the model learns to generate the corresponding real images.

The architecture consists of:

* **U-Net Generator**
* **PatchGAN Discriminator**

The generator produces images while the discriminator evaluates whether the generated images look realistic.

---

## Objective

The goal of this lab is to:

* Understand **paired image-to-image translation**
* Implement **Pix2Pix architecture**
* Train the model using **adversarial loss and L1 reconstruction loss**
* Generate realistic images from edge inputs
* Visualize the generated results

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Torchvision

---

## Architecture

### Generator – U-Net

The generator uses a **U-Net architecture**, which is an encoder–decoder CNN with skip connections.

The encoder extracts features by downsampling the input image, while the decoder reconstructs the output image. Skip connections help preserve spatial information and improve output quality.

---

### Discriminator – PatchGAN

The discriminator uses **PatchGAN**, which evaluates small patches of the image instead of the entire image.

This helps the model learn finer textures and improves realism in generated images.

---

## Dataset

Instead of using a large paired dataset, this implementation generates edge images from the **CIFAR-10 dataset** using the Canny edge detection algorithm.

Steps used:

1. Load CIFAR-10 images
2. Convert images to edge maps using OpenCV
3. Create paired data (Edge Image → Original Image)

---

## Loss Functions

Two losses are used during training:

**Adversarial Loss**

* Helps the generator produce realistic images
* Uses Binary Cross Entropy

**L1 Reconstruction Loss**

* Ensures generated images are close to the target images
* Reduces blur and improves structure

Final Generator Loss:

```
Generator Loss = Adversarial Loss + L1 Loss
```

---

## Training Details

| Parameter     | Value   |
| ------------- | ------- |
| Optimizer     | Adam    |
| Learning Rate | 0.0002  |
| Batch Size    | 16      |
| Epochs        | 3       |
| Framework     | PyTorch |

---

## Project Structure

```
Pix2Pix-Lab6/
│
├── lab6.py
├── result.png
└── README.md
```

---

## How to Run

1. Clone the repository

```
git clone https://github.com/yourusername/pix2pix-lab6.git
```

2. Install dependencies

```
pip install torch torchvision matplotlib opencv-python numpy
```

3. Run the training script

```
python lab6.py
```

---

## Output

The model generates images from edge inputs.

Example output:

* Input Edge Image
* Real Target Image
* Generated Image

These images are saved as:

```
result.png
```

---

## Conclusion

This project demonstrates the use of **Pix2Pix GAN for image-to-image translation** using a U-Net generator and PatchGAN discriminator. The model learns to generate realistic images from edge inputs using adversarial training combined with reconstruction loss.

Pix2Pix performs significantly better than a simple CNN encoder–decoder because adversarial learning helps produce sharper and more detailed images.

---
