# Week 3 – Variational Autoencoder (VAE)

## Course
**CSET-419 – Introduction to Generative AI**

---

## Objective
To implement a **Variational Autoencoder (VAE)** using TensorFlow/Keras in order to:
- Learn compact latent representations of images
- Visualize the latent space
- Generate new handwritten digit images from the learned distribution

---

## Dataset
**MNIST Handwritten Digits Dataset**
- Image size: 28 × 28 (grayscale)
- Total classes: 10 digits (0–9)

---

## Description
This lab demonstrates the implementation of a **Variational Autoencoder (VAE)**.

The model consists of:
- **Encoder**: Learns the mean and variance of the latent distribution
- **Reparameterization Trick**: Enables backpropagation through stochastic sampling
- **Decoder**: Reconstructs images from latent vectors

The VAE is trained using a combined loss function:
- **Reconstruction Loss** (Binary Cross-Entropy)
- **KL Divergence Loss** to regularize the latent space

---

## Files
- `lab3.py` – Python implementation of the VAE model
- `vertopal.com_Lab3.pdf` – Output PDF containing:
  - Sample MNIST images
  - Training loss graph
  - Latent space visualization
  - Generated digit manifold
- `README.md` – Lab documentation

---

## Output
The generated PDF includes:
1. **Sample MNIST Images**
2. **Training Loss Curve**
3. **2D Latent Space Scatter Plot**
4. **VAE Generated Digit Manifold**

These results demonstrate that the VAE successfully learns meaningful latent representations and can generate realistic handwritten digits.

---

## Conclusion
The Variational Autoencoder effectively captures the underlying distribution of MNIST digits.  
Latent space visualization shows smooth transitions between digit classes, and the generated samples confirm the generative capability of the model.

This experiment validates the usefulness of VAEs for representation learning and image generation tasks.

---

## Tools & Libraries Used
- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- SciPy
