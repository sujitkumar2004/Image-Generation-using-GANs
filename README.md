# 🧠 Image Generation using GANs  

This project demonstrates how to **build and train a Generative Adversarial Network (GAN)** from scratch using **TensorFlow** and **Python** to generate high-quality synthetic images of fashion items.  
The GAN architecture includes:  
- A **generator model** that creates realistic fashion images.  
- A **discriminator model** that evaluates the authenticity of generated images.

---

## 📝 Overview  

This project walks through the key steps involved in developing a complete GAN pipeline:  

- 🧩 **Loading and preprocessing** the Fashion MNIST dataset.  
- 🧱 **Building and training** the generator and discriminator models.  
- 🔁 **Implementing a custom training loop** with noise injection, loss calculation, backpropagation, and optimization.  
- 🖼️ **Saving generated fashion images** during training for progress visualization.  

---

## 🚀 Key Features  

- **TensorFlow-based GAN implementation** for fashion image generation.  
- **Efficient data loading and preprocessing** with the TensorFlow Datasets (TFDS) API.  
- **Custom-built Convolutional Neural Networks (CNNs)** for both generator and discriminator.  
- **Upsampling and convolutional layers** to enhance image quality.  
- **Automatic saving of generated images** after each epoch to track training improvements.  

---

## 🧩 Dataset  

This project uses the **Fashion MNIST** dataset, a collection of grayscale fashion images (e.g., shoes, shirts, bags).  
The dataset is automatically downloaded via TensorFlow Datasets — **no manual setup required**.  

```python
import tensorflow_datasets as tfds
(train_ds, test_ds), ds_info = tfds.load(
    'fashion_mnist', 
    split=['train', 'test'], 
    as_supervised=True, 
    with_info=True
)
