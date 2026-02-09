# opencode_task2

Task 2: Neural Network from Scratch (CIFAR-10)

This project implements a neural network from scratch to classify images from the CIFAR-10 dataset, with the goal of understanding training dynamics and generalization without using pretrained models.

# Objective

Design and train a neural network from scratch

Understand how models learn features without transfer learning

Compare performance and convergence with pretrained models

Deploy the trained model using Streamlit

# Dataset

CIFAR-10 Dataset

60,000 color images (32×32)

10 classes:

Airplane, Automobile, Bird, Cat, Deer

Dog, Frog, Horse, Ship, Truck

# Model Architecture

A simple fully connected neural network (MLP) was implemented using PyTorch.

Architecture Details:

Input Layer: 32 × 32 × 3 (flattened)

Hidden Layer: 512 neurons with ReLU activation

Output Layer: 10 neurons (one per class)

Loss Function: CrossEntropyLoss

Optimizer: Adam

This architecture intentionally avoids convolutional layers to study the limitations of shallow networks on image data.

# Training Summary

The model was trained from scratch on CIFAR-10

No pretrained weights were used

Training accuracy improved steadily but saturated early

Generalization performance was limited compared to transfer learning

# Observations & Analysis

Training convergence was slower than pretrained models

The model struggled with visually similar classes (e.g., cat vs dog)

Lack of convolutional layers limited spatial feature learning

Performance highlights the importance of feature extractors in deep learning

A detailed analysis is provided in analysis.txt.

# Deployment

The trained model was deployed using Streamlit for real-time inference.

Deployment Features:

Upload an image (PNG/JPG)

Model predicts CIFAR-10 class

CPU-safe loading for cloud compatibility

# Repository Structure
transfer_learning_task2/

├── app.py                  # Streamlit app
├── model.ipynb             # Training notebook
├── simplenn_cifar10.pth    # Trained model weights
├── requirements.txt        # Dependencies
├── analysis.txt            # Training & generalization analysis
└── README.md               # Project documentation

# Installation & Run Locally
pip install -r requirements.txt
streamlit run app.py

# Key Takeaways

Neural networks trained from scratch require more data and computation

Feature extraction is crucial for image classification

Transfer learning significantly improves efficiency and performance

Deployment completes the full ML lifecycle

# Conclusion

This task demonstrates the limitations of shallow neural networks and provides practical insight into why deep, pretrained models dominate modern computer vision tasks.

# Author
Arshi Mittal

Arshi Mittal Induction Task – Transfer Learning & Deep Learning
