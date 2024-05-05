# DCGAN (Deep Convolutional Generative Adversarial Network)

This repository contains an implementation of a DCGAN (Deep Convolutional Generative Adversarial Network) using TensorFlow/Keras. The DCGAN is a type of generative model that is trained using a two-player minimax game framework, where a generator network learns to generate realistic images, and a discriminator network learns to distinguish between real and fake images.

## Overview

- `dcgan.py`: Python script containing the implementation of the DCGAN model.
- `generator.py`: Python script containing the generator network architecture.
- `discriminator.py`: Python script containing the discriminator network architecture.
- `train.py`: Python script for training the DCGAN model.
- `utils.py`: Python script containing utility functions for data loading and preprocessing.
- `README.md`: This file, providing an overview of the repository.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Usage

1. Clone the repository:

```bash
git clone https://github.com/satwik-math/DCGAN.git
cd dcgan
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Train the DCGAN model:

```bash
python train.py
```

4. Generate images using the trained model:

```bash
python generate.py
```

## Results

The trained DCGAN model will generate realistic-looking digit images, as demonstrated in the `generated_images` directory.

## Animation

An animation of the generated images can be found in the `animation` directory.

## References

- Original DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Alec Radford, Luke Metz, Soumith Chintala.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
