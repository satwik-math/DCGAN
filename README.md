# DCGAN (Deep Convolutional Generative Adversarial Network)

This repository contains an implementation of a DCGAN (Deep Convolutional Generative Adversarial Network) using TensorFlow/Keras. The DCGAN is a type of generative model that is trained using a two-player minimax game framework, where a generator network learns to generate realistic images, and a discriminator network learns to distinguish between real and fake images.

## Overview

- `gan.py`: Python script containing the implementation of the DCGAN model.
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
```

2. Train the DCGAN model:

```bash
python gan.py
```

## Results

The trained DCGAN model will generate realistic-looking digit images, as demonstrated in the `generated_images` directory.
![Generated Image Animation](https://github.com/satwik-math/DCGAN/blob/main/assets/generated_animation.gif)


## References

- Original DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Alec Radford, Luke Metz, Soumith Chintala.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
