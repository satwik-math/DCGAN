# -*- coding: utf-8 -*-

import os
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

(train_images, _), (test_images, _) = mnist.load_data()
images = (train_images - 127.5)/127.5
images = images.reshape(-1, 28, 28, 1)
latent_dim = 128

# CNN Generator
def generator():
    inputs = layers.Input(shape = (latent_dim, ))
    layer = layers.Dense(512*4*4)(inputs)               
    layer = layers.Reshape((4, 4, 512))(layer)          
    layer = layers.Conv2DTranspose(filters = 256, kernel_size = 2, strides = (2, 2), padding = 'valid')(layer)  
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    layer = layers.Conv2DTranspose(filters = 256, kernel_size = 2, strides = (2, 2), padding = 'valid')(layer) 
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    layer = layers.Conv2D(filters = 256, kernel_size = 3, strides = (1, 1), padding = 'valid')(layer)          
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    layer = layers.Conv2DTranspose(filters = 256, kernel_size = 2, strides = (2, 2), padding = 'valid')(layer) 
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    outputs = layers.Conv2D(filters = 1, kernel_size = 3, strides = (1, 1), padding = 'same', activation = 'tanh')(layer)
    return Model(inputs, outputs)

# CNN Discriminator                        
def discriminator():
    inputs = layers.Input(shape = (28, 28, 1))
    layer = layers.Conv2D(filters = 512, kernel_size = 3, strides = (1, 1), padding = 'same')(inputs)                      
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    layer = layers.Conv2D(filters = 256, kernel_size = 2, strides = (2, 2), padding = 'valid')(layer)       
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    layer = layers.Conv2D(filters = 256, kernel_size = 3, strides = (2, 2), padding = 'same')(layer)       
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    layer = layers.Conv2D(filters = 256, kernel_size = 2, strides = (2, 2), padding = 'valid')(layer)      
    layer = layers.LeakyReLU(alpha = 0.2)(layer)
    layer = layers.GlobalAveragePooling2D()(layer)    
    outputs = layers.Dense(1, activation = 'sigmoid')(layer)
    return Model(inputs, outputs)

g_model = generator()
d_model = discriminator()
d_model.compile(optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5), loss = 'binary_crossentropy', metrics = ['accuracy'])
g_model.summary()

# GAN
def GAN(g_model, d_model):
    d_model.trainable = False
    inputs = layers.Input((latent_dim, ))
    generated_images = g_model(inputs)
    outputs = d_model(generated_images)
    return Model(inputs, outputs)

GAN = GAN(g_model, d_model)
GAN.compile(optimizer = Adam(learning_rate = 0.0003, beta_1 = 0.5), loss = 'binary_crossentropy', metrics = ['accuracy'])

# Build a log file
with open('training_log.csv', 'w') as log_file:
    log_file.write('Epoch, Discriminator Loss, Discriminator Accuracy, Generator Loss, Generator Accuracy\n')

# Build model
def build_model(g, d, gan, dataset, noise_dim, epochs, batch_size):
    batch_per_epoch = int(dataset.shape[0]/batch_size)
    half_batch_size = int(batch_size/2)
    for i in range(epochs):
        d_losses = []
        d_accs = []
        g_losses = []
        g_accs = []
        for j in range(batch_per_epoch):
            indeces = np.random.randint(0, dataset.shape[0], half_batch_size)
            X_R = dataset[indeces]
            Y_R = np.ones((half_batch_size, 1))
            random_indeces = np.random.normal(0, 1, noise_dim*half_batch_size)
            random_inputs = random_indeces.reshape(half_batch_size, noise_dim)
            X_F = g.predict(random_inputs)
            Y_F = np.zeros((half_batch_size, 1))
            X, Y = np.vstack((X_R, X_F)), np.vstack((Y_R, Y_F))
            d_loss = d.train_on_batch(X, Y)
            X_gan = np.random.normal(0, 1, noise_dim*batch_size).reshape(batch_size, noise_dim)
            Y_gan = np.ones((batch_size, 1))
            g_loss = GAN.train_on_batch(X_gan, Y_gan)
            d_losses.append(d_loss[0])
            g_losses.append(g_loss[0])
            d_accs.append(d_loss[1])
            g_accs.append(g_loss[1])
            print('Epoch = %d, Step = %d/%d, d_loss = %.3f, gan_loss = %.3f' % (i+1, j+1, batch_per_epoch, d_loss[0], g_loss[0])) 
        d_loss_avg = np.mean(d_losses)
        g_loss_avg = np.mean(g_losses)
        d_accs_avg = np.mean(d_accs)
        g_accs_avg = np.mean(g_accs)
        save(i, g, d, gan, dataset, latent_dim)
        with open('training_log.csv', 'a') as log_file:
             log_file.write(f'{i+1}, {d_loss_avg}, {d_accs_avg*100}, {g_loss_avg}, {g_accs_avg*100}\n')

# Save plots and model
def save(i, g, d, gan, dataset, latent_dim, n_samples = 200):
    X_R, Y_R = dataset[np.random.randint(0, dataset.shape[0], n_samples)], np.ones((n_samples, 1))
    d_r = d.evaluate(X_R, Y_R, verbose=0)
    X_F, Y_F = g.predict(np.random.normal(0, 1, latent_dim*n_samples).reshape(n_samples, latent_dim)), np.zeros((n_samples, 1))
    d_f = d.evaluate(X_F, Y_F, verbose=0)
    print('-- Accuracy real:%.0f%%, fake: %.0f%%' % (d_r[1]*100, d_f[1]*100))
    os.makedirs(model_directory_R, exist_ok = True)
    os.makedirs(model_directory_F, exist_ok = True)
    save_plot_with_probs(X_R, d, os.path.join(model_directory_R,'real_images_%03d.png' % (i + 1)))
    save_plot_with_probs(X_F, d, os.path.join(model_directory_F, 'fake_images_%03d.png' % (i + 1)))
    save_plot(X_F, i)
    os.makedirs(model_dir, exist_ok = True)
    os.makedirs(model_dir1, exist_ok = True)
    os.makedirs(model_dir2, exist_ok = True)
    os.makedirs(model_dir3, exist_ok = True)
    filename1 = os.path.join(model_dir1, 'generator_weights_%03d.keras' % (i + 1))
    filename2 = os.path.join(model_dir2, 'discriminator_weights_%03d.keras' % (i + 1))
    filename3 = os.path.join(model_dir3, 'gan_weights_%03d.keras' % (i + 1))
    g.save_weights(filename1)
    d.save_weights(filename2)
    gan.save_weights(filename3)

os.makedirs(model_dir5, exist_ok = True)

def save_plot_with_probs(examples, model, filename, n=7):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i], cmap = 'gray')
        probability = model.predict(np.expand_dims(examples[i], axis=0))[0][0]
        plt.text(2, 2, f"{probability:.4f}", color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.7))
    plt.savefig(filename)
    plt.close()

def save_plot(examples, epoch, n=7):
    examples = (examples + 1) / 2
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i], cmap = 'gray')
        filename = os.path.join(model_dir5, 'generated_plot%3d.png' % (epoch + 1))
        plt.savefig(filename)
        plt.close

# Generated images
noise = (np.random.normal(0, 1, (49, latent_dim)) + 1)/2.0
list = g_model.predict(noise)
for i in range(49):
    plt.subplot(7, 7, 1+i)
    plt.axis('off')
    plt.imshow(list[i, :, :, 0], cmap = 'gray')
plt.show()
