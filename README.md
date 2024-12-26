# MNIST-Inspired Neural Network OCR

A very simple neural network implementation inspired by the well-known MNIST dataset for the Optical Character Recognition (OCR) problem.
This solution implements forward pass, backpropagation and weights update from scratch.

## Project Structure

The project consists of three main components:

1. **Image Creation and Storage**: Draw and save images for future training.
2. **Model Training**: Train the neural network using the saved images.
3. **Prediction**: Use the trained model to predict drawn numbers.

## Features

### 1. Image Creation

- Use arrow keys to select the number you want to draw.
- Press Enter to save the image.
- Use Backspace to clear the canvas.

### 2. Model Training

- Utilizes a configuration file to specify neural network architecture.
- Creates a network with one input layer, one hidden layer, and one output layer.
- Number of neurons in each layer is customizable via the config file.
- Save weights after each 100 iterations

### 3. Prediction

- Displays the same drawing interface as the image creation component.
- Use Backspace to clear the canvas.
- Press Enter to have the neural network predict the drawn number (showed in console)


Enjoy programming and have a nice day!
