# MNIST-Inspired Neural Network OCR

A simple neural network implementation inspired by the well-known MNIST dataset for Optical Character Recognition (OCR). This solution is built entirely using native Rust functions, with all matrix operations performed manually.

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
- Save weights after each 10 iterations

### 3. Prediction

- Displays the same drawing interface as the image creation component.
- Use Backspace to clear the canvas.
- Press Enter to have the neural network predict the drawn number.


## Support the Author

If you find this project helpful, you can donate:

- USDT (TRC20): `TXiRw82og6KPWbntgRQtj4N77xrqEu11fh`
- BTC: `bc1qjc4rkus9thvpcdafyzu9jecxnwd6fr7m3ew5lg`
- ETH: `0x4455C2F365d801a515e2FF0175D89C97e55105D6`

Enjoy programming and have a nice day!
