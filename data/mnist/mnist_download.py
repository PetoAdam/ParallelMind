import numpy as np
import torch
from torchvision import datasets, transforms

# Define characters for ASCII art based on pixel intensity
ASCII_CHARS = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]

def pixel_to_ascii(pixel_value):
    # Map pixel value (0-255) to an ASCII character
    index = int(float(pixel_value) / 255 * (len(ASCII_CHARS) - 1))
    return ASCII_CHARS[index]

def print_image(image):
    # Convert the normalized image (values [0, 1]) to 0-255 for display
    converted_image = (image * 255).astype(np.uint8)
    for row in converted_image:
        print("".join(pixel_to_ascii(pixel) for pixel in row))

# Save dataset and print images
def save_dataset(dataset, images_file, labels_file):
    images = []
    labels = []
    for i, (image, label) in enumerate(dataset):
        np_image = np.array(image).squeeze()  # Convert PIL image to NumPy array (normalized [0, 1])
        images.append(np_image)
        labels.append(label)

        # Print the first 5 images and labels as a preview
        if i < 5:
            print(f"Label: {label}")
            print_image(np_image)
            print("-" * 40)

    np.save(images_file, np.array(images, dtype=np.float32))  # Save images as float32
    np.save(labels_file, np.array(labels, dtype=np.uint8))    # Save labels as uint8

# Download and prepare the MNIST dataset
data_dir = "./data"
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

# Save the datasets
print("Saving datasets...")
save_dataset(train_dataset, "mnist_train_images.npy", "mnist_train_labels.npy")
save_dataset(test_dataset, "mnist_test_images.npy", "mnist_test_labels.npy")
