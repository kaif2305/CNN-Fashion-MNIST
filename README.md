# Enhanced CIFAR-10 Classification with PyTorch CNN (Data Augmentation & Batch Normalization)

This project provides a PyTorch implementation for image classification on the CIFAR-10 dataset, incorporating best practices such as data augmentation, Batch Normalization, and Dropout to build a more robust and accurate Convolutional Neural Network (CNN).

## Project Overview

The Python script outlines a comprehensive process for training a sophisticated CNN on CIFAR-10:

1.  **Data Preparation with Augmentation**: Loads the CIFAR-10 dataset and applies advanced transformations, including data augmentation, to the training set.
2.  **Enhanced Model Definition**: Defines a custom CNN architecture (`EnhancedCNN`) with multiple convolutional layers, Batch Normalization, Max Pooling, and Dropout. It also dynamically calculates the input size for the fully connected layers.
3.  **Loss Function and Optimizer**: Configures the `CrossEntropyLoss` and the `Adam` optimizer.
4.  **Training Loop**: Implements a standard training loop over several epochs, tracking the training loss.
5.  **Evaluation Loop**: Assesses the trained model's performance by calculating accuracy on the unseen test dataset.
6.  **Visualization**: Plots the training loss over epochs to monitor the learning process.

## Dataset

The **CIFAR-10 dataset** is a widely recognized benchmark in computer vision. It comprises 60,000 32x32 color images distributed across 10 distinct classes. The dataset is split into 50,000 training images and 10,000 test images. `torchvision.datasets` handles the automatic download and management of this dataset.

### Data Preprocessing and Augmentation

To enhance model generalization and prevent overfitting, different transformations are applied to the training and test sets:

**`transform_train` (for Training Data):**
* `transforms.RandomHorizontalFlip()`: Randomly flips the image horizontally with a default probability of 0.5.
* `transforms.RandomCrop(32, padding=4)`: Pads the image by 4 pixels on each side and then randomly crops it back to 32x32. This simulates variations in object positioning within the image.
* `transforms.ToTensor()`: Converts the image to a PyTorch `FloatTensor` and scales pixel values to `[0.0, 1.0]`.
* `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`: Normalizes the RGB channels to a range of `[-1.0, 1.0]`, which can improve training stability.

**`transform_test` (for Test Data):**
* `transforms.ToTensor()`
* `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
    *(Only normalization and conversion to tensor are applied to the test set to ensure consistent evaluation without introducing artificial variations).*

`DataLoader` instances are created for efficient batching and shuffling of the training data.

## Enhanced CNN Model Architecture

The `EnhancedCNN` class defines a more sophisticated CNN structure compared to basic examples:

```python
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # Convolutional Layer 1: Input 3 channels (RGB), Output 6 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6) # Batch Normalization after conv1
        
        # Convolutional Layer 2: Input 6 channels, Output 16 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16) # Batch Normalization after conv2
        
        # Max Pooling Layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout Layer: Applied to the first fully connected layer output
        self.dropout = nn.Dropout(0.5) 
        
        # Dynamically calculate the flattened size for the first fully connected layer
        self._calculate_conv_output() 
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(self.conv_output_size, 120)
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(120, 84)
        # Output Layer: 10 neurons for 10 CIFAR-10 classes
        self.fc3 = nn.Linear(84, 10)
    
    def _calculate_conv_output(self):
        # A dummy forward pass to determine the flattened size after convolutional and pooling layers
        dummy_input = torch.zeros(1, 3, 32, 32) # Batch_size 1, 3 channels, 32x32 image
        with torch.no_grad(): # No need to compute gradients for this
            # The exact sequence of operations reflects the forward pass to get the shape
            output = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(dummy_input)))))))
        self.conv_output_size = output.numel() # Total number of elements in the output tensor
                                               # This will be 16 * 5 * 5 = 400 for 32x32 input
                                               # after two conv-relu-bn-pool blocks.

    def forward(self, x):
        # Conv1 -> BatchNorm -> ReLU -> Pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Flatten the feature maps: x.size(0) preserves the batch dimension
        x = x.view(x.size(0), -1) 
        # FC1 -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Applied after the first fully connected layer
        # FC2 -> ReLU
        x = F.relu(self.fc2(x))
        # FC3 (Output Layer)
        x = self.fc3(x)
        return x