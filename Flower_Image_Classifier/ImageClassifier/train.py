import argparse
from data_processor import load_data
import model_processor
import os
import torch

# Setup argparse to handle command line arguments
parser = argparse.ArgumentParser(description='Neural network data training')
parser.add_argument('data_directory', help='Path to dataset on which the neural network should be trained.')
parser.add_argument('--save_dir', default='', help='Path to directory where the checkpoint should be saved. Default is the current directory.')
parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'resnet18'], help='Network architecture. Supported: vgg16, resnet18. Default is vgg16.')
parser.add_argument('--learning_rate', type=float, default=0.0025, help='Learning rate. Default is 0.0025.')
parser.add_argument('--hidden_units', type=int, default=256, help='Number of hidden units. Default is 512.')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs. Default is 5.')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available.')

# Parse the arguments
args = parser.parse_args()

# Ensuring save_dir is an absolute path
save_dir = os.path.abspath(args.save_dir)

# GPU availability check
gpu = args.gpu and torch.cuda.is_available()

# Load the data
train_data, trainloader, validloader, testloader = load_data(args.data_directory)

# Build and train the network
model = model_processor.build_network(args.arch, args.hidden_units)
model.class_to_idx = train_data.class_to_idx
model, criterion = model_processor.train_network(model, args.epochs, args.learning_rate, trainloader, validloader, gpu)

# Save the trained model
model_processor.save_checkpoint(model, args.arch, args.hidden_units, args.epochs, args.learning_rate, save_dir)