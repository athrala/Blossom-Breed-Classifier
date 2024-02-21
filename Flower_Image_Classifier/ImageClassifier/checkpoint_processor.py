import torch
from torch import nn
from torch import optim
from torchvision import models
import model_processor


def save_checkpoint(model,architecture,hidden_units, epochs, learning_rate, save_dir):
    # Initialize the optimizer used for the model's classifier.
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # Initialize the loss function used during training.
    criterion = nn.NLLLoss()

     # Create the checkpoint dictionary containing all elements to save.
    checkpoint = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'class_to_idx': model.class_to_idx,
        'architecture': architecture,
        'hidden_units': hidden_units,
    }
    
    # Define the path to save the checkpoint.
    checkpoint_path = save_dir + "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    
def load_checkpoint(filepath):
    # Load a pre-trained VGG16 model
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False

     # Define a new, custom classifier
    classifier = nn.Sequential(
          nn.Linear(25088, 256),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(128, 102),
          nn.LogSoftmax(dim = 1)
        )
    # Replace the pre-trained model's classifier with the new custom classifier
    model.classifier = classifier

     # Load the checkpoint from the specified file path
    checkpoint = torch.load(filepath)
    model = model_processor.build_network(checkpoint['architecture'], checkpoint['hidden_units'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def predict_image(processed_image, model, topk): 
     # Set the model to evaluation mode to disable dropout or batch normalization layers
    model.eval()

    # Disable gradient calculations for efficiency and to prevent changes to the model
    with torch.no_grad():
        # Forward pass through the model to get the log probabilities
        logps = model.forward(processed_image.unsqueeze(0))
        ps = torch.exp(logps)

         # Get the topk predictions
        probs, labels = ps.topk(topk, dim=1)

        # Reverse the model's class to index mapping to get a mapping from index to class
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}

        # Translate the indices to classes
        classes = [idx_to_class[label.item()] for label in labels[0]]

        # Return the probabilities and classes of the top predictions
        return probs[0].tolist(), classes