import torch
from torch import nn
from torch import optim
from torchvision import models

def build_network(architecture,hidden_units):
    # Load a pre-trained VGG16 model.
    model = models.vgg16(pretrained = True)
 
    # Freeze all the parameters in the pre-trained model.
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier that fits the specific task.
    classifier = nn.Sequential(
              nn.Linear(25088, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
    
    # Replace the original classifier in the VGG16 model with the newly defined classifier.
    model.classifier = classifier

    return model

def train_step(model, trainloader, optimizer, criterion, device):
     # Set model to training mode
    model.train() 
    running_loss = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad() 

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()

    return running_loss / len(trainloader)


def validation_step(model, validloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    accuracy = 0

    with torch.no_grad():  # Turning off gradient calculation
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(dim=1)
            correct = predicted.eq(labels).sum().item()
            accuracy += correct

    # Calculate the average validation loss and accuracy over all batches.
    val_loss /= len(validloader)
    accuracy /= len(validloader.dataset)

    return val_loss, accuracy


def train_network(model, epochs, learning_rate, trainloader, validloader,gpu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the optimizer with the model's classifier parameters and the specified learning rate.
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # Initialize the loss function.
    criterion = nn.NLLLoss()
    model.to(device)
    
    train_loss = 0

     # Loop over the dataset multiple times, according to the number of epochs.
    for epoch in range(epochs):
         # Perform a training step and calculate the training loss.
        train_loss = train_step(model, trainloader, optimizer, criterion, device)
        # Perform a validation step and calculate the validation loss and accuracy.
        valid_loss, valid_accuracy = validation_step(model, validloader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}, "
            f"Train loss: {train_loss:.3f}, "
            f"Valid loss: {valid_loss:.3f}, "
            f"Valid accuracy: {valid_accuracy:.3f}")
    
    return model, criterion