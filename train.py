# Import required libraries for deep learning, data processing,
# visualization, experiment tracking, and command-line arguments.
import torch
import torchvision
from torchvision.transforms import v2
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

# Define command-line arguments to allow users to configure
# training settings without modifying the source code.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tracker",
    type=str,
    default="wandb",
    choices=["wandb", "mlflow", "none"]
)

parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--momentum", type=float, default=0.9)
args = parser.parse_args()

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
momentum = args.momentum

# Set random seeds and deterministic settings to ensure
# reproducible training results across different runs.
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to {SEED}")

# Define image preprocessing and normalization steps
# for the CIFAR-10 dataset.
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#batch_size = 4
# Download and load training and test datasets.
# Create DataLoader objects for batch processing during training
# and evaluation.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Sample image visualization code from the PyTorch tutorial
# was removed because it is not required for automated training,
# evaluation, checkpointing, or experiment tracking. it kept in 
# the notebook file that is in the part_a evidence folder

# Define a Convolutional Neural Network (CNN) architecture
# for CIFAR-10 image classification.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Automatically select GPU if available; otherwise use CPU.
device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')

print(device)

net.to(device)

# Define the loss function and optimization algorithm
# used during model training.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# Training Loop Implementation
# Train the model for one epoch and return the average loss.
def net_train_one_epoch(model, epoch, trainloader, optimizer, criterion):
    running_loss = 0.0
    total_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Clear previous gradients before each optimization step.
        optimizer.zero_grad()

        # Perform forward pass, loss calculation,
        # backpropagation, and parameter update.
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            total_loss += running_loss
            running_loss = 0.0

    print('Finished Training for epoch: ', epoch)
    return model, total_loss/len(trainloader)

# Evaluate model performance on the test dataset
# and collect predictions for later analysis.
def net_eval(model, testloader, device):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    # Disable gradient calculation to reduce memory usage
    # during evaluation.
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Save labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
    return accuracy, all_labels, all_predictions

# Initialize the selected experiment tracking platform
# (Weights & Biases, MLflow, or no tracking).
rank = int(os.environ.get("RANK", 0))
is_main_process = rank == 0


if args.tracker == "wandb" and is_main_process:
    try:
        import wandb

        wandb.init(
            project="cifar10-assessment",
            config={
                "dataset": "CIFAR10",
                "epochs_total": epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "momentum": momentum,
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss"
            }
        )

    except Exception as e:
        print(f"\nERROR: WandB initialization failed: {e}")
        print("Please login first:")
        print("    wandb login YOUR_API_KEY")
        raise

elif args.tracker == "mlflow" and is_main_process:
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///runs/mlflow.db")
        mlflow.set_experiment("CNN_CIFAR10")
        mlflow.start_run()

    except Exception as e:
        print(f"\nERROR: MLflow initialization failed: {e}")
        raise

elif is_main_process:
    print("Tracking disabled")
    
# Main training loop. Train the model, evaluate performance,
# log metrics, and save the best-performing checkpoint.
PATH = './cifar_best_model.pth'
best_test_accuracy = 0
for epoch in range(epochs):
    
    net, train_loss = net_train_one_epoch(
    net, epoch+1, trainloader, optimizer, criterion
    )
    test_accuracy, all_labels, all_predictions = net_eval(net, testloader, device)
    #logging the metrics by wandb
    if args.tracker == "wandb" and is_main_process:
        wandb.log({
            "train_loss": train_loss,
            "test_accuracy": test_accuracy
        }, step=epoch + 1)
    
    elif args.tracker == "mlflow" and is_main_process:
        mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
        mlflow.log_metric("test_accuracy", test_accuracy, step=epoch + 1)    
    
    # Save the model checkpoint whenever a higher test accuracy
    # is achieved.
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        # Saving the best model
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_accuracy,
        }, PATH)
    
# Calculate classification accuracy for each CIFAR-10 class.
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# Generate and visualize the confusion matrix to analyse
# classification performance across classes.
cm = confusion_matrix(all_labels, all_predictions)
print(cm)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", bbox_inches="tight")
print("Confusion matrix saved to confusion_matrix.png")
plt.close()

# Load the best saved model checkpoint for verification
# and inference testing.
checkpoint = torch.load(PATH)

net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

saved_test_accuracy = checkpoint['test_accuracy']

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# Save final tracking artifacts and properly close
# the experiment tracking session.
if args.tracker == "wandb" and is_main_process:
    artifact = wandb.Artifact("best-cifar10-model", type="model")
    artifact.add_file("cifar_best_model.pth")
    wandb.log_artifact(artifact)
    wandb.finish()

elif args.tracker == "mlflow" and is_main_process:
    mlflow.end_run()

# Re-evaluate the loaded best model to verify checkpoint recovery.
verified_test_accuracy, all_labels, all_predictions = net_eval(
    net, testloader, device
)

print(
    f"Saved checkpoint accuracy is {saved_test_accuracy} "
    f"and verified loaded model accuracy is {verified_test_accuracy}. "
    f"Best training-loop accuracy is {best_test_accuracy}."
)