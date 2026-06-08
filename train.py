# Importing required madules
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

# Using argparse to take the argument from the user
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

# Note all the pakage versions in the requirement.txt file
os.system("pip freeze > requirements.txt")

# Setting random seeds for all random operation and setting determinitic flags
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to {SEED}")

#Data loading and processing
# Defining transform and loading the CIFAR10 dataset
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#batch_size = 4

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

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Model ArchitectureDefining- CNN
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

# Device and optimization setup
device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

net.to(device)

#Defining optimiser and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# Training Loop Implementation
#defining a train function, becuase it can be used later for checkpointing.
def net_train_one_epoch(model, epoch, trainloader, optimizer, criterion):
    running_loss = 0.0
    total_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
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
# defining an evalutaion function that is more suitable when needed to evaluation at checkpointing.
def net_eval(model, testloader, device):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    # since we're not training, we don't need to calculate the gradients for our outputs
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

# Starting the tracker before the first training
rank = int(os.environ.get("RANK", 0))
is_main_process = rank == 0

if args.tracker == "wandb" and is_main_process:
    import wandb
      
    wandb.init(
        project="cifar10-assessment",
        config={
            "dataset": "CIFAR10",
            "epochs_total": epochs,
            "Learning_rate": lr,
            "batch_size": batch_size,
            "momentum": momentum,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss"        
        }
    )
        
elif args.tracker == "mlflow" and is_main_process:
    import mlflow
    
    mlflow.set_tracking_uri("sqlite:///runs/mlflow.db")
    mlflow.set_experiment("CNN_CIFAR10")
    mlflow.start_run()

else:
    print("Tracking disabled")

# Training loop
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
    

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        # Saving the best model
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_accuracy,
        }, PATH)
    

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
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
# Confusion matrix
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

plt.show()
# setting checkpoint
checkpoint = torch.load(PATH)

net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_accuracy = checkpoint['test_accuracy']

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
# finalizing tracking
if args.tracker == "wandb" and is_main_process:

    artifact = wandb.Artifact("best-cifar10-model", type="model")
    
    artifact.add_file("cifar_best_model.pth")
    
    wandb.log_artifact(artifact)
    wandb.finish()
elif args.tracker == "mlflow" and is_main_process:
    mlflow.end_run()
# loading the best model for accuracy calculation
checkpoint = torch.load(PATH)

net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_accuracy = checkpoint['test_accuracy']

test_accuracy, all_labels, all_predictions = net_eval(net, testloader, device)
print(f"Test Accuracy from saved best model is {test_accuracy} and test accuracy from the training loop is {best_test_accuracy}")
