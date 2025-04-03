import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def train_for_epochs(model, optimizer, batch):
    #start of learning
    model.train()
    running_loss = 0.0
    cross = nn.CrossEntropyLoss()
    for images, labels in batch:
        #clear last gradients
        optimizer.zero_grad()
        #the model takes images as input and returns predictions.
        outputs = model(images)
        #calculating the loss between the prediction and real labels
        loss = cross(outputs, labels)
        #Calculation of gradients using backpropagation.
        loss.backward()
        #updating model parameters based on gradients
        optimizer.step()
        #sums up losses across all batches to calculate the average
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(batch.dataset)
    return train_loss

def test_model(model, test_loader):
    #putting the model into evaluation mode 
    model.eval()
    test_loss = 0.0
    correct = 0
    cross = nn.CrossEntropyLoss()
    for images, labels in test_loader:
        #the model receives images as input and returns predictions
        outputs = model(images)
        loss = cross(outputs, labels)
        test_loss += loss.item() * images.size(0)
        #selects the class with the maximum probability from the model predictions 
        _, predicted = torch.max(outputs, 1)
        #counts the number of correctly guessed tags
        correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy

def train_model(model, optimizer, train_loader, test_loader, epochs=10):
    train_losses = []
    test_losses = []
    accuracies = []
    for epoch in range(1, epochs+1):
        #training - training data
        train_loss = train_for_epochs(model, optimizer, train_loader)
        #test data evaluation
        test_loss, accuracy = test_model(model, test_loader)
        #for our future schedule
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, accuracy: {accuracy:.2f}%")
    return train_losses, test_losses, accuracies

def build_model():
    return nn.Sequential(
        #one-dimensional vectors
        nn.Flatten(),
        nn.Linear(784, 512),
        #activation function ReLU
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

def plot_results(train_losses, test_losses, accuracies, title_prefix=""):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title(f'{title_prefix} Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, accuracies, label='Test Accuracy', color='green')
    plt.title(f'{title_prefix} Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    #converts MNIST data into tensors and normalizes them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochs = 8
    #training with optimizer SGD
    print("Learning with SGD:")
    model_sgd = build_model()
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.03)
    train_losses_sgd, test_losses_sgd, acc_sgd = train_model(model_sgd, optimizer_sgd, train_loader, test_loader, epochs)
    plot_results(train_losses_sgd, test_losses_sgd, acc_sgd, title_prefix="SGD")
    #training with optimizer SGD with momuntum
    print("\nLearning with SGD with momentum:")
    model_sgdm = build_model()
    optimizer_sgdm = optim.SGD(model_sgdm.parameters(), lr=0.03, momentum=0.9)
    train_losses_sgdm, test_losses_sgdm, acc_sgdm = train_model(model_sgdm, optimizer_sgdm, train_loader, test_loader, epochs)
    plot_results(train_losses_sgdm, test_losses_sgdm, acc_sgdm, title_prefix="SGD with Momentum")
    #training with optimizer Adam
    print("\nLearning with Adam:")
    model_adam = build_model()
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
    train_losses_adam, test_losses_adam, acc_adam = train_model(model_adam, optimizer_adam, train_loader, test_loader, epochs)
    plot_results(train_losses_adam, test_losses_adam, acc_adam, title_prefix="Adam")

    print("\nResults:")
    print(f"SGD: {acc_sgd[-1]:.2f}%")
    print(f"SGD with momentum: {acc_sgdm[-1]:.2f}%")
    print(f"Adam: {acc_adam[-1]:.2f}%")

if __name__ == "__main__":
    main()