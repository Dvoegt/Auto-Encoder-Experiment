import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import optuna

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(trial):
    #Get parameters
    #learning_rate = 0.0002
    #epoch_rate = 5
    #batch_test = 30
    learning_rate = trial.suggest_float('lr', 0.0002, 0.0005, log = True)
    epoch_rate = trial.suggest_int('epoch_rate', 12, 14, log = True)
    batch_test = trial.suggest_int('batch_test',35, 60)
    #learning_rate = 0.00035
    #batch_test = 31
    #epoch_rate = 12

    
    # Set up the Fashion-MNIST dataset and dataloaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    fashion_mnist_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(fashion_mnist_dataset, batch_size=batch_test, shuffle=True)

    
    '''
    # Set up the MNIST dataset and dataloaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_test, shuffle=True)
    
    '''

    # Initialize the autoencoder
    autoencoder = Autoencoder()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)


    # Training the autoencoder
    num_epochs = epoch_rate

    for epoch in range(num_epochs):
        for data in dataloader:
            inputs, _ = data
            
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs.view(inputs.size(0), -1))
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    '''
    #Get test data
    
    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    '''
    # Get test data for Fashion-MNIST
    mnist_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    

    # Test the autoencoder on a sample from the dataset
    test_data = next(iter(dataloader))
    test_inputs, _ = test_data
    test_outputs = autoencoder(test_inputs)
    loss = criterion(outputs, inputs.view(inputs.size(0), -1))
    
    
    # Visualize the original and reconstructed images
    n = min(test_inputs.size(0), 8)
    plt.figure(figsize=(20, 4))
    
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_inputs[i].numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(test_outputs[i].detach().numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    plt.show()
    
    

    
    return(loss.item())

parameters = [{"name": "lr", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},]
#loss = train_autoencoder(parameters)
#print('total loss is: ',loss)

# Create an Optuna study object with the direction to minimize
study = optuna.create_study(direction='minimize')

# Run optimization
study.optimize(train_autoencoder, n_trials=20)

# Get the best parameters
best_params = study.best_params
print(best_params)