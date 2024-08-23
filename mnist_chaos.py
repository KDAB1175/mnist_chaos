import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd

# Idea behind this was to make the model more like a brain, not all nodes are
# connected and some might go above a layer. This was cardinally stupid idea as it
# bypasses the needed itterations needed for it to work, also it is like 20k times 
# slower. In the end it is not even how the brain works.

epochs = 20
lr = 0.01

sparseMasks = []
sparseAntiMasks = []
firstLayerOut = None
sparseCounter = 0

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5):
        super(SparseLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        # Create a sparse mask
        self.mask = torch.rand(out_features, in_features) > sparsity
        sparseMasks.append(self.mask)

    def forward(self, x):
        global sparseCounter
        global firstLayerOut
        if sparseCounter == 0:
            firstLayerOut = self.fc.weight
        
        sparseCounter += 1
        
        if sparseCounter == 3:
            return F.linear(x, self._fitAntiMasks(), self.fc.bias)
        
        # Apply the mask to the weights        
        sparse_weights = self.fc.weight * self.mask
        return F.linear(x, sparse_weights, self.fc.bias)
    
    def _fitAntiMasks(self):
        global sparseMasks
        global sparseAntiMasks
        global firstLayerOut
        
        outputAntiMask = sparseAntiMasks[0]
        outputAntiWeights = outputAntiMask * firstLayerOut
        helpArray = []
        inputAntiMask = sparseAntiMasks[1]
        
        inputCounter = 0
        
        for i in range(outputAntiWeights.size(0)):
            for j in range(outputAntiWeights.size(1)):
                if outputAntiWeights[i, j] != 0:
                    helpArray.append(outputAntiWeights[i, j].item())
                    
        # Create a new tensor for the updated weights
        updated_weights = self.fc.weight.clone().detach()
        
        for i in range(inputAntiMask.size(0)):
            for j in range(inputAntiMask.size(1)):
                if inputAntiMask[i, j] != 0:
                    if inputCounter < len(helpArray):
                        updated_weights[i, j] = helpArray[inputCounter]
                        inputCounter += 1

        # Update the weights of the linear layer
        self.fc.weight = nn.Parameter(updated_weights)
        
        return self.fc.weight


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            SparseLinear(28*28, 512),
            nn.ReLU(),
            SparseLinear(512, 10),
            nn.ReLU(),
            SparseLinear(10, 10),
        )
        
        global sparseAntiMasks
        
        # Invert values in the mask
        sparseAntiMasks.append(~sparseMasks[0])
        sparseAntiMasks.append(~sparseMasks[-1])

    def forward(self, x):
        global sparseCounter
        sparseCounter = 0
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(model, train_loader, criterion, optimizer, device):
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Zero the gradients before the backward pass
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights

        total_loss += loss.item()  # Accumulate loss

        print(f'Batch {batch_idx}: Loss {loss.item()}')

    avg_loss = total_loss / len(train_loader)
    print(f'Average Loss: {avg_loss}')
        
# Prepare the data
def prepare_data():
    # Load the training and test data
    train_data = pd.read_csv('mnist_train.csv')
    test_data = pd.read_csv('mnist_test.csv')

    # Prepare the data (labels are in the first column, pixels in the remaining columns)
    X_train = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0  # Normalize pixel values
    y_train = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.long)

    X_test = torch.tensor(test_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
    y_test = torch.tensor(test_data.iloc[:, 0].values, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def main():
    # Model, criterion, optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    # Prepare data
    train_loader, _ = prepare_data()
    
    counter = 0 
    
    while counter < epochs:
        # Train the model
        train(model, train_loader, criterion, optimizer, device)
        counter += 1

    # Testing the model (optional)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Load test data
        _, test_loader = prepare_data()

        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main()
