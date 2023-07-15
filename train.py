# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:49:14 2023

@author: starw
"""
import math
from sklearn.model_selection import train_test_split
from preprocess import getData
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
from network import Network
import torch.nn.functional as F
import matplotlib.pyplot as plt



lr = 0.0001
loss_fn = torch.nn.MSELoss()
epochs = 4000

'''
Input: data -> panda dataframe containing records to convert

The following function will convert the input dataframe into a dataloader object,
which will contain the graph data object for each input row.

returns: converted data loader object.
'''
def createLoader(data):
    data_list = []
    
    # edge index to define how the nodes are connected, first array is the source 
    # node and the second array is the destination node.
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], 
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]], dtype=torch.long)
    
    for index, row in data.iterrows():
        
        x = torch.tensor([row['E1Chars'], row['E2Chars'], row['E3Chars'], row['E4Chars']], dtype=torch.float)
        y = row['ML Tm (K)']
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        
    loader = DataLoader(data_list)
    return loader
        

# Trainer class 
class Trainer(object):
    
    # initialize the model and optimizer
    def __init__(self, model): 
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    # The following function will perform one round of training,
    # The weights are updated based on the RMSE after all inputs have been 
    # calculated.
    def train(self, train_loader):
        loss_all = 0
        
        for data in train_loader:
            data = data.to(device)
            output = model(data)
            loss = F.mse_loss(output, data.y)
            loss_all += loss
            
        loss_all = torch.sqrt(loss_all/len(train_loader))
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        return round(loss_all.item(), 5)
    
# Tester Class for 
class Tester(object):
    
    # Initialize the tester with the model
    def __init__(self, model):
        self.model = model
    
    # The following function will calculate the RMSE for the current version
    # of the model for the given dataset.
    def test(self, test_loader):
        
        loss_all = 0
        
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = F.mse_loss(output, data.y)
            loss_all += loss
        
        loss_all = torch.sqrt(loss_all/len(test_loader))
        return round(loss_all.item(), 5)
        
    # The following function will plot the Actual vs predicted temperatures
    # for a given dataset and the current model state.
    def plotActualVsPredicted(self, data_loader):
        
        actual = []
        predicted = []
        
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            predicted.append(output.item())
            actual.append(data.y)
            
        plot = plt.figure(2)
        plt.scatter(actual, predicted, s=7)
        plt.plot([0,4000], [0, 4000], linewidth = 2)
        plt.title("Actual VS Predicted Temperature")
        plt.xlabel("Actual Melting Temperature [K]")
        plt.ylabel("Predicted Melting Temperature [K]")
        
        plot.show()
        

# The following main function must be ran to start the application
# it will load the data, perform training for the specified number of epochs
# then produce the two plots.
if __name__ == "__main__":
    
    print("Pre-processing data")
    
    # Obtain the dataset that has already been pre-processed
    full_data = getData()
    
    # Split the pandas dataframe into a training and test split (80/20)
    train_data, test_data = train_test_split(full_data, test_size=0.2)   

    # Convert the datasets into dataLoader objects
    trainLoader = createLoader(train_data)
    testLoader = createLoader(test_data)
    
    # Initialize the neural network model
    model = Network()
    
    # Initialize the trainer and tester 
    trainer = Trainer(model)
    tester = Tester(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Begin training and initialize the arrays for the RMSE values.    
    model.train()
    Train_RMSEs = []
    Test_RMSEs = []
    epoch_num = []
    for epoch in range(epochs):
        # Update the weights
        loss = trainer.train(trainLoader)
        
        # Calculate the new RMSE values for the updated model for both datasets
        TRAIN_RMSE = tester.test(trainLoader)
        RMSE = tester.test(testLoader)
        
        # Add every fifth RMSE result to the plot data.
        if epoch % 5 == 0:
            
            Train_RMSEs.append(TRAIN_RMSE)
            Test_RMSEs.append(RMSE)
            epoch_num.append(epoch+1)
        print(f'{epoch}  Training Loss: {TRAIN_RMSE} Testing RMSE: {RMSE}')
    
    # Plot the RMSE results
    plt.plot(epoch_num, Train_RMSEs, linewidth=2,label="Training Data")
    plt.plot(epoch_num, Test_RMSEs, linewidth=2, label = "Testing Data")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("RMSE error for training and test datasets by epoch")
    plt.legend()
    
    tester.plotActualVsPredicted(testLoader)

         
    
            
    

    
    