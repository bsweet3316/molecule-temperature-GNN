# Melting Temperature GNN and Residual Nerual Network model.



## Requirements

- PyTorch
- sklearn
- torch_geometric
- Molmass
- pandas
- matplotlib


## Usage

Running the main train.py file will start the training, updating of the hyperparameters is located in this file as well.
```
python train.py
```


After training is complete the plot will be displayed in the browser or console if using spyder (anaconda).


## Files:

- preprocess.py functions:
  - getData(): main function to fully load and pre-process te data from data.xlsx
  - splitFormula(): calculates the mass composition percentage for one chemical formula
  - breakdownElements(): calculates the feature vector for each node
  - removeColumns(): removes the columns in the pandas dataframe that are generated during pre-processing
- network.py functions:
  - init: initializes the network by defining all layers
  - forward: feeds a data point through the network to produce the predicted temperature
- train.py functions/classes:
  - createLoader(): creates the graph data object using the edge_index
  - Trainer Class:
    - init: initialize the model (defined in network.py) and optimizer (PyTorch Adam)
    - trian: perform one round of training then update the weights
  - Tester Class: 
    - init: initialize the model (defined in network.py)
    - test: evaluate the model using RMSE 
    - plotActualVsPredicted: output the predicted vs actual plot
  - main: Read in the data (by calling the getData()), perform the specified rounds of training (calling the Trainer.train() then evaluate both sets of data with Tester.test()), generate the RMSE by epoch plot.

  


