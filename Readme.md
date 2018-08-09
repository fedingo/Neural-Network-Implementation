# Neural Network Implementation

This is a Neural Network implementation made for the Machine Learning Course. This implementation tries to use different optimization algorithms and different model selection techniques to compare them.

The project has been realized using Matlab


## Usage

To run this scripts it is necessary to add the data in a Matrix. The project is organized in folders with different implementations of the functions. You can find some Script examples in the main folder of the project.


## Project Organization

The project is divided in folders:
- Data: Monk's datasets, Cup dataset and Blind test dataset
- Gradient Descent Learning: script for the training with the Gradient Descent Learning
- Neural Network: feed forward and back propagation steps
- Polyak Learning: script for the training with the Polyak Learning
- Regularization L1: regularize function for L1
- Regularization L2: regularize function for L2

Then there are other scripts:
- CrossValidation
- Genetic_Model_Selection
- Random_Model_Selection
- AutoGrid_Model_Selection (for the automatized grid search)
- Model_Selection (for the Grid Search)
- RidgeRegression
- Search_Ridge (finds the best parameter for the regularization of the Ridge Regression)

The following scripts can be used to try the project:

NN
Cup:
- Cup_Grid_Script : Grid search
- Cup_Script: AutoGrid, Random and Genetic search

Monk's (used in the ML Project):
- Genetic_Script: Genetic search
- Grid_Script: Grid search
- Template_Script: Random search
