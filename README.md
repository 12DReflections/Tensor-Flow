# Tensor-Flow
My Codebase for Tensor Flow Neural Networks 
Install the requirements.txt and run the files in order to 
- Train a model
- Test the Model
- Run Apply the Neural Network 

### NN operation flow:
input data > weight > hidden layer 1 ( activation function ) > weights > hiddent layer 2 > ... output layer

compare output to intended output > cost function (cross entropy)
optimisation function (optimizer) > minimize cost (AdamOptimizer...SGP, AdaGrad)

backwards progation of weights for optimisation

feed forward + backprop = epoch   --aka. cycle to lower cost function
'''
