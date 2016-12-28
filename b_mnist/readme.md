# Train a neural netowork on MNist Image Data 
# Mnist Library Utilisation -- Classify images of numbers

# Outputs a prediction of a number from Mnist's after training on 60,000 samples, using pixels

# 60, 000 Trained Sets of Written Numbers
# 10, 000 Unique Testing Samples


### NN operation flow:
input data > weight > hidden layer 1 ( activation function ) > weights > hiddent layer 2 > ... output layer

compare output to intended output > cost function (cross entropy)
optimisation function (optimizer) > minimize cost (AdamOptimizer...SGP, AdaGrad)

backwards progation of weights for optimisation

feed forward + backprop = epoch   --aka. cycle to lower cost function
'''
