
# Mnist Library Utilisation
# Outputs a prediction of a number from Mnist's after training on 60,000 samples

# 60, 000 Trained Sets of Written Numbers
# 10, 000 Unique Testing Samples

'''
Typical operation:
input date > weight > hidden layer 1 ( activation function ) > weights > hiddent layer 2... output layer

compare output to intended output > cost function (cross entropy)
optimisation function (optimizer) > minimize cost (AdamOptimizer...SGP, AdaGrad)

backwards progation of weights

feed forward + backprop = epoch   --aka. cycle to lower cost function
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # the 'hot 1' in array of 0's

# Nodes on each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes   = 10
batch_size  = 100

# Matrix = heigh * width
x = tf.placeholder('float',[None, 784 ]) # Input data, converted to 784 pixel matrix
y = tf.placeholder('float') 			 # Label of data


# Model to run data through
def neural_network_model(data):
	# Layer Passing
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), # Create tensor, aka array of random numbers in the hiddel layer
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes])),}

	# Model  = (input_data * weights) + biases
					# layer input data
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']) # l1 = 'sum box', whether the threshhold function fires
	l1 = tf.nn.relu(l1) # activation function 'recitified linear' applied to layer 1

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) 

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']) 
	l3 = tf.nn.relu(l3) 

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output #output is array with a '1' on a category and the remainder categories having a '0'

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) ) # cost of prediction to prediction label
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10 # cycles feed forward + backprop

	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )

		# Train data -- Optimize weight cost with x,y weight modification
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)): 
				epoch_x, epoch_y = mnist.train.next_batch( batch_size )
				_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y} )
				epoch_loss += c
			print("Epoch ", epoch, " completed out of ", hm_epochs, " loss: ", epoch_loss)

		# Run through Optimized Weights to model
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels})) # Evaluate

train_neural_network(x)