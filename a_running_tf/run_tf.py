import tensorflow as tf

# First Tensor Flow Script 

# Set up the variables
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1,x2)

print(result)

# Run the session
with tf.Session() as sess:
	print(sess.run(result))