from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import matplotlib.pyplot as plt

import time
start_time = time.time()

# We downloaded the dataset directly from the tutorial page of TensorFlow.
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def deepnn(x):
  """Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
     Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
  """
  #############################

  ## 1. Convolutional Layers ##

  #################################
  ###     Number of filters     ###
  #################################
  global num_filters 
  num_filters = 16

  # We will use 2 Conv. Layers (16 ("num_filters") 3x3 filters  with 1 Stride), each followed by ReLU activation
  # and a max pooling layer.

  # First, we reshape our data (x) to a 4D Tensor, with the second and third dimension corresponding 
  # to the image weight and height, and the final dimension being the number of color channels.

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to "num_filters" feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 1, num_filters]) #16 3x3 filters
    b_conv1 = bias_variable([num_filters])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # [3, 3, 1, num_filters]: The first two dimensions are the patch size. The third dimension is the number 
  # of input channels, and the last is the number of output channels.

  # Pooling layer - downsamples by 2X (from 28x28 to 14x14).
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps "num_filters" feature maps to "num_filters" again (we keep the number of outputs).
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, num_filters, num_filters]) #16 3x3 filters
    b_conv2 = bias_variable([num_filters])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer - downsamples by 2X (from 14x14 to 7x7).
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  #################################

  ## 2. Densely Connected Layer ##

  #################################
  ###      Number of units      ###
  #################################
  global num_units 
  num_units = 128
  
  # We use now a Fully Connected Layer with 128 units (and others for the exercise, what we'll call "num_units") and a softmax layer to do the classification.
  # We reshape the tensor from the pooling layer into a batch of vectors, then we multiply by the
  # weight matrix, we add the bias and finally apply the ReLU Activation Function.


  # Fully connected layer 1 - Our 28x28 image is down to 7x7x"num_filters" feature maps.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * num_filters, num_units]) #"num_units", "num_filters" output, reshaped 7 (same)
    b_fc1 = bias_variable([num_units]) 

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters]) 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  #################################

  ## 3. Dropout ##

  # By doing Dropout, we can reduce the effect of overfitting.
  # It is very usefull, specially when we are dealing with large training data and neural networks.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  #################################

  ## 4. Readout Layer ##

  # Map the "num_units" features to 10 classes, one for each digit.
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([num_units, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

  #################################

  ## 5. Convolution and Pooling ##

  # According to the exercise, we will use a stride of 1 for our convolution layers.
  # They are also zero padded, so the output has the same size than the input.
  # To do so, we name two functions:

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

  #################################

  ## 6. Weight and Bias Initialization ##

  # We initialize the weights with a small amount of noise to prevent gradients equal to 0.
  # Because of the ReLU Activation Function, if we set them with a small positive initial bias,
  # we can avoid "dead neurons" which do not work.


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

  #################################

def main(_):

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the variable for the input data (images). Each image is a 784-dimensional vector. "None" sets the number of samples. Whole data set: 55000. 
  # There are 10000 points for test data, and 5000 for validation data.
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer. It is a 10-dimensional vector (each one for each possible prediction).
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net.
  y_conv, keep_prob = deepnn(x)

  # We use the Cross-Entropy to measure how inneficient our predictions are for describing the truth.
  # To calculate the entropy, we could also calculate log(y) and multiply it by y_. Then, we would 
  # need to calculate the sum and compute the mean over every example in the batch.
  # However, this method can led to numerical unbalances and it is not stable.
  # For that reason we will compute with this more stable method:

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  #################################
  ###    Training parameters    ###
  #################################

  batch_size = 50
  epochs = 10
  learning_rate = 0.1

  average_cost = 0
  train_Err_print = []
  valid_Err_print = []
  loss_print = []

  #################################

  # We use Gradient Descent Algorithm.
  with tf.name_scope('gradient_descent'):
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy) ### LEARNING RATE ###

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # We will use a nested loop in order to train our data. In our case, our batch size is set to 50, and the number of epochs set to 10.
  # With this configuration, and being the total amount of data 55000 samples, we know that the total number of batches will be 1100 (Each batch has 50 data, then: 55000/50).
  # For one epoch, we will train for each different batch size. When we have completed all batches, we display the validation error and loss, and immediately we start with the next epoch.
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
      for i in range (total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
	_, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
	average_cost += c / total_batch
	train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})       
	if i % 100 == 0: 
	  print("Epoch: ", (epoch+1), "Batch: ", i, "out of", total_batch, "Training accuracy: {:.3f}".format(train_accuracy))

      valid_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})       
      loss_function = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0}) 
      train_Err_print.append(1-train_accuracy) #We substract the accuracy to 1 to obtain the error. Then we add it to our lists.
      valid_Err_print.append(1-valid_accuracy)
      loss_print.append(loss_function)
      print("Validation accuracy: {:.3f}".format(valid_accuracy))
      print("Loss: {:.3f}".format(loss_function))
    
    # Print results:
    print("\n#############################")
    print("\nTraining has been completed")
    print("\nTotal number of Epochs trained:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate: ", learning_rate)
    print("Number of units: ", num_units)
    print("Number of filters: ", num_filters)
    print("\nValidation accuracy: {:.3f}".format(valid_accuracy))
    print("Loss: {:.3f}".format(loss_function))
    print("Test Error: ", 1-(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
    print("Runtime: %s seconds" % (time.time() - start_time))

    # To calculate the total number of parameters of our network, we iterate over all tensors in the graph, then multiply
    # by the dimensions of our filters, and then by adding the previously multipled values.
    total_parameters = 0
    #iterating over all variables
    for variable in tf.trainable_variables():  
    	local_parameters=1
    	shape = variable.get_shape()  #getting shape of a variable
    	for i in shape:
        	local_parameters*=i.value  #mutiplying dimension values
    	total_parameters+=local_parameters
    print("Total number of parameters:", total_parameters) 
    print("\n#############################")

    # To plot the validation and training error we will use the Matplot library
    line1, = plt.plot(valid_Err_print, label="Validation Error", linewidth=1)
    line2, = plt.plot(train_Err_print, label="Training Error", linewidth=1) 
    first_legend = plt.legend(handles=[line1,line2], loc=1)
    plt.xlabel('Number of Epoch')
    plt.ylabel('Error')
    plt.ylim((0,0.9))
    plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
