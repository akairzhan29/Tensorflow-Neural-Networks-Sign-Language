#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time


# ### The dataset twe are using is a subset of the sign language digits. It contains six different classes representing the digits from 0 to 5.

# In[3]:


train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")


# In[4]:


#get the slices in the form of objects

x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])


# In[6]:


print(x_train.element_spec) #returns the shape and dtype of each element


# In[7]:


print(next(iter(x_train)))


# In[8]:


unique_labels = set()
for element in y_train:
    unique_labels.add(element.numpy())
print(unique_labels)


# The following cell shows some of the images in the dataset.

# In[9]:


images_iter = iter(x_train)
labels_iter = iter(y_train)
plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(next(images_iter).numpy().astype("uint8"))
    plt.title(next(labels_iter).numpy().astype("uint8"))
    plt.axis("off")


# In[10]:


# Transform an image into a tensor and normalize its components.
def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1,])
    return image


# In[11]:


new_train = x_train.map(normalize)
new_test = x_test.map(normalize)


# In[12]:


new_train.element_spec


# In[13]:


print(next(iter(new_train)))


# In[14]:


# define the linear function for future use in neural network steps

def linear_function():
    
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.Variable(np.random.randn(4,3), name = "W")
    b = tf.Variable(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W,X), b)
    
    return Y


# In[15]:


result = linear_function()
print(result)

assert type(result) == EagerTensor, "Use the TensorFlow API"
assert np.allclose(result, [[-2.15657382], [ 2.95891446], [-1.08926781], [-0.84538042]]), "Error"
print("\033[92mAll test passed")


# In[16]:


# create the sigmoid function

def sigmoid(z):
    
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    
    return a


# In[17]:


result = sigmoid(-1)
print ("type: " + str(type(result)))
print ("dtype: " + str(result.dtype))
print ("sigmoid(-1) = " + str(result))
print ("sigmoid(0) = " + str(sigmoid(0.0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

def sigmoid_test(target):
    result = target(0)
    assert(type(result) == EagerTensor)
    assert (result.dtype == tf.float32)
    assert sigmoid(0) == 0.5, "Error"
    assert sigmoid(-1) == 0.26894143, "Error"
    assert sigmoid(12) == 0.9999939, "Error"

    print("\033[92mAll test passed")

sigmoid_test(sigmoid)


# **Expected Output**: 
# <table>
# <tr> 
# <td>
# type
# </td>
# <td>
# class 'tensorflow.python.framework.ops.EagerTensor'
# </td>
# </tr><tr> 
# <td>
# dtype
# </td>
# <td>
# "dtype: 'float32'
# </td>
# </tr>
# <tr> 
# <td>
# Sigmoid(-1)
# </td>
# <td>
# 0.2689414
# </td>
# </tr>
# <tr> 
# <td>
# Sigmoid(0)
# </td>
# <td>
# 0.5
# </td>
# </tr>
# <tr> 
# <td>
# Sigmoid(12)
# </td>
# <td>
# 0.999994
# </td>
# </tr> 
# 
# </table> 

# In[18]:


# Computes the one hot encoding for a single label
def one_hot_matrix(label, depth=6):
    
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), shape=[-1, ])
    
    return one_hot


# In[19]:


def one_hot_matrix_test(target):
    label = tf.constant(1)
    depth = 4
    result = target(label, depth)
    print("Test 1:",result)
    assert result.shape[0] == depth, "Use the parameter depth"
    assert np.allclose(result, [0., 1. ,0., 0.] ), "Wrong output. Use tf.one_hot"
    label_2 = [2]
    result = target(label_2, depth)
    print("Test 2:", result)
    assert result.shape[0] == depth, "Use the parameter depth"
    assert np.allclose(result, [0., 0. ,1., 0.] ), "Wrong output. Use tf.reshape as instructed"
    
    print("\033[92mAll test passed")

one_hot_matrix_test(one_hot_matrix)


# **Expected output**
# ```
# Test 1: tf.Tensor([0. 1. 0. 0.], shape=(4,), dtype=float32)
# Test 2: tf.Tensor([0. 0. 1. 0.], shape=(4,), dtype=float32)
# ```

# In[20]:


new_y_test = y_test.map(one_hot_matrix)
new_y_train = y_train.map(one_hot_matrix)


# In[21]:


print(next(iter(new_y_test)))


# In[27]:


# initialize parameters for neural network step

def initialize_parameters():
                                
    initializer = tf.keras.initializers.GlorotNormal(seed=1)   
    
    W1 = tf.Variable(initializer(shape=(25,12288)))
    b1 = tf.Variable(initializer(shape=(25, 1)))
    W2 = tf.Variable(initializer(shape=(12, 25)))
    b2 = tf.Variable(initializer(shape=(12,1)))
    W3 = tf.Variable(initializer(shape=(6,12)))
    b3 = tf.Variable(initializer(shape=(6,1)))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# In[28]:


def initialize_parameters_test(target):
    parameters = target()

    values = {"W1": (25, 12288),
              "b1": (25, 1),
              "W2": (12, 25),
              "b2": (12, 1),
              "W3": (6, 12),
              "b3": (6, 1)}

    for key in parameters:
        print(f"{key} shape: {tuple(parameters[key].shape)}")
        assert type(parameters[key]) == ResourceVariable, "All parameter must be created using tf.Variable"
        assert tuple(parameters[key].shape) == values[key], f"{key}: wrong shape"
        assert np.abs(np.mean(parameters[key].numpy())) < 0.5,  f"{key}: Use the GlorotNormal initializer"
        assert np.std(parameters[key].numpy()) > 0 and np.std(parameters[key].numpy()) < 1, f"{key}: Use the GlorotNormal initializer"

    print("\033[92mAll test passed")
    
initialize_parameters_test(initialize_parameters)


# **Expected output**
# ```
# W1 shape: (25, 12288)
# b1 shape: (25, 1)
# W2 shape: (12, 25)
# b2 shape: (12, 1)
# W3 shape: (6, 12)
# b3 shape: (6, 1)
# ```

# In[29]:


parameters = initialize_parameters()


# In[30]:


# implementing the forward propagation

def forward_propagation(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)                # Z1 = np.dot(W1, X) + b1
    A1 = tf.keras.activations.relu(Z1)                           # A1 = relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)               # Z2 = np.dot(W2, A1) + b2
    A2 = tf.keras.activations.relu(Z2)                           # A2 = relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)               # Z3 = np.dot(W3, A2) + b3
    
    return Z3


# In[31]:


def forward_propagation_test(target, examples):
    minibatches = examples.batch(2)
    parametersk = initialize_parameters()
    W1 = parametersk['W1']
    b1 = parametersk['b1']
    W2 = parametersk['W2']
    b2 = parametersk['b2']
    W3 = parametersk['W3']
    b3 = parametersk['b3']
    index = 0
    minibatch = list(minibatches)[0]
    with tf.GradientTape() as tape:
        forward_pass = target(tf.transpose(minibatch), parametersk)
        print(forward_pass)
        fake_cost = tf.reduce_mean(forward_pass - np.ones((6,2)))

        assert type(forward_pass) == EagerTensor, "Your output is not a tensor"
        assert forward_pass.shape == (6, 2), "Last layer must use W3 and b3"
        assert np.allclose(forward_pass, 
                           [[-0.13430887,  0.14086473],
                            [ 0.21588647, -0.02582335],
                            [ 0.7059658,   0.6484556 ],
                            [-1.1260961,  -0.9329492 ],
                            [-0.20181894, -0.3382722 ],
                            [ 0.9558965,   0.94167566]]), "Output does not match"
    index = index + 1
    trainable_variables = [W1, b1, W2, b2, W3, b3]
    grads = tape.gradient(fake_cost, trainable_variables)
    assert not(None in grads), "Wrong gradients. It could be due to the use of tf.Variable whithin forward_propagation"
    print("\033[92mAll test passed")

forward_propagation_test(forward_propagation, new_train)


# **Expected output**
# ```
# tf.Tensor(
# [[-0.13430887  0.14086473]
#  [ 0.21588647 -0.02582335]
#  [ 0.7059658   0.6484556 ]
#  [-1.1260961  -0.9329492 ]
#  [-0.20181894 -0.3382722 ]
#  [ 0.9558965   0.94167566]], shape=(6, 2), dtype=float32)
# ```

# In[70]:


# the function to compute the total loss

def compute_total_loss(logits, labels):    
    
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))

    return total_loss


# In[71]:


def compute_total_loss_test(target, Y):
    pred = tf.constant([[ 2.4048107,   5.0334096 ],
             [-0.7921977,  -4.1523376 ],
             [ 0.9447198,  -0.46802214],
             [ 1.158121,    3.9810789 ],
             [ 4.768706,    2.3220146 ],
             [ 6.1481323,   3.909829  ]])
    minibatches = Y.batch(2)
    for minibatch in minibatches:
        result = target(pred, tf.transpose(minibatch))
        break
        
    print(result)
    assert(type(result) == EagerTensor), "Use the TensorFlow API"
    assert (np.abs(result - (0.50722074 + 1.1133534) / 2.0) < 1e-7), "Test does not match. Did you get the reduce sum of your loss functions?"

    print("\033[92mAll test passed")

compute_total_loss_test(compute_total_loss, new_y_train )


# **Expected output**
# ```
# tf.Tensor(0.810287, shape=(), dtype=float32)
# ```

# In[72]:


# training the model
# Implement a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
    costs = []                                        # To keep track of the cost
    train_acc = []
    test_acc = []
    
    # Initialize parameters
    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # The CategoricalAccuracy will track the accuracy for this multiclass problem
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    
    # We can get the number of elements of a dataset using the cardinality method
    m = dataset.cardinality().numpy()
    
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)

    # Do the training loop
    for epoch in range(num_epochs):

        epoch_total_loss = 0.
        
        #We need to reset object to start measuring from 0 the accuracy each epoch
        train_accuracy.reset_states()
        
        for (minibatch_X, minibatch_Y) in minibatches:
            
            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

                # 2. loss
                minibatch_total_loss = compute_total_loss(Z3, tf.transpose(minibatch_Y))

            # We accumulate the accuracy of all the batches
            train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_total_loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_total_loss += minibatch_total_loss
        
        # We divide the epoch total loss over the number of samples
        epoch_total_loss /= m

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
            print("Train accuracy:", train_accuracy.result())
            
            # We evaluate the test set every 10 epochs to avoid computational overhead
            for (minibatch_X, minibatch_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_total_loss)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()


    return parameters, costs, train_acc, test_acc


# In[73]:


parameters, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, num_epochs=100)


# **Expected output**
# 
# ```
# Cost after epoch 0: 1.830244
# Train accuracy: tf.Tensor(0.17037037, shape=(), dtype=float32)
# Test_accuracy: tf.Tensor(0.2, shape=(), dtype=float32)
# Cost after epoch 10: 1.552390
# Train accuracy: tf.Tensor(0.35925925, shape=(), dtype=float32)
# Test_accuracy: tf.Tensor(0.30833334, shape=(), dtype=float32)
# ...
# ```
# Numbers you get can be different, just check that your loss is going down and your accuracy going up!

# In[74]:


# Plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show()


# In[75]:


# Plot the train accuracy
plt.plot(np.squeeze(train_acc))
plt.ylabel('Train Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
# Plot the test accuracy
plt.plot(np.squeeze(test_acc))
plt.ylabel('Test Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show()

