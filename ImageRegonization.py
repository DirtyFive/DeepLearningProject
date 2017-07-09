import cv2
import numpy as np
import os
import tensorflow as tf
from random import shuffle

def getData():
    dog_path = "C:\\Users\\gross\\Documents\\Data\\dogs_union"
    frog_path = "C:\\Users\\gross\\Documents\\Data\\frogs_union"
    i = 0
    set = []
    data = []
    labels = []
    # Rescale same..
    dim =(100,100)
    # Dog
    dog_amount = len(os.listdir(dog_path))
    for file_path in os.listdir(dog_path):
        try:
            img = cv2.imread(dog_path + "\\" + file_path,0)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            #cv2.imshow("dog",img)
            #cv2.waitKey(1)
            n_img = np.asarray(img).flatten()
            n_img.astype(float)
            data.append(n_img)
            i = i + 1
            print("Dog X:" + str(n_img.shape))
        except:
            pass
    i = 0
    while i < dog_amount:
        labels.append(np.asarray([1,0],dtype=float))
        i = i + 1
        print("Dog Y:" + str(labels[-1].shape)) 
    # Frog
    i = 0
    frog_amount = len(os.listdir(frog_path))
    for file_path in os.listdir(frog_path):
        try:
            img = cv2.imread(frog_path + "\\" + file_path,0)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            #cv2.imshow("frog",img)
            #cv2.waitKey(1)
            n_img = np.asarray(img,).flatten()
            n_img.astype(float)
            data.append(n_img)
            i = i + 1
            print("Frog:" + str(n_img.shape))
        except:
            pass
    i = 0
    while i < frog_amount:
        labels.append(np.asarray([0,1],dtype=float))
        i = i + 1
        print("Frog Y:"+ str(labels[-1].shape)) 
    data = np.array(data,dtype=float)
    labels = np.array(labels,dtype=float)
    set = (data,labels)
    return set

def shuffle_batch(batch):
    rows = batch[0].shape[0]
    lottery = [i for i in range(rows)]
    shuffle(lottery)
    return (batch[0][lottery], batch[1][lottery])

def train_test_batch(batch):
    total_rows = batch[0].shape[0]
    train_rows = int(batch[0].shape[0]*0.9)
    print("Training-Rows: " + str(int(batch[0].shape[0]*0.9)))
    print("Test-Rows: " + str(total_rows - train_rows))
    print("Total-Rows: " + str(total_rows))
    return batch[:][0:train_rows], batch[:][train_rows:]

def subBatch(batch, batchIndex, size):
    return (np.array(batch[0][batchIndex:(batchIndex+size)],dtype=float), np.array(batch[1][batchIndex:(batchIndex+size)],dtype=float))

batch = getData()
shuffled_batch = shuffle_batch(batch)
train_rows = int(batch[0].shape[0]*0.9)
train_batch = (shuffled_batch[0][0:train_rows], shuffled_batch[1][0:train_rows])
test_batch = (shuffled_batch[0][train_rows:], shuffled_batch[1][train_rows:])
#train_batch, test_batch = train_test_batch(shuffle_batch(data))


d = 0
train_batch[0].astype(float)
while d < train_batch[0].shape[0]:
    train_batch[0][d].astype(float)
    d = d + 1

train_batch[1].astype(float)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# TENSORFLOW!!!!!!!!!
image_size = 100
flatten_image_size = image_size * image_size
x = tf.placeholder(tf.float32, shape=[None, flatten_image_size])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
# First Convolutional Layer
W_conv1 = weight_variable([5,5,1,32]) # (5,5) = patch size, 1 = input_channels, 32 = output_channels
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,image_size,image_size,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([5,5,32,64]) # (5,5) = patch size, 32 = input_channels, 64 = output_channels
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([25*25*64,1024]) # 7*7*64
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*64]) # 7*7*64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
d = 0
batch_size = 50
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(1000):
    batch = subBatch(train_batch, d, batch_size)
    d += batch_size
    if(d>=train_batch[0].shape[0]):
        d = 0
    #print(str(batch[0].shape) + " " + str(batch[1].shape))
    #print(str(batch[0].dtype) + " " + str(batch[1].dtype))
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  #print(test_batch)
  #print("------------------")
  print('test accuracy %g' % accuracy.eval(feed_dict={x: test_batch[0], y_:test_batch[1]}))
  save_path = saver.save(sess, "./model.ckpt")
  print("Model saved in file: %s" % save_path)
  saver.restore(sess, "./model.ckpt")
  print("Model restored.")
