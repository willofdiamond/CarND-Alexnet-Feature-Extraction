import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
file_location = "/Users/hemanth/Udacity/alexdata/"
file_name = "train.p"
input_data = pickle.load(open(file_location+file_name,"rb"))
print(input_data["features"].shape)
X_data = input_data["features"]
y_labels = input_data["labels"]
nb_classes = 43
# TODO: Split data into training and validation sets.
X_train,X_test,y_train,y_test = train_test_split(X_data,y_labels,test_size = 0.3,random_state = 10)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#print("X_tain: {},y_train:{},X_test: {}, y_test:{}").format(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# TODO: Define placeholders and resize operation.
EPOCHS = 10
BATCH_SIZE = 150
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
rate = 0.005
features = tf.placeholder(tf.float32,(None,32,32,3))
resized = tf.image.resize_images(features,(227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)


# TODO: Add the final layer for traffic sign classification.
fc7shape = (fc7.get_shape().as_list()[-1], nb_classes)
print(fc7.get_shape().as_list())
fc8W = tf.Variable(tf.random_normal(shape = fc7shape, stddev = 1e-4))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7,fc8W)+fc8b
#probabilities = tf.nn.softmax(fc8

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
## Trained model here.
# Model training parameters

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
# Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Feel free to use as many code cells as needed.
save_file = '/Users/hemanth/Udacity/CarND-Alexnet-Feature-Extraction/train_model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            loss = sess.run(training_operation, feed_dict={features: batch_x, y: batch_y})

        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_test, y_test)
        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')




# TODO: Train and evaluate the feature extraction model.
