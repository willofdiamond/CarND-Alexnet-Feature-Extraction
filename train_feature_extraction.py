import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
file_location = "D:/Udacity_selfdriving/CarND-Alexnet-Feature-Extraction/"
file_name = "train.p"
input_data = pickle.load(open(file_location+file_name,"rb"))
print(input_data["features"].shape)
X_data = input_data["features"]
y_labels = input_data["labels"]
# TODO: Split data into training and validation sets.
X_train,y_train,X_test,y_test = train_test_split(input_data,input_labels,test_size = 0.3,random_state = 10)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32,(None,32,32,3))
resized = tf.image.resize(features,(227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)


# TODO: Add the final layer for traffic sign classification.

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
