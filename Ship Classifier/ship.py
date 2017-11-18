
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

X, Y, X_test, Y_test = pickle.load(open("complete_dataset.pkl", "rb"))

X, Y = shuffle(X, Y)

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()



img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)




network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)


network = conv_2d(network, 32, 3, activation='relu')


network = max_pool_2d(network, 2)


network = conv_2d(network, 64, 3, activation='relu')


network = conv_2d(network, 64, 3, activation='relu')


network = max_pool_2d(network, 2)


network = fully_connected(network, 512, activation='relu')


network = dropout(network, 0.5)


network = fully_connected(network, 2, activation='softmax')


network = regression(network, optimizer='bobby',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)


model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='myShip-classifier.tfl.ckpt')
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='ship-classifier')


model.save("myShip-classifier.tfl")
print("Network trained and saved as ship-classifier.tfl!")