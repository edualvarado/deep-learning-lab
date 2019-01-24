import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

class cnn_model:
	def __init__(self):
		self.model = Sequential()

# Initialize model
cnn_m = cnn_model()
# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
# 
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
# Hint: to ease loading your model later create a model.py file
# where you define your network configuration
######################################

train_data = trans.get_train()
valid_data = trans.get_valid()

x = train_data[0].copy()
y = train_data[1].copy()

x_val = valid_data[0].copy()
y_val = valid_data[1].copy()

#y = y.astype(int)
#y_val = y_val.astype(int)


# Reshaping
x_rows = int(np.sqrt(opt.state_siz))
x_cols = x_rows
x_val_rows = int(np.sqrt(opt.state_siz))
x_val_cols = x_val_rows
x_samples = x.shape[0]
x_val_samples = x_val.shape[0]
hist = opt.hist_len
x = x.reshape(x_samples, x_rows , x_cols, hist)
x_val = x_val.reshape(x_val_samples, x_val_rows, x_val_cols, hist)

# Create cnn
cnn_m.model.add(Conv2D(128, kernel_size=5, padding='same', data_format="channels_last", input_shape=(x_rows, x_cols, hist)))
cnn_m.model.add(Activation('relu'))
cnn_m.model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_m.model.add(Conv2D(64, kernel_size=3, padding='same'))
cnn_m.model.add(Activation('relu'))
cnn_m.model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_m.model.add(Conv2D(32, kernel_size=3, padding='same'))
cnn_m.model.add(Activation('relu'))
cnn_m.model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_m.model.add(Dropout(0.25))

cnn_m.model.add(Flatten())
cnn_m.model.add(Dense(512))
cnn_m.model.add(Activation('relu'))
cnn_m.model.add(Dropout(0.5))
cnn_m.model.add(Dense(5))
cnn_m.model.add(Activation('softmax'))

# Define attributes of the cnn; categorial, optimizer_type, performance metrics
cnn_m.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
history = cnn_m.model.fit(x, y, epochs=20, batch_size=64, validation_data=(x_val, y_val), shuffle=True)

# 2. save your trained model
# serialize model to JSON
model_json = cnn_m.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_m.model.save_weights("model.h5")
print("Saved model to disk")

# list all data in history
print(history.history.keys())

# summarize history for accuracy
fig1 = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig1.savefig('model accuracy.png')

# summarize history for loss
fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig2.savefig('model loss.png')


