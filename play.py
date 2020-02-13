import game

import numpy as np

from time import sleep

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import model_from_json

from keras.applications.vgg16 import VGG16
from matplotlib import pyplot

def plot_model(model):
	# retrieve weights from the second hidden layer
	filters, biases = model.layers[4].get_weights()
	# normalize filter values to 0-1 so we can visualize them
	f_min, f_max = filters.min(), filters.max()
	filters = (filters - f_min) / (f_max - f_min)
	# plot first few filters
	n_filters, ix = 8, 1
	for i in range(n_filters):
		# get the filter
		f = filters[:, :, :, i]
		# plot each channel separately
		for j in range(4):
			# specify subplot and turn of axis
			ax = pyplot.subplot(n_filters, 4, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(f[:, :, j], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()

def matrix_to_array(matrix):
	return np.expand_dims(image.img_to_array(matrix),axis=0)

def get_movement(predict):
	i = np.argmax(predict)
	if i == 0:
		return 'w'
	elif i == 1:
		return 'd'
	elif i == 2:
		return 's'
	else:
		return 'a'

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#loaded_model.summary()

#plot_model(loaded_model)


g = game.game(random = False)
g.print_matrix()

state = 'not over'

while state == 'not over':
	predict = loaded_model.predict(matrix_to_array(g.get_matrix()))
	a = get_movement(predict)
	r = g.movement(a)
	if r != 0:
		g.print_matrix(False)
		print(a)
		print(predict)
	sleep(0.1)
	state = g.get_state()