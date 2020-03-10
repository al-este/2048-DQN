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

g = game.game(random = False)
g.print_matrix()

state = 'not over'

while state == 'not over':
	predict = loaded_model.predict(matrix_to_array(g.get_matrix()))[0]
	state = g.get_matrix()
	a = get_movement(predict)
	g.movement(a)
	if True:
		for i in range(3):
			if g.get_matrix() == state:
				predict[np.argmax(predict)] = -1000
				g.movement(get_movement(predict))
	else:
		if g.get_matrix() == state:
			break
	g.print_matrix(True)
	print(a)
	print(predict)
	print(g.get_score())
	#sleep(0.1)
	state = g.get_state()
