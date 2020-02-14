import game

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing import image
from keras.optimizers import Adam

from random import choices, choice, randint

from math import log
import matplotlib.pyplot as plt

N_SET = 1000

DELTA = 0.9
eps = 0.25

def create_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',
		             input_shape=(4, 4, 12)))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (2, 2), padding='valid'))
	model.add(Activation('relu'))

	model.add(Conv2D(128, (2, 2), padding='valid'))
	model.add(Activation('relu'))

	model.add(Conv2D(256, (2, 2), padding='valid'))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(4))
	model.add(Activation('linear'))

	model.summary()

	return model

def copy_model(source, target):
	target.set_weights(source.get_weights())

def matrix_to_array(matrix):
	return np.expand_dims(image.img_to_array(matrix),axis=0)

def get_movement(predict):
	if predict == 0:
		return 'w'
	elif predict == 1:
		return 'd'
	elif predict == 2:
		return 's'
	else:
		return 'a'

def save_model(s_model):
	model_json = s_model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	s_model.save_weights("model.h5")
	print("Saved model to disk")

def random_game_set():
	gset = list()
	for i in range(N_SET):
		gset.append(game.game(False, True))
	return gset

def random1_game_set():
	gset = list()
	for i in range(int(N_SET/2)):
		gset.append(game.game(False, True))
		gset[i].create_random1()
	return gset

def new_game_set(size):
	gset = list()
	for i in range(size):
		gset.append(game.game(False, False))
	return gset

def train(iteration, g_set):
	game_set = random_game_set()
	game_set1 = random1_game_set()
	q_set = list()
	state_set = list()
	def create_set(i, games):
			pre = model.predict(matrix_to_array(games[i].get_matrix()))
			if np.random.random() < eps:
				a = randint(0, 3)
			else:
				a = np.argmax(pre)

			state = games[i].get_matrix()
			state_set.append(image.img_to_array(state))

			r = games[i].movement(get_movement(a))

			if r == 0:
				r = -0.05
			else:
				r = log(r, 2)/11.0

			predict = freezed_model.predict(matrix_to_array(games[i].get_matrix()))

			target = r + DELTA*max(predict[0])

			q = pre[0]
			q[a] = target
			q_set.append(q)

	for i in range(N_SET):
		create_set(i, game_set)

	for i in range(int(N_SET/2)):
		create_set(i, game_set1)

	for i in range(len(g_set)):
		if g_set[i].get_state() != 'not over':
			g_set[i] = game.game(False, False)
		create_set(i, g_set)

	res = model.fit(np.array(state_set), np.array(q_set), epochs = 1, batch_size = 10, verbose=1)
	history.append(res.history['loss'])

	#print(np.array(state_set))
	#print(np.array(q_set))

	if iteration == 0:
		copy_model(model, freezed_model)
		print('model -> freezed_model')

def load_model(model):
	model.load_weights("model.h5")
	copy_model(model, freezed_model)
	print("Loaded model from disk")

	return model

model = create_model()
freezed_model = create_model()

opt = Adam(lr=0.001)

model.compile(loss='mean_squared_error', optimizer=opt)

freezed_model.compile(loss='mean_squared_error', optimizer=opt)

#model = load_model(model)

copy_model(model, freezed_model)

global historial

history = list()

g = game.game()
it = 0
gset = new_game_set(10)
try:
	while True:
		train(it, gset)

		it += 1
		if it > 10:
			it = 0
			if len(gset) < 500:
				gset.extend(new_game_set(10))
			else:
				gset = new_game_set(10)
			save_model(model)

		predict = model.predict(matrix_to_array(g.get_matrix()))
		a = get_movement(np.argmax(predict))
		g.movement(a)
		g.print_matrix(False)
		print(a)
		print("Epoch " + str(len(history)))
		print(model.predict(matrix_to_array(g.get_matrix())))

		if g.get_state() != 'not over':
			g = game.game()
except KeyboardInterrupt:
	plt.plot(history)
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.show()