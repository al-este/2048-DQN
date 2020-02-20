import game

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Activation, Flatten, Add
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing import image
from keras.optimizers import Adam
from keras import backend as K

from random import choices, choice, randint

from math import log
import matplotlib.pyplot as plt

N_SET = 0

DELTA = 0.9
eps = 0.3

def create_model_DDQN():
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

def create_model():
	X_input = Input((4, 4, 12))
	X = X_input

	X = Conv2D(32, (3, 3), input_shape=(4, 4, 12), activation="relu", kernel_initializer='he_uniform', padding='same')(X)

	X = Conv2D(64, (2, 2), activation="relu", kernel_initializer='he_uniform', padding='valid')(X)

	X = Conv2D(128, (2, 2), activation="relu", kernel_initializer='he_uniform', padding='valid')(X)

	X = Conv2D(256, (2, 2), activation="relu", kernel_initializer='he_uniform', padding='valid')(X)
	X = Flatten()(X)

	state_value = Dense(256, kernel_initializer='he_uniform', activation="relu")(X)
	state_value = Dense(1, kernel_initializer='he_uniform')(state_value)
	state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(4,))(state_value)

	action_advantage = Dense(256, kernel_initializer='he_uniform', activation="relu")(X)
	action_advantage = Dense(4, kernel_initializer='he_uniform')(action_advantage)
	action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(4,))(action_advantage)

	X = Add()([state_value, action_advantage])

	model = Model(inputs = X_input, outputs = X)

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
	def create_set(game):
			pre = model.predict(matrix_to_array(game.get_matrix()))
			if np.random.random() < eps:
				a = randint(0, 3)
			else:
				a = np.argmax(pre)

			state = game.get_matrix()
			state_set.append(image.img_to_array(state))

			r = game.movement(get_movement(a))

			if r == 0:
				if state==game.get_matrix():
					r = -0.07
				else:
					r = 0
			else:
				r = log(r, 2)/11.0

			predict = freezed_model.predict(matrix_to_array(game.get_matrix()))

			target = r + DELTA*max(predict[0])

			q = pre[0]
			q[a] = target
			q_set.append(q)

	for g in game_set:
		create_set(g)

	for g in game_set1:
		create_set(g)

	mean_score = 0
	for g in g_set:
		loses = 0
		while True:
			if g.get_state() == 'not over':
				state = g.get_matrix()
				create_set(g)
				if state == g.get_matrix():
					loses += 1
					if loses >= 10:
						break
				else:
					loses = 0
			else:
				break
		mean_score += g.get_score()
	score.append(mean_score/len(g_set))

	res = model.fit(np.array(state_set), np.array(q_set), epochs = 5, batch_size = 10, verbose=1)
	history.append(res.history['loss'][0])

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

#model = load_model(model)

copy_model(model, freezed_model)

global historial

history = list()
score = list()

g = game.game()
it = 0
try:
	while True:
		gset = new_game_set(10)
		train(it, gset)

		it += 1
		if it > 10:
			it = 0
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
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss', color=color)
	ax1.plot(history, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('Score', color=color)  # we already handled the x-label with ax1
	ax2.plot(score, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()