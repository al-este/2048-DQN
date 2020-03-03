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

DELTA = 0.95
eps = 0.15

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

def new_game_set(size):
	gset = list()
	for i in range(size):
		gset.append(game.game(False))
	return gset

def train(g_set):
	state_set = list()
	a_set = list()
	pre_set = list()
	post_set = list()
	r_set = list()
	def create_set(game, epsilon):
		pre = model.predict(matrix_to_array(game.get_matrix()))
		if np.random.random() < epsilon:
			a = randint(0, 3)
		else:
			a = np.argmax(pre)

		pre_set.append(pre[0])
		a_set.append(a)
		state = game.get_matrix()
		state_set.append(image.img_to_array(state))

		r = game.movement(get_movement(a))

		if r == 0:
			if state==game.get_matrix():
				r = -0.1
			else:
				r = 0
		else:
			r = log(r, 2)/11.0

		r_set.append(r)

		post_set.append(game.get_matrix())

	mean_score = 0
	scores=list()
	i_set=list()
	i=0
	for g in g_set:
		loses = 0
		while True:
			if g.get_state() == 'not over':
				state = g.get_matrix()
				create_set(g, loses/100.0)
				i_set.append(i)
				if state == g.get_matrix():
					loses += 1
					if loses >= 50:
						break
				else:
					loses = 0
			else:
				break
		i+=1
		scores.append(g.get_score())
		mean_score += g.get_score()
		g.print_matrix(False)
		print(i)

	max_score = max(scores)
	print(scores)
	print("{} -> {}".format(max_score, np.argmax(scores)+1))
	max_score_history.append(max_score)

	predicts = freezed_model.predict(np.array(post_set))

	r_set = np.array(r_set)*(np.array(scores)[np.array(i_set)]/max_score)

	targets = r_set + DELTA*np.max(np.array(predicts), axis=1)

	pre_set = np.array(pre_set)

	pre_set[range(len(pre_set)), a_set] = targets
	print(mean_score/len(g_set)+1)
	score.append(mean_score/len(g_set))

	res = model.fit(np.array(state_set), np.array(pre_set), epochs = 5, batch_size = 100, verbose=1)
	history.append(res.history['loss'][0])

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

history = list()
score = list()
max_score_history=list()

g = game.game()
it = 0
try:
	while True:
		gset = new_game_set(10)
		train(gset)

		it += 1
		if it >= 5:
			it = 0
			copy_model(model, freezed_model)
			print('model -> freezed_model')
			save_model(model)

		predict = model.predict(matrix_to_array(g.get_matrix()))
		a = get_movement(np.argmax(predict))
		g.movement(a)
		#g.print_matrix(False)
		#print(a)
		print("Epoch " + str(len(history)))
		#print(model.predict(matrix_to_array(g.get_matrix())))

		if g.get_state() != 'not over':
			g = game.game()
except KeyboardInterrupt:
	fig, ax1 = plt.subplots()

	color = 'tab:blue'
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Mean score', color=color)
	ax1.plot(score, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:red'
	ax2.set_ylabel('Max core', color=color)  # we already handled the x-label with ax1
	
	ax2.plot(max_score_history, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()