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

DELTA = 0.9
eps = 0.3

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
	#print("Saved model to disk")

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

	if iteration == 0:
		copy_model(model, target_model)
		print('model -> target_model')

def load_model(model):
	model.load_weights("model.h5")
	copy_model(model, freezed_model)
	print("Loaded model from disk")

	return model

def create_set(game, epsi):
		pre = model.predict(matrix_to_array(game.get_matrix()))
		if np.random.random() < epsi:
			a = randint(0, 3)
		else:
			a = np.argmax(pre)

		actions.append(a)
		state = game.get_matrix()
		prestates.append(state)
		state_set.append(image.img_to_array(state))

		r = game.movement(get_movement(a))

		poststates.append(game.get_matrix())

		if r == 0:
			if state==game.get_matrix():
				r = -0.07
			else:
				r = 0
		else:
			r = log(r, 2)/11.0

		rewards.append(r)

		predict = target_model.predict(matrix_to_array(game.get_matrix()))

		if game.get_state() == 'not over':
			terminals.append(False)
		else:
			terminals.append(True)

model = create_model()
target_model = create_model()

opt = Adam(lr=0.001)

model.compile(loss='mean_squared_error', optimizer=opt)

#model = load_model(model)

copy_model(model, target_model)

history = list()
score = list()

q_set = list()
state_set = list()

prestates = list()
actions = list()
rewards = list()
poststates = list()
terminals = list()

batch_size = 100
replay_memory_size = 100000
min_size = 5000
train_freq = 10
train_repeat = 10

g = game.game()
it = 0
try:
	while True:
		create_set(g, eps+0.05)
		eps = eps*0.9999

		if len(prestates) >= replay_memory_size:
			delidx = np.random.randint(0, len(prestates) - 1 - batch_size)
			del prestates[delidx]
			del actions[delidx]
			del rewards[delidx]
			del poststates[delidx]
			del terminals[delidx]

		if it > min_size:
			if it % train_freq == 0:
				for k in range(train_repeat):
					indexes = np.random.randint(len(prestates), size=batch_size)
					pre_sample = np.array([prestates[i] for i in indexes])
					post_sample = np.array([poststates[i] for i in indexes])
					qpre = model.predict(pre_sample)
					qpost = target_model.predict(post_sample)
					for i in range(len(indexes)):
						if terminals[indexes[i]]:
							qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
						else:
							qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + DELTA * np.amax(qpost[i])
					model.train_on_batch(pre_sample, qpre)

			if it % (train_freq*10) == 0:
				save_model(model)
				copy_model(model, target_model)
				

		it += 1

		if g.get_state() != 'not over':
			score.append(g.get_score())
			print("Score:{}, epsilon:{}".format(g.get_score(), eps))
			g = game.game()

except KeyboardInterrupt:
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Score', color=color)
	ax1.plot(score, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	ax2.set_ylabel('Score', color=color)  # we already handled the x-label with ax1
	ax2.plot(score, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()