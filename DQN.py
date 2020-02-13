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

from random import choices, choice

from math import log

N_SET = 500
BATCH_SIZE = 100

DELTA = 0.85
EPS = 0.3

def create_model_1():
	def CBGN(model,filters,ishape=0):
		if (ishape!=0):
			model.add(Conv2D(filters, (3, 3), padding='same',
		             input_shape=ishape))
		else:
			model.add(Conv2D(filters, (3, 3), padding='same'))
		#model.add(BN())
		#model.add(GN(0.3))
		model.add(Activation('relu'))

		model.add(Conv2D(filters, (2, 2), padding='valid'))
		#model.add(BN())
		#model.add(GN(0.3))
		model.add(Activation('relu'))

		return model

	model = Sequential()
	model = CBGN(model,32,(4, 4, 12))
	model = CBGN(model, 128)
	model = CBGN(model, 128)

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dense(4))
	model.add(Activation('linear'))

	model.summary()

	return model

def create_model_2():
	def CBGN(model,filters,ishape=0):

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

	model.add(Dense(128))
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
	i = np.argmax(predict)
	if i == 0:
		return 'w'
	elif i == 1:
		return 'd'
	elif i == 2:
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


def train1():
	game_set = new_game_set()
	for it in range(1000):
		q_set = list()
		state_set = list()
		for i in range(N_SET):
			if game_set[i].get_state() != 'not over':
				game_set[i] = game.game(False, i>(N_SET/2))

			if i >= N_SET/2:
				game_set[i] = (game.game(False, True))

			predict = model.predict(matrix_to_array(game_set[i].get_matrix()))
			r = game_set[i].movement(get_movement(predict))
			r = r / 2048.0
			if r == 0:
				r = -0.5
			state_set.append(game_set[i].get_matrix())
			predict = model.predict(matrix_to_array(game_set[i].get_matrix()))
			predict = predict[0]	
			predict[np.argmax(predict)] = r + DELTA*max(predict)
			q_set.append(predict)


		model.fit(np.array(state_set), np.array(q_set), epochs = 1, steps_per_epoch = int(N_SET / BATCH_SIZE))

		if it%200 == 0 and it != 0:
			save_model(model)
			print(it)

def train2():
	game_set = new_game_set()
	r_set = list()
	lose_counter = 0

	for i in range(N_SET*2):
		predict = model.predict(matrix_to_array(game_set[i].get_matrix()))
		r = game_set[i].movement(get_movement(predict))
		r = r / 2048.0
		if r == 0:
			r = -0.2
		r_set.append(r)

	for it in range(100000):
		q_set = list()
		state_set = list()
		for i in range(N_SET*2):
			if game_set[i].get_state() != 'not over' and r_set[i] != 0:
				game_set[i].print_matrix()
				print(game_set[i].get_state())
				print(game_set[i].get_score())
				print(i)
				r_set[i]=0
				lose_counter += 1
			elif game_set[i].get_state() == 'not over':
				state_set.append(game_set[i].get_matrix())
				predict = model.predict(matrix_to_array(game_set[i].get_matrix()))
				r = game_set[i].movement(get_movement(predict))
				r = r / 2048.0
				if r == 0:
					r = -1
				pre = [0, 0, 0, 0]
				predict = predict[0]
				#game_set[i].print_matrix(False)
				print(predict)
				pre[np.argmax(predict)] = r_set[i] + DELTA*max(predict)
				q_set.append(pre)
				
				r_set[i] = r

		if lose_counter >= N_SET*2:
			break

		model.fit(np.array(state_set), np.array(q_set), epochs = 10, batch_size = 1)

def train3(eps):
	game_set = new_game_set()
	done = list()
	lose_counter = 0

	decay_factor = 0.99

	for i in range(N_SET*2):
		done.append(False)
	
	for it in range(10000):
		q_set = list()
		state_set = list()
		eps *= decay_factor
		for i in range(N_SET*2):
			if i >= N_SET:
				game_set[i].create_random()
				done[i] = False
			if not done[i]:
				if np.random.random() < eps:
					predict = model.predict(matrix_to_array(game_set[i].get_matrix()))
					a = choices(['w', 'd', 's', 'a'], predict[0])[0]
					#a = choice(['w', 'd', 's', 'a'])
				else:
					predict = model.predict(matrix_to_array(game_set[i].get_matrix()))
					a = get_movement(predict)

				state = game_set[i].get_matrix()
				state_set.append(state)

				r = game_set[i].movement(a)

				r = r / 2048.0
				if r == 0:
					r = -0.1

				if game_set[i].get_state() == 'lose':
					done[i] = True
					lose_counter += 1
					game_set[i].print_matrix(False)
					#print(game_set[i].get_state())
					#print(game_set[i].get_score())
					print('Lose counter '+str(lose_counter))
					print('i '+str(i))
					print('eps '+str(eps))

				target = r + DELTA*max(model.predict(matrix_to_array(game_set[i].get_matrix()))[0])

				predict = model.predict(matrix_to_array(game_set[i].get_matrix()))

				q = predict[0]
				#print(r)
				q[np.argmax(predict)] = target
				q_set.append(q)

				#model.fit( np.expand_dims(state,axis=0), np.expand_dims(q,axis=0), epochs = 5, verbose=0)

		if lose_counter >= N_SET-1:
			break
		'''print(np.array(state_set).shape)
								print(np.array(q_set).shape)'''
		model.fit(np.array(state_set), np.array(q_set), epochs = 25, batch_size = 10, verbose=1)

		if it%100==0 and it!=0:
			print('Iteración' + str(it))
			save_model(model)

def train4(iteration, g_set):
	game_set = random_game_set()
	game_set1 = random1_game_set()
	
	q_set = list()
	state_set = list()
	def create_set(i, games):
			pre = model.predict(matrix_to_array(games[i].get_matrix()))
			a = get_movement(pre)

			state = games[i].get_matrix()
			state_set.append(image.img_to_array(state))

			r = games[i].movement(a)

			if r == 0:
				r = -0.1
			else:
				r = log(r, 2)/12.0

			predict = freezed_model.predict(matrix_to_array(games[i].get_matrix()))

			target = r + DELTA*max(predict[0])

			q = pre[0]
			q[np.argmax(predict)] = target
			q_set.append(q)

	for i in range(N_SET):
		create_set(i, game_set)

	for i in range(int(N_SET/2)):
		create_set(i, game_set1)

	for i in range(len(g_set)):
		if g_set[i].get_state() != 'not over':
			g_set[i] = game.game(False, False)
		create_set(i, g_set)

	model.fit(np.array(state_set), np.array(q_set), epochs = 1, batch_size = 10, verbose=1)

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

model = create_model_3()
freezed_model = create_model_3()

opt = Adam(lr=0.001)

model.compile(loss='mean_squared_error',
              optimizer=opt, metrics=['mse'])

freezed_model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['mse'])

#model = load_model(model)

copy_model(model, freezed_model)

g = game.game()
it = 0
gset = new_game_set(10)
while True:
	train4(it, gset)

	it += 1
	if it > 10:
		it = 0
		if len(gset) < 500:
			gset.append(new_game_set(10))
		else:
			gset = new_game_set(10)
		save_model(model)

	predict = model.predict(matrix_to_array(g.get_matrix()))
	a = get_movement(predict)
	g.movement(a)
	g.print_matrix(False)
	print(a)
	print(model.predict(matrix_to_array(g.get_matrix())))

	if g.get_state() != 'not over':
		g = game.game()
