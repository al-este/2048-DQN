import logic
import constants as c
from os import system, name
from math import log
from random import choices, choice, randint

class game():
	def __init__(self, random = False):
		self.matrix = logic.new_game(4)
		self.matrix = logic.add_two(self.matrix)
		self.matrix = logic.add_two(self.matrix)

		if(random):
			self.create_random()

		self.state = 'not over'

		self.score = 0

	def create_random1(self):
		nums = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
		weights1 = (0.177, 0.152, 0.126, 0.126, 0.101, 0.088, 0.076, 0.063, 0.05, 0.025, 0.012)
		weights = (0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001)

		for i in range(4):
			for j in range(4):
				self.matrix[i][j] = choices(nums, weights)[0]

	def create_random(self):
		nums = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

		for i in range(4):
			for j in range(4):
				self.matrix[i][j] = 0

		for i in range(randint(1,4)):
			num = choice(nums)
			self.matrix = logic.add_num(self.matrix, num)
			self.matrix = logic.add_num(self.matrix, num)

	def print_matrix(self, clear = True):
		if clear:
		    if name == 'nt': 
		        _ = system('cls') 
		  
		    else: 
		        _ = system('clear') 

		print('________________________')
		for i in range(4):
			print("|{: ^4}  {: ^4}  {: ^4}  {: ^4}|".format(self.matrix[i][0],
															self.matrix[i][1],
															self.matrix[i][2],
															self.matrix[i][3]))
		print('________________________')

	def get_matrix(self, show = False):
		out = list()

		for i in range(4):
			out.append(list())
			for j in range(4):
				out[i].append(list())
				if self.matrix[i][j] == 0:
					out[i][j] = [1,0,0,0,0,0,0,0,0,0,0,0]
				else:
					out[i][j].append(0)
					for k in range(1, 12):
						if int(log(self.matrix[i][j],2)) == k:
							out[i][j].append(1)
						else:
							out[i][j].append(0)
				if show:
					print(out[i][j])

		return out

	def get_state(self):
		return self.state

	def get_score(self):
		return self.score

	#Las teclas para cada movimiento son:
	# w - arriba
	# s - abajo
	# a - izquierda
	# d - derecha
	def movement(self, move):
		if move == 'w':
			self.matrix, done, points = logic.up(self.matrix)
		elif move == 's':
			self.matrix, done, points = logic.down(self.matrix)
		elif move == 'a':
			self.matrix, done, points = logic.left(self.matrix)
		elif move == 'd':
			self.matrix, done, points = logic.right(self.matrix)
		else:
			done = False
			print('Error, not a valid move')
			points = 0

		self.score += points

		self.state = logic.game_state(self.matrix)

		if done and self.state == 'not over':
			self.matrix = logic.add_two(self.matrix)

		return points