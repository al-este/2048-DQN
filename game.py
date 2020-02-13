import logic
import constants as c
from os import system, name
from math import log
from tkinter import Frame, Label, CENTER
from random import choices, choice, randint

class game(Frame):
	def __init__(self, visualize = False, random = False):
		self.matrix = logic.new_game(4)
		self.matrix = logic.add_two(self.matrix)
		self.matrix = logic.add_two(self.matrix)

		if(random):
			self.create_random()

		self.state = 'not over'

		self.score = 0

		self.visualize = visualize
		if visualize:
			Frame.__init__(self)
			self.grid()
			self.grid_cells = []
			self.init_grid()
			self.update_grid()

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


	def init_grid(self):
		background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
		background.grid()

		for i in range(c.GRID_LEN):
			grid_row = []
			for j in range(c.GRID_LEN):
				cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
				cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
				t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
				t.grid()
				grid_row.append(t)

			self.grid_cells.append(grid_row)

	def update_grid(self):
		for i in range(c.GRID_LEN):
			for j in range(c.GRID_LEN):
				new_number = self.matrix[i][j]
				if new_number == 0:
					self.grid_cells[i][j].configure(
						text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
				else:
					self.grid_cells[i][j].configure(text=str(
						new_number), bg=c.BACKGROUND_COLOR_DICT[new_number],
						fg=c.CELL_COLOR_DICT[new_number])
		self.update_idletasks()

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

		if self.visualize:
			self.update_grid()

		return points

	def normalice_points(points):
		return points/2048.0