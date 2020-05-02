from pygame import gfxdraw
import pygame as pg
import matrix
import sys

'''
    File name: main.py
    Author: Michael Berge
    Date created: 7/19/2018
    Date modified: 5/2/2020
    Python Version: 3.8.1
'''

class Graphics:
    def __init__(self, args):
        if args[3] == 0:
            self.__num_i_nodes = args[0]
            self.__num_h1_nodes = args[1]
            self.__num_h2_nodes = args[1]
            self.__num_o_nodes = args[2]

            self.__SCREEN_WIDTH = 640
            self.__SCREEN_HEIGHT = 480
            self.white = (255, 255, 255)
            self.black = (50, 50, 50)
            self.red = (235, 0, 0)
            self.blue = (0, 0, 235)

            self.__i_node_spacing = self.__SCREEN_HEIGHT / (args[0] + 1)
            self.__h1_node_spacing = self.__SCREEN_HEIGHT / (args[1] + 1)
            self.__h2_node_spacing = self.__SCREEN_HEIGHT / (args[1] + 1)
            self.__o_node_spacing = self.__SCREEN_HEIGHT / (args[2] + 1)

            self.__i_nodes = []
            self.__h1_nodes = []
            self.__h2_nodes = []
            self.__o_nodes = []
            self.__ih1_weights = []
            self.__h2o_weights = []

            self.init_i_nodes(args[1])
            self.init_h1_nodes(args[1], False)
            self.init_h2_nodes(args[1], False)
            self.init_o_nodes(args[1])

            self.init_ih1_weights()
            self.init_h2o_weights()

            self.screen = self.open_window()
        
        else:
            self.__num_i_nodes = args[0]
            self.__num_h1_nodes = args[1]
            self.__num_h2_nodes = args[3]
            self.__num_o_nodes = args[2]

            self.__SCREEN_WIDTH = 640
            self.__SCREEN_HEIGHT = 480
            self.white = (255, 255, 255)
            self.black = (50, 50, 50)
            self.red = (235, 0, 0)
            self.blue = (0, 0, 235)

            self.__i_node_spacing = self.__SCREEN_HEIGHT / (args[0] + 1)
            self.__h1_node_spacing = self.__SCREEN_HEIGHT / (args[1] + 1)
            self.__h2_node_spacing = self.__SCREEN_HEIGHT / (args[3] + 1)
            self.__o_node_spacing = self.__SCREEN_HEIGHT / (args[2] + 1)

            self.__i_nodes = []
            self.__h1_nodes = []
            self.__h2_nodes = []
            self.__o_nodes = []
            self.__ih1_weights = []
            self.__h1h2_weights = []
            self.__h2o_weights = []

            max_h = max(args[1], args[3])
            self.init_i_nodes(max_h)
            self.init_h1_nodes(max_h, True)
            self.init_h2_nodes(max_h, True)
            self.init_o_nodes(max_h)

            self.init_ih1_weights()
            self.init_h1h2_weights()
            self.init_h2o_weights()

            self.screen = self.open_window()

    def open_window(self):
        pg.init()
        size = (self.__SCREEN_WIDTH, self.__SCREEN_HEIGHT)
        screen = pg.display.set_mode(size)
        pg.display.set_caption("Neural Network Visualization")
        return screen

    # Initialize
    def init_i_nodes(self, h):
        y = self.__i_node_spacing
        for i in range(self.__num_i_nodes):
            self.__i_nodes.append(Node(self.__SCREEN_WIDTH / 8, y, (int) (240 / (h + 15))))
            y += self.__i_node_spacing

    def init_h1_nodes(self, h, two_hidden_layers):
        y = self.__h1_node_spacing
        for i in range(self.__num_h1_nodes):
            if two_hidden_layers:
                self.__h1_nodes.append(Node((self.__SCREEN_WIDTH / 8) * 3, y, (int) (240 / (h + 15))))
            else:
                self.__h1_nodes.append(Node(self.__SCREEN_WIDTH / 2, y, (int) (240 / (h + 15))))
            y += self.__h1_node_spacing

    def init_h2_nodes(self, h, two_hidden_layers):
        y = self.__h2_node_spacing
        for i in range(self.__num_h2_nodes):
            if two_hidden_layers:
                self.__h2_nodes.append(Node((self.__SCREEN_WIDTH / 8) * 5, y, (int) (240 / (h + 15))))
            else:
                self.__h2_nodes.append(Node(self.__SCREEN_WIDTH / 2, y, (int) (240 / (h + 15))))
            y += self.__h2_node_spacing

    def init_o_nodes(self, h):
        y = self.__o_node_spacing
        for i in range(self.__num_o_nodes):
            self.__o_nodes.append(Node((self.__SCREEN_WIDTH / 8) * 7, y, (int) (240 / (h + 15))))
            y += self.__o_node_spacing

    def init_ih1_weights(self):
        for i in range(len(self.__i_nodes)):
            for j in range(len(self.__h1_nodes)):
                self.__ih1_weights.append(Weight((self.__i_nodes[i].x, self.__i_nodes[i].y), (self.__h1_nodes[j].x, self.__h1_nodes[j].y)))

    def init_h1h2_weights(self):
        for i in range(len(self.__h1_nodes)):
            for j in range(len(self.__h2_nodes)):
                self.__h1h2_weights.append(Weight((self.__h1_nodes[i].x, self.__h1_nodes[i].y), (self.__h2_nodes[j].x, self.__h2_nodes[j].y)))

    def init_h2o_weights(self):
        for i in range(len(self.__h2_nodes)):
            for j in range(len(self.__o_nodes)):
                self.__h2o_weights.append(Weight((self.__h2_nodes[i].x, self.__h2_nodes[i].y), (self.__o_nodes[j].x, self.__o_nodes[j].y)))

    # Draw
    def draw_input_nodes(self, screen, black, red):
        for i in range(len(self.__i_nodes)):
            if self.__i_nodes[i].data >= 0:
                color = black
            else:
                color = red
            pg.gfxdraw.filled_circle(screen, int(self.__i_nodes[i].x), int(self.__i_nodes[i].y), self.__i_nodes[i].size, color)
            pg.gfxdraw.aacircle(screen, int(self.__i_nodes[i].x), int(self.__i_nodes[i].y), self.__i_nodes[i].size, color)

    def draw_hidden_1_nodes(self, screen, black, red):
        for i in range(len(self.__h1_nodes)):
            if self.__h1_nodes[i].data >= 0:
                color = black
            else:
                color = red
            pg.gfxdraw.filled_circle(screen, int(self.__h1_nodes[i].x), int(self.__h1_nodes[i].y), self.__h1_nodes[i].size, color)
            pg.gfxdraw.aacircle(screen, int(self.__h1_nodes[i].x), int(self.__h1_nodes[i].y), self.__h1_nodes[i].size, color)

    def draw_hidden_2_nodes(self, screen, black, red):
        for i in range(len(self.__h2_nodes)):
            if self.__h2_nodes[i].data >= 0:
                color = black
            else:
                color = red
            pg.gfxdraw.filled_circle(screen, int(self.__h2_nodes[i].x), int(self.__h2_nodes[i].y), self.__h2_nodes[i].size, color)
            pg.gfxdraw.aacircle(screen, int(self.__h2_nodes[i].x), int(self.__h2_nodes[i].y), self.__h2_nodes[i].size, color)

    def draw_output_nodes(self, screen, black, red):
        for i in range(len(self.__o_nodes)):
            if self.__o_nodes[i].data >= 0:
                color = black
            else:
                color = red
            pg.gfxdraw.filled_circle(screen, int(self.__o_nodes[i].x), int(self.__o_nodes[i].y), self.__o_nodes[i].size, color)
            pg.gfxdraw.aacircle(screen, int(self.__o_nodes[i].x), int(self.__o_nodes[i].y), self.__o_nodes[i].size, color)

    def draw_ih1_weights(self, screen, blue, red):
        for i in range(len(self.__ih1_weights)):
            if self.__ih1_weights[i].data >= 0:
                color = blue
            else:
                color = red
            if 1 > self.__ih1_weights[i].data > -1:
                thickness = 1
            else:
                thickness = abs(round(self.__ih1_weights[i].data))
            pg.draw.line(screen, color, self.__ih1_weights[i].coor1, self.__ih1_weights[i].coor2, thickness)

    def draw_h1h2_weights(self, screen, blue, red):
        for i in range(len(self.__h1h2_weights)):
            if self.__h1h2_weights[i].data >= 0:
                color = blue
            else:
                color = red
            if 1 > self.__h1h2_weights[i].data > -1:
                thickness = 1
            else:
                thickness = abs(round(self.__h1h2_weights[i].data))
            pg.draw.line(screen, color, self.__h1h2_weights[i].coor1, self.__h1h2_weights[i].coor2, thickness)

    def draw_h2o_weights(self, screen, blue, red):
        for i in range(len(self.__h2o_weights)):
            if self.__h2o_weights[i].data >= 0:
                color = blue
            else:
                color = red
            if 1 > self.__h2o_weights[i].data > -1:
                thickness = 1
            else:
                thickness = abs(round(self.__h2o_weights[i].data))
            pg.draw.line(screen, color, self.__h2o_weights[i].coor1, self.__h2o_weights[i].coor2, thickness)

    # Update
    def update_input_nodes(self, arr):
        for i in range(len(arr)):
            self.__i_nodes[i].data = arr[i]

    def update_hidden_1_nodes(self, arr):
        for i in range(len(arr)):
            self.__h1_nodes[i].data = arr[i]

    def update_hidden_2_nodes(self, arr):
        for i in range(len(arr)):
            self.__h2_nodes[i].data = arr[i]

    def update_output_nodes(self, arr):
        for i in range(len(arr)):
            self.__o_nodes[i].data = arr[i]

    def update_ih1_weights(self, arr):
        for i in range(len(arr)):
            self.__ih1_weights[i].data = arr[i]

    def update_h1h2_weights(self, arr):
        for i in range(len(arr)):
            self.__h1h2_weights[i].data = arr[i]

    def update_h2o_weights(self, arr):
        for i in range(len(arr)):
            self.__h2o_weights[i].data = arr[i]

    # Draw screen
    def draw1(self, input_, hidden, output, __weights_ih, __weights_ho):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit(0)

        self.screen.fill(self.white)
        # Update
        self.update_input_nodes(matrix.Matrix.to_array(input_))
        self.update_hidden_1_nodes(matrix.Matrix.to_array(hidden))
        self.update_output_nodes(matrix.Matrix.to_array(output))
        self.update_ih1_weights(matrix.Matrix.to_array(__weights_ih))
        self.update_h2o_weights(matrix.Matrix.to_array(__weights_ho))
        # Draw
        self.draw_ih1_weights(self.screen, self.blue, self.red)
        self.draw_h2o_weights(self.screen, self.blue, self.red)
        self.draw_input_nodes(self.screen, self.black, self.red)
        self.draw_hidden_1_nodes(self.screen, self.black, self.red)
        self.draw_output_nodes(self.screen, self.black, self.red)
        pg.display.update()

    def draw2(self, input_, hidden_1, hidden_2, output, __weights_ih1, __weights_h1h2, __weights_h2o):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit(0)

        self.screen.fill(self.white)
        # Update
        self.update_input_nodes(matrix.Matrix.to_array(input_))
        self.update_hidden_1_nodes(matrix.Matrix.to_array(hidden_1))
        self.update_hidden_2_nodes(matrix.Matrix.to_array(hidden_2))
        self.update_output_nodes(matrix.Matrix.to_array(output))
        self.update_ih1_weights(matrix.Matrix.to_array(__weights_ih1))
        self.update_h1h2_weights(matrix.Matrix.to_array(__weights_h1h2))
        self.update_h2o_weights(matrix.Matrix.to_array(__weights_h2o))
        # Draw
        self.draw_ih1_weights(self.screen, self.blue, self.red)
        self.draw_h1h2_weights(self.screen, self.blue, self.red)
        self.draw_h2o_weights(self.screen, self.blue, self.red)
        self.draw_input_nodes(self.screen, self.black, self.red)
        self.draw_hidden_1_nodes(self.screen, self.black, self.red)
        self.draw_hidden_2_nodes(self.screen, self.black, self.red)
        self.draw_output_nodes(self.screen, self.black, self.red)
        pg.display.update()

class Node:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.data = 0.0

class Weight:
    def __init__(self, coor1, coor2):
        self.coor1 = coor1
        self.coor2 = coor2
        self.data = 0.0