from math import radians
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
import numpy as np
from numpy import sin, cos


class Display(ABC):

    """Abstract class to represent the visualizing of the animation"""

    def __init__(self, board_size, figsize=(6, 6), frame_delay=500):
        self.frame_delay = frame_delay
        self.figsize = figsize
        self.board_size = board_size
        self.graph = None

    def animate_episode(self, states, players=None):
        """Main method used to animate the graph"""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.clear()

        nodepos = self.node_pos()

        state = states[0]
        neighbour_matrix = self.transform()
        self.graph = nx.from_numpy_array(neighbour_matrix)
        node_colors, outline_colors = self.color_mapping(state)

        nx.draw(self.graph, node_color=node_colors,
                edgecolors=outline_colors, pos=nodepos)

        anim = animation.FuncAnimation(fig, self.update, frames=(range(1, len(states))), fargs=(
            ax, self, nodepos, states, players), interval=self.frame_delay, repeat=False)

        plt.show()

    def update(self, i, ax, obj, nodepos, states, players):
        """Method used to update the animation for every action performed"""
        ax.clear()
        if players:
            plt.title(
                f'Player 1: {players[0]} (red)\nPlayer 2: {players[1]}(black)')
        state = states[i]
        node_colors, outline_colors = obj.color_mapping(state)
        nx.draw(obj.graph, node_color=node_colors,
                edgecolors=outline_colors, pos=nodepos)

    @abstractmethod
    def transform(self):
        """Returns an adjacency matrix used to represent the
        drawing of the graph with all the node and its edges"""

    @abstractmethod
    def color_mapping(self, state):
        """"Returns a tuple containing an array of the color of each node
        and an array containing the outline color of each node"""

    @abstractmethod
    def node_pos(self):
        """Returns an array containing the position of
        all the nodes that are to be used for the animation"""


class DiamondDisplay(Display):

    """Class for represening the visualizing of the diamond board"""

    def transform(self):
        """Returns an adjacency matrix used to represent the
        drawing of the graph with all the node and its edges"""
        num_of_nodes = self.board_size ** 2
        adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes), int)

        for i in range(num_of_nodes):
            if (i+1) < num_of_nodes and ((i+1) % self.board_size) != 0:
                adjacency_matrix[i, i+1] = 1
                adjacency_matrix[i+1, i] = 1

            if (i+self.board_size) < num_of_nodes:
                adjacency_matrix[i, i+self.board_size] = 1
                adjacency_matrix[i+self.board_size, i] = 1

            if (i+self.board_size-1) < num_of_nodes and i % self.board_size != 0:
                adjacency_matrix[i, i+self.board_size-1] = 1
                adjacency_matrix[i+self.board_size-1, i] = 1

        return adjacency_matrix

    def color_mapping(self, state):
        node_colors = []
        outline_colors = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if state[row][col] == 1:
                    node_colors.append("#EB4025")
                    outline_colors.append("None")
                elif state[row][col] == 2:
                    node_colors.append("#000000")
                    outline_colors.append("None")
                elif state[row][col] == 0:
                    node_colors.append("#FFFFFF")
                    outline_colors.append("#3E8C27")
        return (node_colors, outline_colors)

    def node_pos(self):
        nodepos = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                x, y = (row, col)
                updated_x, updated_y = rotate(135, x, y)
                nodepos.append((updated_x, updated_y))

        return nodepos


class TriangularDisplay(Display):

    """Class to represent a triangular board animation"""

    def __init__(self, board_size, figsize=(6, 6), frame_delay=500):
        x, _ = figsize
        ratio = sin(radians(60))
        scaled_figsize = (x/ratio, x)
        super().__init__(board_size, scaled_figsize, frame_delay)

    def transform(self):
        num_of_nodes = self.board_size * (self.board_size + 1) // 2

        adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes), int)

        k = 0
        m = 0
        for i in range(1, self.board_size+1):
            for _ in range(i):
                if k+i < num_of_nodes:
                    adjacency_matrix[k, k+i] = 1
                    adjacency_matrix[k+i, k] = 1

                if k+i+1 < num_of_nodes:
                    adjacency_matrix[k, k+i+1] = 1
                    adjacency_matrix[k+i+1, k] = 1

                k += 1

            for _ in range(i-1):
                adjacency_matrix[m, m+1] = 1
                adjacency_matrix[m+1, m] = 1
                m += 1
            m += 1

        return adjacency_matrix

    def color_mapping(self, state):
        node_colors = []
        outline_colors = []
        for row in range(self.board_size):
            for col in range(row+1):
                if state[row][col] == 1:
                    node_colors.append("#EB4025")
                    outline_colors.append("None")
                elif state[row][col] == 2:
                    node_colors.append("#000000")
                    outline_colors.append("None")
                elif state[row][col] == 0:
                    node_colors.append("#FFFFFF")
                    outline_colors.append("#3E8C27")
        return (node_colors, outline_colors)

    def node_pos(self):
        nodepos = []
        for row in range(self.board_size):
            for col in range(row+1):
                x, y = (col, row)
                updated_x, updated_y = rotate(180, x, y)
                updated_x = updated_x - (self.board_size - row) * 0.5
                nodepos.append((updated_x, updated_y))

        return nodepos


def rotate(degree, x, y):
    """Used to rotate the graph used for animation by a given degree"""
    degree = -degree
    new_x = cos(radians(degree))*x - sin(radians(degree))*y
    new_y = sin(radians(degree))*x + cos(radians(degree))*y

    return new_x, new_y
