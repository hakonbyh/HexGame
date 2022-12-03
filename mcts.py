from math import sqrt, log
from copy import deepcopy
import numpy as np
from node import Node


class MCTS:
    """Class representing the MCTS"""

    def __init__(self, c, actor):
        self.c = c
        self.default_policy = actor
        self.root = None
        self.tree = {}

    def initialize(self):
        """Initializes the MCT with a empty tree"""
        self.tree = {}

    def tree_search(self, search_games, initial_epsilon, epsilon_decay):
        """Used to perform simulated actions on the root board"""
        while search_games > 0:
            epsilon = initial_epsilon * epsilon_decay
            self.simulate(deepcopy(self.root), epsilon)
            search_games -= 1

        return self.tree

    def simulate(self, board, epsilon):
        """Performs a simulation and updates the values
        of each node backwards from the leaf node to the root"""
        path = self.sim_tree(board)
        if path:
            z = self.sim_default(deepcopy(board), epsilon)
            self.backup(path, z)

    def sim_tree(self, board):
        """Method for performing a simulation of actions down the tree"""
        path = []
        while not board.is_final_state():
            if str(board.get_pid_state()) not in self.tree:
                node = Node(board.get_pid_state(),
                            board.get_possible_actions())
                self.tree[str(board.get_pid_state())] = node
                return path
            node = self.tree[str(board.get_pid_state())]
            path.append(node)
            action = self.select_move(board, node)
            node.set_action(action)
            board.update_state(action)
        return path

    def sim_default(self, board, epsilon):
        """Performing a rollout on the given board until the
        game ends. Returns 1 or -1 based on which player wins"""
        while not board.is_final_state():
            action = self.default_policy.select_action(board, epsilon)
            board.update_state(action)
        return 1 if board.get_winner() == 1 else -1

    def set_root(self, board):
        """Updates the root of the mct to the the new board given as parameter"""
        self.root = board
        board_copy = deepcopy(board)
        self.tree[str(board_copy.get_pid_state())] = Node(
            board_copy.get_pid_state(), board_copy.get_possible_actions())

    def select_move(self, board, node):
        """Returns the most valuable action based on the node values"""
        actions = board.get_possible_actions()
        values = np.zeros(len(actions))

        for i, action in enumerate(actions):
            if board.get_player() == 1:
                value = node.get_Q(action) + self.c * sqrt(log(node.get_N()
                                                               if node.get_N() > 0 else 1) / (node.get_N_v(action)+1))
            elif board.get_player() == 2:
                value = node.get_Q(action) - self.c * sqrt(log(node.get_N()
                                                               if node.get_N() > 0 else 1) / (node.get_N_v(action)+1))

            values[i] = value

        if board.get_player() == 1:
            return actions[np.argmax(values)]
        elif board.get_player() == 2:
            return actions[np.argmin(values)]

    def backup(self, path, z):
        """Propagates the tree backwards adjusting all
        the values from the leaf node to the root node"""
        for node in path:
            action = node.get_action()
            node.increment_N()
            node.increment_N_v(action)
            node.update_Q(action, z)
