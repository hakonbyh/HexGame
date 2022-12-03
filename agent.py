from collections import deque
from copy import deepcopy
import random
import os
import numpy as np
from anet import ActorNN
from mcts import MCTS
from statemanager import DiamondBoard
from visualizer import DiamondDisplay


class Agent:
    """Class representing the agent responsible for coordinating
    the training and all the components together"""

    def __init__(self, alpha, hidden_layer_sizes, board_size, board_shape, epsilon, search_games, optimizer, activation, c=1, frame_delay=500):
        self.search_games = search_games
        self.board_shape = board_shape
        self.epsilon = epsilon
        self.c = c
        self.board_size = board_size

        self.actor = ActorNN(alpha, hidden_layer_sizes,
                             board_size, optimizer, activation)
        self.mcts = MCTS(self.c, self.actor)
        self.display = DiamondDisplay(board_size, frame_delay=frame_delay)
        self.board = DiamondBoard(board_size)

    def train(self, no_episodes, epochs=10, M=0, episodes_to_display=[], rbuf_size=400, batch_size=100, filename=None):
        """Main method for performing all the training"""

        if M > 0:
            interval = no_episodes // M
            filepath = 'saved_model/'
            os.makedirs(f'{filepath}{filename}', exist_ok=True)

        rbuf = deque(maxlen=rbuf_size)

        self.actor.initialize_network()

        animation = []

        start_player = 2
        score = 0  # keeps track of the score for player 1

        for i in range(no_episodes):

            print(f"Episode: {i}")

            epsilon_decay = 1-(i/no_episodes)

            if start_player == 2:
                start_player = 1
            elif start_player == 1:
                start_player = 2

            self.board.set_initial_state(start_player)

            root = self.board

            self.mcts.initialize()

            if i in episodes_to_display:
                # saves the state used for animation
                animation.append([deepcopy(self.board.get_state())])

            while not self.board.is_final_state():

                self.mcts.set_root(root)

                tree = self.mcts.tree_search(
                    self.search_games, self.epsilon, epsilon_decay)

                D = self.calculate_D(tree, root)

                rbuf.append((root.get_pid_state(), D))

                all_actions = np.array([(row, col) for row in range(self.board_size)
                                        for col in range(self.board_size)], dtype=tuple)

                # chooses the best action based on all the simulations done
                action = all_actions[np.argmax(D)]

                self.board.update_state(action)

                root = self.board

                if i in episodes_to_display:
                    animation[-1].append(deepcopy(self.board.get_state()))

            if self.board.get_winner() == 1:
                score += 1
            print(f'Won {score}/{i+1} games\n')

            mbs = random.randint(1, min(len(rbuf), batch_size))
            # generates a batch of a random number of random samples used to train the model
            batch = random.choices(rbuf, k=mbs)

            self.actor.train(np.array([elem[0] for elem in batch]),
                             np.array([elem[1] for elem in batch]), mbs, epochs)

            if M > 0 and (i+1) % interval == 0 or (M > 0 and i == 0):
                self.actor.model.save(
                    f'{filepath}{filename}/{filename}_{str(i+1)}')
                print("Saved model")

        for episode in animation:
            self.display.animate_episode(episode)

    def calculate_D(self, tree, root):
        """Returns an normalized array containing the
        node values for all children boards from the root"""
        array = np.zeros(self.board_size**2)
        node = tree[str(root.get_pid_state())]
        actions = root.get_possible_actions()
        for row in range(self.board_size):
            for col in range(self.board_size):
                if (row, col) in actions:
                    array[row*self.board_size+col] = node.get_N_v((row, col))

        D = [i/sum(array) for i in array]

        return D
