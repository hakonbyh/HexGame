import random
import numpy as np
from tensorflow import keras
import tensorflow as tf


class ActorNN:
    """Class representing the actor"""

    def __init__(self, alpha, hidden_layer_sizes, board_size, optimizer, activation):
        self.optimizer = optimizer
        self.activation = activation
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model = None
        self.input_dim = board_size**2 + 1
        self.output_dim = board_size**2

    def initialize_network(self):
        """Initializes a neural net with its layers"""
        self.model = keras.Sequential()

        self.model.add(keras.Input(shape=(self.input_dim, )))

        for dim in self.hidden_layer_sizes:
            self.model.add(keras.layers.Dense(dim, activation=self.activation))

        self.model.add(keras.layers.Dense(
            self.output_dim, activation='softmax'))

        cce = keras.losses.CategoricalCrossentropy()

        self.model.compile(optimizer=self.optimizer(
            learning_rate=self.alpha), loss=keras.losses.KLDivergence(), metrics=[tf.keras.metrics.KLDivergence()])

        self.model.summary()

    def select_action(self, board, epsilon):
        """Returns the most promising action based on the predictions of the model"""
        actions = board.get_possible_actions()
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            pid_state = board.get_pid_state()
            action_space = board.get_action_space()
            predictions = self.model(pid_state.reshape((1, len(pid_state))))[0]
            possible_actions = np.multiply(predictions, action_space)

            normalized_actions = possible_actions / np.sum(possible_actions)

            action_index = np.argmax(normalized_actions)
            action = (action_index // board.get_size(),
                      action_index % board.get_size())

        return action

    def train(self, input_values, target_values, minibatch_size, epochs):
        """Takes in data from the RBUF data set and trains the model"""
        self.model.fit(input_values, target_values, minibatch_size, epochs, verbose=0)


class ActorTOPP:
    """Actor class used to play and do actions in the topp class"""
    def __init__(self, model):
        self.model = model
        self.won_games = 0
        self.lost_games = 0

    def won_game(self):
        """Used to keep count of won games of the model in the topp class"""
        self.won_games += 1

    def lost_game(self):
        """Used to keep count of lost games of the model in the topp class"""
        self.lost_games += 1

    def get_result(self):
        """Returns the result of the model used in the topp class"""
        return (self.won_games, self.lost_games)

    def select_action(self, board):
        """Returns the most promising action based on the predictions of the model"""
        if random.uniform(0, 1) < 0.1:
            actions = board.get_possible_actions()
            action = random.choice(actions)
            return action

        pid_state = board.get_pid_state()
        action_space = board.get_action_space()
        predictions = self.model(pid_state.reshape((1, len(pid_state))))[0]
        possible_actions = np.multiply(predictions, action_space)

        normalized_actions = possible_actions / np.sum(possible_actions)

        action_index = np.argmax(normalized_actions)
        action = (action_index // board.get_size(),
                  action_index % board.get_size())

        return action
