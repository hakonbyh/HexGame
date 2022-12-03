"""module from regex used to sort the files in ascending order"""
import re

import os
from copy import deepcopy
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from anet import ActorTOPP
from statemanager import DiamondBoard
from visualizer import DiamondDisplay


class TOPP:
    """Class represening the tournament between the M agents"""
    def __init__(self, board_size, filename):
        self.board_size = board_size
        self.display = DiamondDisplay(self.board_size)
        self.actors = []

        directory = f'saved_model/{filename}'
        folders = os.listdir(directory)
        folders.sort(key=lambda f: int(re.sub('\D', '', f)))
        for path in folders:
            full_path = os.path.join(directory, path)
            if os.path.isdir(full_path):
                model = keras.models.load_model(full_path)
                self.actors.append((ActorTOPP(model), path))

        self.M = len(self.actors)

        self.board = DiamondBoard(board_size)

    def play_tournament(self, G, agents_to_display=[]):
        """Plays a round robin tournament where all players
        plays against each other a G number of times"""
        q = 0
        for i, actor in enumerate(self.actors):
            for j, opponent in enumerate(self.actors[i+1:], start=1):

                start_player = 1
                q += 1
                print(f"Starting series {q}/{int(self.M*(self.M-1)/2)}\n")

                for k in range(G):
                    visualize = False
                    animation = []

                    self.board.set_initial_state(start_player)

                    if (i, j) in set(map(tuple, map(sorted, agents_to_display))) and k == G // 2:
                        animation.append([deepcopy(self.board.get_state())])
                        visualize = True


                    while not self.board.is_final_state():
                        if self.board.get_player() == 1:
                            action = actor[0].select_action(self.board)
                        else:
                            action = opponent[0].select_action(self.board)

                        self.board.update_state(action)

                        if visualize:
                            animation[-1].append(deepcopy(self.board.get_state()))

                    if self.board.get_winner() == 1:
                        actor[0].won_game()
                        opponent[0].lost_game()
                    else:
                        opponent[0].won_game()
                        actor[0].lost_game()

                    if visualize:
                        for episode in animation:
                            self.display.animate_episode(
                                episode, (actor[1], opponent[1]))

                    if start_player == 1:
                        start_player = 2
                    else:
                        start_player = 1

        results = [actor[0].get_result() for actor in self.actors]

        max_win = max(results, key=lambda x: x[0])
        winner = results.index(max_win)

        print(str(winner)+' won the most games.')

        results = [100*result[0]/(result[0]+result[1]) for result in results]

        bars = tuple(actor[1] for actor in self.actors)
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, results, color=(0.2, 0.4, 0.6, 0.6))

        plt.xticks(y_pos, bars)
        plt.show()
