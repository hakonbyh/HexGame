import numpy as np

class DiamondBoard:
    """Class representing the diamond board and
    inheriting from the abstract class HexBoard"""

    def __init__(self, board_size):
        self.board_size = board_size

    def set_initial_state(self, start_player=1):
        """Sets game to initial state"""
        self.player = start_player       
        self.state = np.zeros((self.board_size, self.board_size))
        self.action_space = np.ones((self.board_size, self.board_size), dtype=int)
        self.winner = None

    def is_final_state(self):
        """ Checks to see if the game is over and sets the winner"""
        player = self.evaluate_game()
        if player:
            self.winner = player
            return True
        return False

    def get_winner(self):
        """ Returns the winner"""
        return self.winner

    def reset(self):
        """Resets the game board"""
        self.set_initial_state()

    def get_state(self):
        """Returns the state of the game"""
        return self.state

    def update_state(self, action):
        """Updates the board state with the given action"""
        row, col = action
        self.action_space[row, col] = 0
        self.state[row][col] = self.player
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def get_player(self):
        """Returns the player whos turn it is to make a move"""
        return self.player

    def get_size(self):
        """Returns dimensions of the given board"""
        return self.board_size

    def is_final_state(self):
        """Checks to see if the one of the players won"""
        result = self.evaluate_game()
        if result in [1, 2]:
            self.winner = result
            return True
        else:
            return False

    def get_pid_state(self):
        """"Returns an array containing the
        player id and the board state of the game"""
        return np.concatenate(([self.player], np.array(self.state).flatten()))

    def get_action_space(self):
        """Returns a binary array showing which
        moves are possible and which are not"""
        return np.array(self.action_space).flatten()

    def get_possible_actions(self):
        """Returns all possible actions"""
        actions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.state[row, col] == 0:
                    actions.append((row, col))
        return actions

    def get_neighbours(self, position, player):
        """Returns a list of all neighbours given a position"""
        r, c = position
        neighbour_offset = [(r-1, c), (r-1, c+1), (r, c-1),
                            (r, c+1), (r+1, c-1), (r+1, c)]
        neighbours = []
        for pos in neighbour_offset:
            if 0 <= pos[0] < self.board_size and 0 <= pos[1] < self.board_size:
                if self.state[pos] == player:
                    neighbours.append(pos)

        return neighbours

    def evaluate_game(self):
        """ Checks to see if a player has won and returns
        this player, returns None if no player has won"""
        n = self.board_size
        for player in [1, 2]:
            if player == 1:
                start_edge = [(0, i)
                              for i in range(n) if self.state[0, i] == 1]
                opposite_edge = [(n-1, i)
                                 for i in range(n) if self.state[n-1, i] == 1]
            else:
                start_edge = [(i, 0)
                              for i in range(n) if self.state.T[0, i] == 2]
                opposite_edge = [(i, n-1)
                                 for i in range(n) if self.state.T[n-1, i] == 2]

            if start_edge and opposite_edge:
                for pos in start_edge:
                    prune_list = []
                    result = self.find_path(
                        pos, player, opposite_edge, prune_list)
                    if result:
                        return player

        return None

    def find_path(self, pos, player, opposite_edge, prune_list):
        """Returns true if there exists a path
        from one edge to the opposite edge"""
        if pos in opposite_edge:
            return True

        prune_list.append(pos)
        neighbours = [pos for pos in self.get_neighbours(
            pos, player) if pos not in prune_list]
        for pos in neighbours:

            result = self.find_path(pos, player, opposite_edge, prune_list)

            if result:
                return True

        return False