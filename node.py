class Node:
    """Class for representing a node in the mcts tree"""
    def __init__(self, pid_state, actions):

        self.pid_state = pid_state
        self.action = None
        self.Q = {}
        self.N = 0
        self.N_v = {}

        for action in actions:
            self.Q[action] = 0
            self.N_v[action] = 0

    def update_Q(self, action, z):
        """Updates the Q value"""
        self.Q[action] += (z - self.Q[action]) / self.N_v[action]

    def get_Q(self, action):
        """Returns the Q value"""
        return self.Q[action]

    def increment_N(self):
        """Increments N by one"""
        self.N += 1

    def get_N(self):
        """Returns the node count"""
        return self.N

    def increment_N_v(self, action):
        """Increments the node value by one"""
        self.N_v[action] += 1

    def get_N_v(self, action):
        """Returns the node value"""
        return self.N_v[action]

    def get_action(self):
        """Returns the action used to go to this node"""
        return self.action

    def set_action(self, action):
        """Sets the action that was used to go to this node"""
        self.action = action

    def __repr__(self):
        return str(self.pid_state)
