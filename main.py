from tensorflow import keras
from agent import Agent
from topp import TOPP


board_shape = 0
available_optimizers = {"adagrad": keras.optimizers.Adagrad,
                        "sgd": keras.optimizers.SGD,
                        "rmsprop": keras.optimizers.RMSprop,
                        "adam": keras.optimizers.Adam}


if __name__ == "__main__":

    def test1():
        """test 1"""
        # board parameters
        board_size = 4

        # mcts parameters
        no_of_episodes = 300
        search_games = 700
        episodes_to_display = [1, 2, 3]

        # anet parameters
        alpha = 0.01
        hidden_layer_sizes = [128, 128]

        activation = "tanh"
        optimizer = available_optimizers["sgd"]

        epochs = 8
        rbuf_size = 1000
        epsilon = 0.9
        batch_size = 32

        # topp parameters
        M = 10
        G = 10

        agent = Agent(alpha, hidden_layer_sizes, board_size,
                      board_shape, epsilon, search_games, optimizer, activation)
        agent.train(no_of_episodes, epochs, M, episodes_to_display,
                    rbuf_size=rbuf_size, batch_size=batch_size, filename="demo1")

        topp = TOPP(board_size, filename="demo")
        topp.play_tournament(G)

    def test2():
        """test 2"""
        # board parameters
        board_size = 4

        # mcts parameters
        no_of_episodes = 20
        search_games = 700
        episodes_to_display = [1, 2, 3]

        # anet parameters
        alpha = 0.03
        hidden_layer_sizes = [128, 128]

        activation = "tanh"
        optimizer = available_optimizers["sgd"]

        epochs = 8
        rbuf_size = 1000
        epsilon = 0.9
        batch_size = 999

        # topp parameters
        M = 4
        G = 100

        agent = Agent(alpha, hidden_layer_sizes, board_size,
                      board_shape, epsilon, search_games, optimizer, activation)
        agent.train(no_of_episodes, epochs, M, episodes_to_display,
                    rbuf_size=rbuf_size, batch_size=batch_size, filename="demo5")

        topp = TOPP(board_size, filename="demo5")
        topp.play_tournament(G)

    def test3():
        """test 3"""
        # board parameters
        board_size = 4

        # mcts parameters
        no_of_episodes = 20
        search_games = 500
        episodes_to_display = []

        # anet parameters
        alpha = 0.03
        hidden_layer_sizes = [128, 128, 128]

        activation = "tanh"
        optimizer = available_optimizers["adam"]

        epochs = 10
        rbuf_size = 1000
        epsilon = 0.9
        batch_size = 1000

        # topp parameters
        M = 4
        G = 100

        """agent = Agent(alpha, hidden_layer_sizes, board_size,
                      board_shape, epsilon, search_games, optimizer, activation)
        agent.train(no_of_episodes, epochs, M, episodes_to_display,
                    rbuf_size=rbuf_size, batch_size=batch_size, filename="new_test")"""

        topp = TOPP(board_size, filename="demo10")
        topp.play_tournament(100, agents_to_display=[])

    # test1()
    # test2()
    test3()
