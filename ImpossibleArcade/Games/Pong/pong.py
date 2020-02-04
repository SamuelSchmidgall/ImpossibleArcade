import numpy as np
from ImpossibleArcade.Games.game import Game

class Pong(Game):
    def __init__(self):
        """
        Initialization of Pong game, a subclass of Game
        """
        super(Game, self).__init__(game_id="pong", screen_dimension=(150, 50))
        # initialize empty ball velocity vector
        self._ball_velocity = (0, 0)
        # paddle metadata dictionary
        self._paddle_data = dict()
        # initialize empty score tuple
        self.score = {"AI":0, "player":0}
        # initialize empty pong grid
        self._pong_grid = self._initialize_board()
        # perform a full machine reset to correctly
        # initialize various parameters
        self.machine_reset()

    def _initialize_board(self):
        """
        Initialize and return an empty board
        :return: (list(list(str))) -> initial board state information
        """
        # initialize paddle data
        self._paddle_data["p_depth"] = 5
        self._paddle_data["p_width"] = 2
        # initialize paddle 1 position
        self._paddle_data["p1_center"] = 75
        # initialize paddle 2 position
        self._paddle_data["p2_center"] = 75
        # generate board containing empty cells
        board = [["E" for _j in range(150)] for _i in range(50)]

        # fill board with paddle cells -- not necessary
        # computationally, but drastically simplifies graphics
        for _i in range(len(self._paddle_data["p_depth"])):
            for _j in range(len(self._paddle_data["p_width"])):
                # compute paddle 1 cell x and y, fill board location with paddle cell
                p1_cell_x = self._paddle_data["p_depth"][_i] + self._paddle_data["p1_center"]
                p1_cell_y = self._paddle_data["p_width"][_j]
                board[p1_cell_x][p1_cell_y] = "P"

                # compute paddle 2 cell x and y, fill board location with paddle cell
                p2_cell_x = self._paddle_data["p_depth"][_i] + self._paddle_data["p2_center"]
                p2_cell_y = self._paddle_data["p_width"][_j]
                board[p2_cell_x][p2_cell_y] = "P"

        return board

    def _reset_ball_velocity(self):
        """
        Randomly initialize ball velocity vector
        :return: (tuple(int)) -> ball velocity vector
        """
        x_vel = np.random.choice([-1, 1])
        y_vel = np.random.choice([-1, 1])
        return x_vel, y_vel

    def update(self, action):
        """
        Update pong based on discretized binary action
        :param action: (ndarray) -> numpy array referencing player action
          more explicit information contained in this and sub-functions
        :return: (bool) -> termination information
        """
        pass

    def render(self, screen):
        """
        Render game information to display pong screen
        :param screen: (pygame.screen) -> pygame screen to render on to
        :return: None
        """
        pass

    def game_reset(self):
        """
        Reset the current game, but not entire system
          Here this refers to resetting the board state if point is scored
        :return: None
        """
        self._ball_velocity = self._reset_ball_velocity()
        self._pong_grid = self._initialize_board()

    def machine_reset(self):
        """
        Reset the game conditions to original state
        :return: None
        """
        self.game_reset()
        self.score = {"AI": 0, "player": 0}










