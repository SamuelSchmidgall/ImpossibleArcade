import numpy as np
from ImpossibleArcade.Games.game import Game

class Pong(Game):
    def __init__(self):
        """
        Initialization of Pong game, a subclass of Game
        """
        super(Game, self).__init__(game_id="pong", screen_dimension=(300, 100))
        # initialize empty score tuple
        self.score = None
        # initialize empty pong grid
        self._pong_grid = None
        # initialize empty ball velocity vector
        self._ball_velocity = None
        # initialize empty ball position vector
        self._ball_position = None
        # designated RGB color scheme for grid types
        self.colors = {
            "E": (0, 0, 0), "P": (0, 0, 0), "B": (0, 0, 0)}

        # paddle metadata dictionary
        self._paddle_data = dict()
        # generate the initial empty board for reference
        self._empty_board = [["E" for _j in range(
            self.screen_dimension[0])] for _i in range(self.screen_dimension[1])]
        # perform a full machine reset to correctly
        # initialize various parameters including paddle positions
        self.machine_reset()

    def update(self, action):
        """
        Update pong based on discretized binary action
        :param action: (ndarray) -> numpy array referencing player action
          more explicit information contained in this and sub-functions
        :return: (bool) -> termination information
        """
        # update user paddle position
        self._update_user_interaction(action)

        # update AI paddle position
        self._update_AI()

        # update ball position
        termination = self._update_ball()

        return termination

    def render(self, screen):
        """
        Render game information to display pong screen
        :param screen: (pygame.screen) -> pygame screen to render on to
        :return: None
        """
        # reset board to empty grid
        self._pong_grid = self._empty_board
        # reposition paddle 1
        self._reposition_paddle(1)
        # reposition paddle 2
        self._reposition_paddle(2)
        # reposition ball
        self._reposition_ball()
        # render grid on to screen
        self._draw_grid(screen, self._pong_grid)

    def game_reset(self):
        """
        Reset the current game, but not entire system
          Here this refers to resetting the board state if point is scored
        :return: None
        """
        # reset ball velocity to random vector
        self._ball_velocity, self._ball_position = self._reset_ball()
        # reset paddle and ball positions
        self._initialize_board()

    def machine_reset(self):
        """
        Reset the game conditions to original state
        :return: None
        """
        self.game_reset()
        self.score = {"AI": 0, "player": 0}

        # initialize default paddle information
        self._paddle_data["paddle_velocity"] = 3
        self._paddle_data["paddle_actions"] = {0: "up", 1: "down", 2: "stay"}

    def _initialize_board(self):
        """
        Initialize or reset board positions
        :return: None
        """
        # initialize paddle data
        self._paddle_data["p_depth"] = 10
        self._paddle_data["p_width"] = 4

        # initialize paddle 1 position
        self._paddle_data["p1_center"] = 150
        # initialize paddle 2 position
        self._paddle_data["p2_center"] = 150

    def _reset_ball(self):
        """
        Randomly initialize ball velocity vector
        :return: (tuple(int)) -> ball velocity vector
        """
        x_vel = np.random.choice([-1, 1])
        y_vel = np.random.choice([-2, 2])

        return (x_vel, y_vel), (150, 50)

    def _update_AI(self):
        """
        Return an action command from corresponding game AI
        :return: (ndarray) -> velocity command given from game AI
        """
        pass

    def _update_ball(self):
        """
        Update the ball position based on its velocity
        :return: (bool) -> whether game reached termination state
        """
        # updated ball position base don velocity
        pos = self._ball_position[0] + self._ball_velocity[0],\
              self._ball_position[1] + self._ball_velocity[1]

        # check paddle collision
        # ball hit paddle 1
        if pos[0] >= 300 - self._paddle_data["p_width"] \
            and abs(pos[0] - self._paddle_data["p1_center"]) < self._paddle_data["p_height"]:
            self._ball_velocity[0] = -2
            self._ball_position[0] = 300 - self._paddle_data["p_width"] - 1
        # ball went past paddle 1, p2 score
        elif pos[0] >= 300 - self._paddle_data["p_width"]:
            self.score["Player"] += 1
            return True
        # ball hit paddle 2
        elif pos[0] <= self._paddle_data["p_width"] \
            and abs(pos[0] - self._paddle_data["p2_center"]) < self._paddle_data["p_height"]:
            self._ball_velocity[0] = 2
            self._ball_position[0] = self._paddle_data["p_width"] + 1
        # ball went past paddle 2, p1 score
        elif pos[0] <= self._paddle_data["p_width"]:
            self.score["AI"] += 1
            return True

        # ball hits bottom
        if pos[1] >= 150:
            self._ball_velocity[1] = -1
            self._ball_position[1] = 149
        # ball hits top
        elif pos[1] < 0:
            self._ball_velocity[1] = 1
            self._ball_position[1] = 0

        return False

    def _update_user_interaction(self, action):
        """
        Update paddle data based on user interaction
        :param action: (ndarray) -> numpy array referencing player action
          here, the actions 0, 1, 2 refer to paddle velocity commands
        :return: None
        """
        # paddle up action
        if action[0] == 0:
            self._paddle_data["p1_center"] = min(max(
                self._paddle_data["p1_center"] + self._paddle_data["paddle_velocity"], 5), 145)

        # paddle down action
        elif action[0] == 1:
            self._paddle_data["p1_center"] = min(max(
                self._paddle_data["p1_center"] - self._paddle_data["paddle_velocity"], 5), 145)

        # paddle stay action
        elif action[0] == 2:
            pass  # do nothing

    def _reposition_paddle(self, paddle_id=1):
        """
        Reposition paddle on grid for visualization purposes
        :param paddle_id: (int) paddle identification number
        :return: None
        """
        for _i in range(self._paddle_data["p_depth"]):
            for _j in range(self._paddle_data["p_width"]):
                # compute paddle 1 cell x and y, fill board location with paddle cell
                p1_cell_x = self._paddle_data["p_depth"] + _i + self._paddle_data["p{}_center".format(paddle_id)]
                p1_cell_y = self._paddle_data["p_width"] + _j
                self._pong_grid[p1_cell_x][p1_cell_y] = "P"

    def _reposition_ball(self):
        """
        Reposition ball on to pong grid
        :return: None
        """
        ball_pos = self._ball_position
        self._pong_grid[ball_pos[0]][ball_pos[1]] = "B"

    def _draw_grid(self, screen, board):
        """
        Render game board on to pygame screen
        :param screen: (pygame.screen) -> screen to render on to
        :param board: (list(list(str))) -> list of board values
        :return: None
        """
        pass








