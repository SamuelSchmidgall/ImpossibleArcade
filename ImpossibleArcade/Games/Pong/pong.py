import pygame
import numpy as np
from copy import deepcopy
from ImpossibleArcade.Games.game import Game

class Pong(Game):
    def __init__(self):
        """
        Initialization of Pong game, a subclass of Game
        """
        super().__init__(game_id="pong", screen_dimension=(300, 100))
        # initialize empty score tuple
        self.score = None
        # initialize pygame fonts
        pygame.font.init()
        # initialize empty pong grid
        self._pong_grid = None
        # initialize empty ball velocity vector
        self._ball_velocity = None
        # initialize empty ball position vector
        self._ball_position = None
        # designated RGB color scheme for grid types
        self.colors = {
            "E": (0, 0, 0), "P": (255, 255, 255), "B": (255, 255, 255)}

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
        self._pong_grid \
            = deepcopy(self._empty_board)

        # reposition paddle 1
        self._reposition_paddle(1)
        # reposition paddle 2
        self._reposition_paddle(2)
        # reposition ball
        self._reposition_ball()

        # render grid on to screen
        self._draw_grid(screen, self._pong_grid)

        # render score on to board
        self._draw_score(screen)

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
        # perform game and score reset
        self.game_reset()
        self.score = {"AI": 0, "Player": 0}

        # initialize default paddle information
        self._paddle_data["paddle_velocity"] = 3
        self._paddle_data["paddle_actions"] = {0: "up", 1: "down", 2: "stay"}

    def train_policy(self, soft_actor_critic):
        """
        Train a reinforcement policy using self-play
        :param soft_actor_critic: (SoftActorCritic) ->
          Soft Actor Critic architecture to train policy on
        :return: (SoftActorCritic) -> trained policy
        """
        pass

    def _initialize_board(self):
        """
        Initialize or reset board positions
        :return: None
        """
        # initialize paddle data
        self._paddle_data["p_depth"] = 1
        self._paddle_data["p_width"] = 10

        # initialize paddle 1 position
        self._paddle_data["p1_center"] = 50
        # initialize paddle 2 position
        self._paddle_data["p2_center"] = 50

    def _reset_ball(self):
        """
        Randomly initialize ball velocity vector
        :return: (tuple(int)) -> ball velocity vector
        """
        x_vel = np.random.choice([-1, 1])
        y_vel = np.random.choice([-2, 2])

        return (x_vel, y_vel), (50, 150)

    def _update_AI(self):
        """
        Return an action command from corresponding game AI
        :return: (ndarray) -> velocity command given from game AI
        """
        action = [1]

        # paddle up action
        if action[0] == 0:
            self._paddle_data["p2_center"] = \
                min(max(self._paddle_data["p2_center"]
                + self._paddle_data["paddle_velocity"], 6), 93)

        # paddle down action
        elif action[0] == 1:
            self._paddle_data["p2_center"] =\
                min(max(self._paddle_data["p2_center"]
                - self._paddle_data["paddle_velocity"], 6), 93)

        # paddle stay action
        elif action[0] == 2:
            pass  # do nothing

    def _update_ball(self):
        """
        Update the ball position based on its velocity
        :return: (bool) -> whether game reached termination state
        """
        # updated ball position based on velocity
        self._ball_position = (
            self._ball_position[0] + self._ball_velocity[0],
            self._ball_position[1] + self._ball_velocity[1])

        pos = self._ball_position

        # check paddle collision
        # ball hit paddle 1
        if pos[1] >= 300 - 6 and abs(pos[0] - self._paddle_data["p2_center"]) <= 6:
            self._ball_velocity = (self._ball_velocity[0], -2)
            self._ball_position = (self._ball_position[0]-1, self._ball_position[1])
            #self._ball_position = (300 - self._paddle_data["p_width"] - 1, self._ball_position[1])
        # ball went past paddle 1, p2 score
        elif pos[1] >= 300 - 3:
            self.score["AI"] += 1
            return True
        # ball hit paddle 2
        elif pos[1] <= 6 and abs(pos[0] - self._paddle_data["p1_center"]) <= 6:
            self._ball_velocity = (self._ball_velocity[0], 2)
            self._ball_position = (self._ball_position[0]+1, self._ball_position[1])
        # ball went past paddle 2, p1 score
        elif pos[1] <= 3:
            self.score["Player"] += 1
            return True

        # ball hits bottom
        if pos[0] >= 98:
            self._ball_velocity = (-1, self._ball_velocity[1])
            self._ball_position = (97, self._ball_position[1])
        # ball hits top
        elif pos[0] <= 1:
            self._ball_velocity = (1, self._ball_velocity[1])
            self._ball_position = (2, self._ball_position[1])

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
            self._paddle_data["p1_center"] = \
                min(max(self._paddle_data["p1_center"] +
                self._paddle_data["paddle_velocity"], 6), 93)

        # paddle down action
        elif action[0] == 1:
            self._paddle_data["p1_center"] = \
                min(max(self._paddle_data["p1_center"] -
                self._paddle_data["paddle_velocity"], 6), 93)

        # paddle stay action
        elif action[0] == 2:
            pass  # do nothing

    def _reposition_paddle(self, paddle_id=1):
        """
        Reposition paddle on grid for visualization purposes
        :param paddle_id: (int) paddle identification number
        :return: None
        """
        # iterate through depth and width to update color grid
        for _i in range(self._paddle_data["p_depth"]):
            for _j in range(self._paddle_data["p_width"]//2):
                # compute paddle 1 cell x and y, fill board location with paddle cell
                p_cell_x = _j + self._paddle_data["p{}_center".format(paddle_id)]
                p_cell_y = _i + 5 if paddle_id == 1 else 294
                self._pong_grid[p_cell_x][p_cell_y] = "P"
            for _j in range(self._paddle_data["p_width"]//2):
                # compute paddle 1 cell x and y, fill board location with paddle cell
                p_cell_x = -1*_j + \
                    self._paddle_data["p{}_center".format(paddle_id)]
                p_cell_y = _i + 5 if paddle_id == 1 else 294
                self._pong_grid[p_cell_x][p_cell_y] = "P"

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
        #import time
        #time.sleep(1)
        # fill screen black
        screen.fill((0, 0, 0))
        for _i in range(len(board)):
            for _j in range(len(board[_i])):
                # integrate color for cell type
                _grid_color = self.colors[board[_i][_j]]
                # compute top left corner
                _tl_corner = _i*self.grid_size, _j*self.grid_size
                # render rectangle
                pygame.draw.rect(screen, _grid_color,
                    [_tl_corner[1], _tl_corner[0], self.grid_size, self.grid_size])

    def _draw_score(self, screen):
        text = "{} | {}".format(self.score["Player"], self.score["AI"])
        font = pygame.font.Font(pygame.font.get_default_font(), 30)
        text = font.render(text, True, (255, 255, 255))
        screen.blit(text, (750, 30))






