"""
This file contains high level structure for any game to be implemented,
  through the use of an abstract game class. Any game object can be expected
  to have fully developed abstract classes, as well as its full implementation
  only having to utilize such structure (i.e. no other function calls are made externally)
"""

from abc import ABC, abstractmethod


class Game(ABC):
    def __init__(self, game_id, screen_dimension):
        """
        Abstract initialization function for game class
        :param game_id: (str) -> string reference to game
        :param screen_dimension: tuple(int, int) -> (heigh, width) integer tuple,
          referencing the game screen dimensionality for graphics purposes
        """
        self.game_id = game_id  # string reference to game
        self.screen_dimension = screen_dimension  # representation-wise dimensionality of game

    @abstractmethod
    def update(self, action):
        """
        Update game based on user-input action [AI are self-contained]
        :param action: (ndarray) -> numpy array referencing player action
        :return: (bool) -> termination information
        """
        pass

    @abstractmethod
    def render(self, screen):
        """
        Render game information on to screen
        :param screen: (pygame.screen) -> pygame screen to render on to
        :return: None
        """
        pass

    @abstractmethod
    def game_reset(self):
        """
        Reset the current game, but not entire system (e.g. if new pac-man level reset board)
        :return: None
        """
        pass

    @abstractmethod
    def machine_reset(self):
        """
        Reset the game conditions to original state
        :return: None
        """
        pass





