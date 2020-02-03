"""
This is the high level file that will wrap together all of
the game sub components and allow for selection among those

Below I give an example of default commenting style
  and class structure in python

Implementation note: Hovering over game before selection
  should play sample music for that game...?
"""


class Arcade:
    def __init__(self):
        pass

    def play_game(self, game_id="pong"):
        """
        Arcade game loop that runs a specified game, terminating
          when termination signal is received by lower level game object.
        :param game_id: (str) -> string ID of game currently being played
        :return: (dict) game meta-data, such as high-score information, etc to hold on to
        """
        pass



