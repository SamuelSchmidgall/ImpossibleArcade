import pygame
from ImpossibleArcade.Games.Pong.pong import Pong

pong = Pong()
width, height = pong.screen_dimension
screen = pygame.display.set_mode((width, height))
while True:
    pong.update([1])
    pong.render(screen)
    pygame.display.flip()

#print()

















