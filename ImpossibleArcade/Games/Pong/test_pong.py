import pygame
from ImpossibleArcade.Games.Pong.pong import Pong

pong = Pong()
width, height = pong.screen_dimension
screen = pygame.display.set_mode((width*pong.grid_size, height*pong.grid_size))
k_s, k_w = False, False
while True:
    k_s, k_w = False, False
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                k_w = True
            if event.key == pygame.K_s:
                k_s = True
    if k_s:
        act = [0]
    elif k_w:
        act = [1]
    else:
        act = [2]
    term = pong.update(act)
    if term:
        pong.game_reset()
    pong.render(screen)
    pygame.display.flip()

#print()

















