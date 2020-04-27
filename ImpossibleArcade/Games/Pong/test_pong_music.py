import pygame
from pygame.mixer import Sound, get_init, pre_init

from ImpossibleArcade.Games.Pong.pong import Pong
from ImpossibleArcade.Games.note import Note
from ImpossibleArcade.Games.note_generator import Note_Generator

pong = Pong()

game_intensity = 1
note_gen = Note_Generator()
pre_init(44100, -16, 1, 1024)
pygame.init()

width, height = pong.screen_dimension
screen = pygame.display.set_mode(
    (width*pong.grid_size, height*pong.grid_size))

k_s, k_w = False, False

note_count = 0
while True:

    if game_intensity == 1:
        if note_count == 0:
            freq = note_gen.generate_note()
            note = Note(freq)
            note.play(150)

        note_count += 1

        if note_count > 3:
            note_count = 0

    elif game_intensity == 2:

        if note_count == 0:
            freq = note_gen.generate_note()
            note = Note(freq)
            note.play(50)

        note_count += 1
        if note_count > 2:
            note_count = 0

    else:
        freq = note_gen.generate_note()
        note = Note(freq)
        note.play(100)


    if(pong.score["AI"] > 2 or pong.score["Player"] > 2):
        game_intensity = 2
        note_gen.game_intensity = 2

    if (pong.score["AI"] > 5 or pong.score["Player"] > 5):
        game_intensity = 3
        note_gen.game_intensity = 3

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

















