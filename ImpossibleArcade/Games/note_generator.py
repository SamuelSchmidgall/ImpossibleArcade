# generates a frequency value based on a sequence of midi notes from .npy file

import os
import numpy as np

file_path = "../midi_notes.npy"

class Note_Generator():

    def __init__(self):

        self.note_vals = {"60": 261,"61": 277,"62": 293,"63": 311,"64": 329,"65": 349,"66": 370,"67": 392,"68": 415,"69": 440,
                "70": 466,"71": 493,"72": 523,"73": 554,"74": 587,"75": 622,"76": 659,"77": 698,"78": 739,"79": 783,
                "80": 830,"81": 880,"82": 932,"83": 987,"84": 1046,"85": 1108,"86": 1174,"87": 1244,"88": 1318,"89": 1396}

        self.note_array = np.load(file_path)
        self.note_counter = 0
        self.game_intensity = 1

        #print(self.note_array)

    def generate_note(self):
        n = self.note_array[self.note_counter]
        self.note_counter += 1
        if(self.note_counter >= self.note_array.size - 1):
            self.note_counter = 0
        #print(self.note_vals[n])

        try:
            freq = self.note_vals[n]
        except:
            freq = 10

        return freq
