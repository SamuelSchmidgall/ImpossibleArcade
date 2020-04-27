# parses MIDI data from folder, outputs a .npy file containing the note values.

MIDI_FOLDER = 'midis/'
OUTPUT_FILENAME = "midi_notes.npy"

import os
import py_midicsv
import numpy as np

path = os.path.join(os.getcwd(), MIDI_FOLDER)
print(path)

files_skipped = 0
files_read = 0
notes_written = 0

a = np.array([])

for filename in os.listdir(path):
    print("READING: ",filename)
    file_path = os.path.join(path, filename)
    try:
        csv_string = py_midicsv.midi_to_csv(file_path)

        for data in csv_string:
            
            temp = data.split()
            
            if temp[2] == "Note_on_c," and (temp[0] == "1," or temp[0] == "2,"):
                print(data)
                
                a = np.append(a,temp[4][:-1])
                
                notes_written += 1

        files_read += 1

    except:
        print("FILE SKIPPED")
        files_skipped += 1

np.save(OUTPUT_FILENAME,a)
print(a)

print("\nWRITTEN TO: ", OUTPUT_FILENAME)
print("\tFILES SKIPPED: ", files_skipped)
print("\tFILES READ: ", files_read)
print("\tNOTES WRITTEN: ", notes_written)
print("\n")


