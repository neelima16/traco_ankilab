import numpy as np
import ffmpeg
import os

directory = './videos/'
i = 51
for f in os.listdir(directory):
    f = os.path.join(directory, f)
    try:
        ffmpeg.input(f).trim(start=0, duration=10).filter('fps', fps=10, round='up').output('./training/training0{0}.mp4'.format(i)).run()
    except Exception as e:
        print(e)
    i +=1