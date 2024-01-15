from moviepy.editor import *
import sys
import pathlib
import os

sb = str(sys.argv[1])
t1 = float(sys.argv[2])
t2 = float(sys.argv[3])
osv = str(sys.argv[4])
# clip = VideoFileClip(sb)
# clip1 = clip.subclip(t1,t2)
# clip1.write_videofile(osv,codec='libx264')

os.system("ffmpeg -y -ss "+str(t1)+" -t "+str(t2-t1)+" -accurate_seek -i "+sb+" -vcodec "+"libx264"+" "+osv+"")