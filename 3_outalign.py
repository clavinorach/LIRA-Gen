import pathlib
import os
import subprocess
import math
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import moviepy.editor as mp
import moviepy

alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamatinput = os.path.join(alamatparent, "video", "output_shot_video")
alamatinputsubt = os.path.join(alamatparent, "video", "output_shot_info_video")
alamatoutput_align_input = os.path.join(alamatparent, "video", "output_align_input")

listdir = os.listdir(alamatinput)
listchild = len(listdir)

print("\n---Check parent folder input and output---")
if os.path.exists(alamatinput)==True:
    print("There is already folder output_shot_video")
else:
    print("There is no folder output_shot_video")
if os.path.exists(alamatinputsubt)==True:
    print("There is already folder output_shot_info_video")
else:
    print("There is no folder output_shot_info_video")
if os.path.exists(alamatoutput_align_input)==True:
    print("There is already folder output_align_input")
else:
    print("There is no folder output_align_input")
    print("Create folder output_align_input")
    os.mkdir(alamatoutput_align_input)

print("\n---Check child folder input and output---")
i=0
while(i<listchild):
    if os.path.exists(os.path.join(alamatoutput_align_input,str(listdir[i])))==True:
        print("There is already folder output_align_input "+str(listdir[i]))
    else:
        print("There is no folder output_align_input "+str(listdir[i]))
        print("Create folder output_align_input "+str(listdir[i]))
        os.mkdir(os.path.join(alamatoutput_align_input,str(listdir[i])))
    i = i+1

print("\n---Process preparing for align input---")
i=0
while (i<listchild):
    listdirfold = os.listdir(os.path.join(alamatinput,str(listdir[i])))
    totalfold = len(listdirfold)
    print("-"+listdir[i]+"-")

    j=0
    while (j<totalfold):
        try:
            listdirfile = os.listdir(os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j])))
            listdirsub = os.listdir(os.path.join(alamatinputsubt,str(listdir[i]),str(listdirfold[j])))
        except:
            j = j+1
            continue
        totalfile = len(listdirfile)
        folder = os.path.join(alamatoutput_align_input,str(listdir[i]),listdirfold[j])

        if os.path.exists(folder)==True:
            pass
        else:
            os.mkdir(folder)
        k = 0
        while (k<totalfile):
            invid = os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j]),listdirfile[k])
            outwav = os.path.join(alamatoutput_align_input,str(listdir[i]),str(listdirfold[j]),listdirfile[k].replace(".mp4",".wav"))
            subprocess.call(["ffmpeg","-y", "-i", invid, outwav], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
            
            
            f = open(os.path.join(alamatinputsubt,str(listdir[i]),str(listdirfold[j]),listdirfile[k].replace(".mp4",".txt")),'r')
            datalines=f.readlines()
            f.close()
            control = 0
            word = " --> "
            teks = ""
            for linenumber in range(0, len(datalines)):
                buff=datalines[linenumber]
                if control==0:
                    if word in buff:
                        control=1
                elif control==1:
                    if buff=="\n":
                        control=0
                    else:
                        teks = teks+" "+buff.replace("\n","")
            outlab = os.path.join(alamatoutput_align_input,str(listdir[i]),str(listdirfold[j]),listdirfile[k].replace(".mp4",".lab"))
            f = open(outlab,'w')
            f.write(teks)
            f.close()

            k = k+1
        j = j+1
    i = i+1
