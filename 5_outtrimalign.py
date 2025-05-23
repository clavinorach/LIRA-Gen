import pathlib
import os
import subprocess
import math
from moviepy import *
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.editor as mp
import moviepy
import csv
import pandas as pd
import cv2
from moviepy.video.fx.all import speedx

# MIN_FREQ = 10

alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamatinput = os.path.join(alamatparent,"video","inputalign")
alamatinputshotvid = os.path.join(alamatparent,"video","output_shot_video")
alamatoutput_trim_align = os.path.join(alamatparent,"video","output_trim_align")

listdir = os.listdir(alamatinput)
listchild = len(listdir)

print("\n---Check parent folder input and output---")
if os.path.exists(alamatinput)==True:
    print("There is already folder inputalign")
else:
    print("There is no folder inputalign")
if os.path.exists(alamatinputshotvid)==True:
    print("There is already folder output_shot_video")
else:
    print("There is no folder output_shot_video")
if os.path.exists(alamatoutput_trim_align)==True:
    print("There is already folder output_trim_align")
else:
    print("There is no folder output_trim_align")
    print("Create folder output_trim_align")
    os.mkdir(alamatoutput_trim_align)

print("\n---Check child folder input and output---")
i=0
while(i<listchild):
    if os.path.exists(os.path.join(alamatoutput_trim_align,str(listdir[i])))==True:
        print("There is already folder output_trim_align "+str(listdir[i]))
    else:
        print("There is no folder output_trim_align "+str(listdir[i]))
        print("Create folder output_trim_align "+str(listdir[i]))
        os.mkdir(os.path.join(alamatoutput_trim_align,str(listdir[i])))
    i = i+1

print("\n---Process preparing for trim align video---")
header = ['word', 'start', 'end', 'vidstart','vidend','vidname','address']

i=0
with open('word_draft.csv', 'w', encoding='UTF8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
csvfile.close()

while (i<listchild):
    listdirfold = os.listdir(os.path.join(alamatinput,str(listdir[i])))
    totalfold = len(listdirfold)
    print("-"+listdir[i]+"-")
    j=0
    while (j<totalfold):
        listdirfile = os.listdir(os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j])))
        totalfile = len(listdirfile)
        folder = os.path.join(alamatoutput_trim_align,str(listdir[i]),listdirfold[j])
        if os.path.exists(folder)==True:
            pass
        else:
            os.mkdir(folder)
        k = 0
        while (k<totalfile):
            print(listdirfile[k].replace(".TextGrid",""))
            textgridname = os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j]),listdirfile[k])
            vidname = os.path.join(alamatinputshotvid,str(listdir[i]),str(listdirfold[j]),listdirfile[k].replace(".TextGrid",".mp4"))
            f = open(textgridname,"r")
            datatextgrid = f.readlines()
            f.close()
            arrword = []
            arrstart = []
            arrend = []
            control = 0
            line=0
            word = "item [1]"
            wordend = "item [2]"
            worddurasi = "xmax = "
            durasi = 0
            while(1):
                if wordend in datatextgrid[line]:
                    break
                if control==0:
                    if worddurasi in datatextgrid[line]:
                        datatemp = datatextgrid[line].replace("xmax = ","")
                        datatemp = datatemp.replace(' ',"")
                        datatemp = datatemp.replace('\n',"")
                        durasi = float(datatemp)
                        durasi = round(durasi,2)
                    if word in datatextgrid[line]:
                        control=1
                        word = "intervals ["
                elif control==1:
                    if word in datatextgrid[line]:
                        word = "text = "
                        control = 2
                if control==2:
                    if word in datatextgrid[line]:
                        datatemp = datatextgrid[line].replace("text = ","")
                        datatemp = datatemp.replace('"',"")
                        datatemp = datatemp.replace(' ',"")
                        datatemp = datatemp.replace('\n',"")
                        if datatemp == "":
                            control=1
                        else:
                            arrword.append(datatemp)
                            datatemp = datatextgrid[line-2].replace("xmin = ","")
                            datatemp = datatemp.replace(' ',"")
                            datatemp = datatemp.replace('\n',"")
                            arrstart.append(datatemp)
                            datatemp = datatextgrid[line-1].replace("xmax = ","")
                            datatemp = datatemp.replace(' ',"")
                            datatemp = datatemp.replace('\n',"")
                            arrend.append(datatemp)
                line = line+1
            l = 0
            df = pd.read_csv('word_check.csv')
            with open('word_draft.csv', 'a', encoding='UTF8') as csvfile:
            # with open('word_draft_'+str(i)+'.csv', 'a', encoding='UTF8') as csvfile:
                writer = csv.writer(csvfile)
                while (l<len(arrword)):
                    arrtemp=[]
                    if arrword[l] in df['word'].unique():
                        arrtemp.append(arrword[l])
                        osv = os.path.join(alamatoutput_trim_align,str(listdir[i]),str(listdirfold[j]),(listdirfile[k].replace(".TextGrid",""))+"_"+str(l+1)+".mp4")
                        shotfile = os.path.join(alamatparent,"2_1_shot.py")
                        t1=float(arrstart[l])
                        t2=float(arrend[l])
                        arrtemp.append(t1)
                        arrtemp.append(t2)
                        t1_fix=0
                        t2_fix=0
                        check_video = VideoFileClip(vidname)
                        check_frame_rate = check_video.fps
                        calc_duration = 1
                        # calc_duration = check_frame_rate / check_frame_rate
                        if t2-t1<=calc_duration:
                            if(t1-((calc_duration-(t2-t1))/2))<0:
                                t1_fix=0.00
                                t2_fix=calc_duration
                            elif(t2+((calc_duration-(t2-t1))/2))>float(durasi):
                                t1_fix=float(durasi)-calc_duration
                                t1_fix=round(t1_fix,2)
                                t2_fix=float(durasi)
                                t2_fix=round(t2_fix,2)
                            else:
                                t1_fix = t1-((calc_duration-(t2-t1))/2)
                                t1_fix = round(t1_fix,2)
                                t2_fix = t1_fix+calc_duration
                                t2_fix = round(t2_fix,2)

                            # os.system("python "+shotfile+" "+vidname+" "+str(t1_fix)+" "+str(t2_fix)+" "+osv)
                            subprocess.call(["python", shotfile, vidname, str(t1_fix), str(t2_fix), osv], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
                            namatemp = listdirfile[k].replace(".TextGrid","")+"_"+str(l+1)+"TEMP_MPY_wvf_snd.mp3"
                            if os.path.isfile(namatemp)==True:
                                os.remove(namatemp)
                            arrtemp.append(t1_fix)
                            arrtemp.append(t2_fix)
                            arrtemp.append(listdirfile[k].replace(".TextGrid","")+"_"+str(l+1)+".mp4")
                            arrtemp.append(os.path.join(alamatoutput_trim_align,str(listdir[i]),str(listdirfold[j]),(listdirfile[k].replace(".TextGrid",""))+"_"+str(l+1)+".mp4"))
                            temp = os.path.join(alamatoutput_trim_align,str(listdir[i]),str(listdirfold[j]),(listdirfile[k].replace(".TextGrid",""))+"_"+str(l+1)+".mp4")
                            cap = cv2.VideoCapture(temp)
                            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            # print(length)
                            cap.release()
                            # clip = VideoFileClip(temp)
                            # # Slow down the clip to make it 1 second long
                            # # Original duration is 0.8333 seconds, so we slow it down by that factor
                            # slow_clip = speedx(clip, final_duration=1)
                            # # Write the slowed down clip to a file (this will have a lower frame rate)
                            # slow_clip.write_videofile(temp, codec="libx264")
                            if length==check_frame_rate:
                                writer.writerow(arrtemp)
                            else:
                                os.remove(temp)
                    l = l+1
            csvfile.close()
            k = k+1
        j = j+1
    i = i+1