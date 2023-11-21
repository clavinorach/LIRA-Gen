import pathlib
import os
import subprocess
import math
from tqdm import tqdm
import csv
from os import path
import time
import shutil

start = time.time()

alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamatinput=os.path.join(alamatparent,"video","output_trim_align")
alamatoutput_crop=os.path.join(alamatparent,"video","output_crop")

listdir = os.listdir(alamatinput)
listchild = len(listdir)

print("\n---Check parent folder input and output---")
if os.path.exists(alamatinput)==True:
    print("There is already folder output_trim_align")
else:
    print("There is no folder output_trim_align")
if os.path.exists(alamatoutput_crop)==True:
    print("There is already folder output_crop")
else:
    print("There is no folder output_crop")
    print("Create folder output_crop")
    os.mkdir(alamatoutput_crop)

# # belumm
# print("\n---Check child folder input and output---")
# header = ['start', 'end', 'vidname','address']
# with open('word_resolution.csv', 'w', encoding='UTF8') as csvfile:
#     writer = csv.writer(csvfile)
#     # write the header
#     writer.writerow(header)
# csvfile.close()

i=0
while(i<listchild):
    print("-"+str(i+1)+"-")
    if os.path.exists(os.path.join(alamatoutput_crop,str(listdir[i])))==True:
        print("There is already folder output_crop "+str(listdir[i]))
    else:
        print("There is no folder output_crop "+str(listdir[i]))
        print("Create folder output_crop "+str(listdir[i]))
        os.mkdir(os.path.join(alamatoutput_crop,str(listdir[i])))
    i = i+1

print("\n---Process crop---")
i=0
while (i<listchild):
    listdirfold = os.listdir(os.path.join(alamatinput,str(listdir[i])))
    totalfold = len(listdirfold)
    print("-"+listdir[i]+"-")

    j=0
    while (j<totalfold):
        oriaddr = os.path.join(alamatinput,listdir[i],listdirfold[j])
        print(listdirfold[j])
        # subtaddr = alamatinputsubt+"/"+listdir[i]+"/"+listdirfold[j]+".txt"
        listdirfile = os.listdir(os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j])))
        totalfile = len(listdirfile)
        outfolder = os.path.join(alamatoutput_crop,listdir[i],listdirfold[j])
        if os.path.exists(outfolder)==True:
            pass
        else:
            os.mkdir(outfolder)
        k = 0
        while (k<totalfile):
            # break
            print(str(k+1)+"/"+str(totalfile), end='\r')
            temp=listdirfile[k].replace(".mp4","")
            print(temp)
            vid = os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j]),listdirfile[k])
            demotalk = os.path.join(alamatparent,"TalkNet-ASD","demoTalkNet.py")
            os.system("python "+demotalk+" --videoName "+temp+" --videoFolder "+oriaddr)
            # subprocess.call(["python", demotalk, "--videoName", temp, "--videoFolder", oriaddr], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
            if path.isdir(vid.replace(".mp4",""))==True:
                # print("rm -rf "+vid)
                # os.remove(vid.replace(".mp4",""))
                shutil.rmtree(vid.replace(".mp4",""))
                # os.system("rm -rf "+vid.replace(".mp4",""))
            if path.isfile(alamatparent+"/hasil.mp4")==True:
                # print("cp "+alamatparent+"/hasil.mp4"+" "+alamatoutput_crop+"/"+str(listdir[i])+"/"+str(listdirfold[j]+"/"+listdirfile[k])+".mp4")
                shutil.copy(os.path.join(alamatparent,"hasil.mp4"),os.path.join(alamatoutput_crop,str(listdir[i]),str(listdirfold[j]),listdirfile[k]))
                # os.system("cp "+alamatparent+"/hasil.mp4"+" "+alamatoutput_crop+"/"+str(listdir[i])+"/"+str(listdirfold[j]+"/"+listdirfile[k]))
                os.remove(os.path.join(alamatparent,"hasil.mp4"))
                # os.system("rm -rf "+alamatparent+"/hasil.mp4")



            k = k+1
        j = j+1
    i = i+1

end = time.time()
total_time = end - start
print(total_time)
