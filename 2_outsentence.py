import pathlib
import os
import subprocess
import shutil
from pathlib import Path

alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamatinput = os.path.join(alamatparent, "video", "input")
alamatinputsubt = os.path.join(alamatparent, "video", "inputsubt")
alamatoutput_shot = os.path.join(alamatparent, "video", "output_shot")
alamatoutput_shot_video = os.path.join(alamatparent, "video", "output_shot_video")
alamatoutput_shot_info_video = os.path.join(alamatparent, "video", "output_shot_info_video")

listdir = os.listdir(alamatinput)
listchild = len(listdir)

def split_path(path):
    return [str(part) for part in Path(path).parts]

print("\n---Check parent folder input and output---")
if os.path.exists(alamatinput)==True:
    print("There is already folder input")
else:
    print("There is no folder input")
if os.path.exists(alamatinputsubt)==True:
    print("There is already folder inputsubt")
else:
    print("There is no folder inputsubt")

if os.path.exists(alamatoutput_shot)==True:
    print("There is already folder output_shot")
else:
    print("There is no folder output_shot")
    print("Create folder output_shot")
    os.mkdir(alamatoutput_shot)
if os.path.exists(alamatoutput_shot_video)==True:
    print("There is already folder output_shot_video")
else:
    print("There is no folder output_shot_video")
    print("Create folder output_shot_video")
    os.mkdir(alamatoutput_shot_video)
if os.path.exists(alamatoutput_shot_info_video)==True:
    print("There is already folder output_shot_info_video")
else:
    print("There is no folder output_shot_info_video")
    print("Create folder output_shot_info_video")
    os.mkdir(alamatoutput_shot_info_video)

print("\n---Check child folder input and output---")
i=0
while(i<listchild):
    if os.path.exists(os.path.join(alamatoutput_shot,str(listdir[i])))==True:
        print("There is already folder output_shot "+str(listdir[i]))
    else:
        print("There is no folder output_shot "+str(listdir[i]))
        print("Create folder output_shot "+str(listdir[i]))
        os.mkdir(os.path.join(alamatoutput_shot,str(listdir[i])))
    if os.path.exists(os.path.join(alamatoutput_shot_video,str(listdir[i])))==True:
        print("There is already folder output_shot_video "+str(listdir[i]))
    else:
        print("There is no folder output_shot_video "+str(listdir[i]))
        print("Create folder output_shot_video "+str(listdir[i]))
        os.mkdir(os.path.join(alamatoutput_shot_video,str(listdir[i])))
    if os.path.exists(os.path.join(alamatoutput_shot_info_video,str(listdir[i])))==True:
        print("There is already folder output_shot_info_video "+str(listdir[i]))
    else:
        print("There is no folder output_shot_info_video "+str(listdir[i]))
        print("Create folder output_shot_info_video "+str(listdir[i]))
        os.mkdir(os.path.join(alamatoutput_shot_info_video,str(listdir[i])))
    i = i+1

print("\n---Process shot---")
i=0
listdirfile = os.listdir(os.path.join(alamatinput,str(listdir[i])))
totalfile = len(listdirfile)
while (i<listchild):
    print("-"+listdir[i]+"-")
    j=0
    while(j<totalfile):
        temp=listdirfile[j].replace(".mp4","")
        shotfolder = os.path.join(alamatoutput_shot,str(listdir[i]),temp)

        if os.path.exists(shotfolder)==True:
            pass
        else:
            os.mkdir(shotfolder)
        sb = os.path.join(alamatinput,str(listdir[i]),listdirfile[j])
        pred = sb+".predictions.txt"
        sce = sb+".scenes.txt"
        print("Processing "+str(listdirfile[j]))
        os.system("transnetv2_predict "+sb)
        # subprocess.call(["transnetv2_predict", sb], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
        shutil.move(pred, shotfolder)
        shutil.move(sce, shotfolder)

        sce = os.path.join(shotfolder,listdirfile[j]+".scenes.txt")
        f = open(sce,"r")
        rl = f.readlines()
        f.close()
        if os.path.exists(os.path.join(alamatoutput_shot_video,str(listdir[i]),temp))==True:
            pass
        else:
            os.mkdir(os.path.join(alamatoutput_shot_video,str(listdir[i]),temp))
        for k in range(len(rl)):
            t = rl[k].replace("\n","")
            t = t.split(" ")
            t1 = float(float(t[0])/25) #25fps
            startshot = t1
            t2 = float(float(t[1])/25) #25fps
            endshot = t2
            if abs(endshot-startshot)<1:
                continue
            insubt = os.path.join(alamatinputsubt,str(listdir[i]),temp+".txt")
            f = open(insubt,"r")
            datasubt = f.readlines()
            f.close()
            # check datasubt empty or not
            if len(datasubt)==0:
                continue
            # get end video
            word = " --> "
            counter_end = 1
            end_data_value = 0
            while 1:
                end_data=datasubt[len(datasubt)-counter_end]
                if word in end_data:
                    end_data = end_data.replace("\n","")
                    end_data = end_data.split(word)
                    end_data = end_data[1].split(":")
                    end_data_value = (float(end_data[0])*3600)+(float(end_data[1])*60)+(float(end_data[2].replace(",",".")))
                    break
                counter_end = counter_end+1

            line1, line2 = 0, 0
            word = " --> "
            kontrol = 0
            counter_subt = 0
            acu=1
            startsub,endsub=0,0
            startfix, endfix, tempend = 0, 0, 0
            while (counter_subt<len(datasubt)):
                sentence = datasubt[counter_subt]
                if word in sentence:
                    textlama = datasubt[counter_subt].replace("\n","")
                    textlama = textlama.split(word)
                    startsub = textlama[0].split(":")
                    startsub = (float(startsub[0])*3600)+(float(startsub[1])*60)+(float(startsub[2].replace(",",".")))
                    endsub = textlama[1].split(":")
                    endsub = (float(endsub[0])*3600)+(float(endsub[1])*60)+(float(endsub[2].replace(",",".")))
                    if kontrol==0:
                        if (startshot<=startsub)or(abs(startshot-startsub)<1):
                            startfix = startsub
                            if (counter_subt-1)<0:
                                line1 = 0
                            else:
                                line1 = counter_subt-1
                            kontrol = 1
                    if kontrol==1:
                        if endshot>endsub:
                            tempend = endshot
                        else:
                            if counter_subt-1<0:
                                line2 = 0
                            else:
                                line2 = counter_subt-1
                            endfix = tempend
                            break
                counter_subt = counter_subt+1
            if endfix==0:
                line2 = line1+3
                endfix=endsub
            
            start = startfix
            end = endfix
            control = 0
            if len(rl)==1:
                control = 1
            elif len(rl)>1:
                if (start==0) and (end>=end_data_value):
                    control = 0
                else:
                    control = 1
            if control==1:
                osv = os.path.join(alamatoutput_shot_video,str(listdir[i]),temp,temp+"_"+str(k+1)+".mp4")
                shotfile = os.path.join(alamatparent,"2_1_shot.py")
                # os.system("python "+shotfile+" "+sb+" "+str(start)+" "+ str(end)+" "+ osv)
                subprocess.call(["python", shotfile, sb, str(start), str(end), osv], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
                alamatparent = str(pathlib.Path(__file__).parent.resolve())
                # osv = osv.split("/")
                osv = split_path(osv)
                nameosv=osv[len(osv)-1]
                nameosv=nameosv.replace(".mp4","")
                nameosv=os.path.join(alamatparent,nameosv+"TEMP_MPY_wvf_snd.mp3")
                if os.path.exists(nameosv)==True:
                    os.system("rm "+nameosv)
                if os.path.exists(os.path.join(alamatoutput_shot_info_video,str(listdir[i]),temp))==True:
                    pass
                else:
                    os.mkdir(os.path.join(alamatoutput_shot_info_video,str(listdir[i]),temp))
                osubt = os.path.join(alamatoutput_shot_info_video,str(listdir[i]),temp,temp+"_"+str(k+1)+".txt")
                f = open(osubt,"w")
                linenumber = line1
                while(1):
                    linenumber=linenumber+1
                    if linenumber>line2:
                        if linenumber<len(datasubt):
                            break
                        if linenumber>=len(datasubt):
                            break
                        if datasubt[linenumber]=="\n":
                            break
                        
                    f.write(datasubt[linenumber-1])
                f.close()

        j = j+1
    i = i+1
