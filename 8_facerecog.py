import warnings
import PIL
import cv2
import face_recognition
import cv2
import numpy as np
import pathlib
import os
import pandas as pd
import shutil

alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamatinput=os.path.join(alamatparent,"video","output_crop")
alamatoutput_word=os.path.join(alamatparent,"video","output_word_final")

listdir = os.listdir(alamatinput)
listchild = len(listdir)

print("\n---Check parent folder input and output---")
if os.path.exists(alamatinput)==True:
    print("There is already folder output_crop")
else:
    print("There is no folder output_crop")
if os.path.exists(alamatoutput_word)==True:
    print("There is already folder output_word")
else:
    print("There is no folder output_word")
    print("Create folder output_word")
    os.mkdir(alamatoutput_word)

def get_face(namafile):
    vcap = cv2.VideoCapture(namafile) # 0=camera

    if vcap.isOpened(): 
        # get vcap property 
        width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        # or
        width  = vcap.get(3)  # float `width`
        height = vcap.get(4)  # float `height`

        # it gives me 0.0 :/
        fps = vcap.get(cv2.CAP_PROP_FPS)

    success,image = vcap.read()
    cv2.imwrite("frame%d.jpg" % 0, image)
    vcap.release()
    try:
        sample_image = face_recognition.load_image_file("frame0.jpg")
    except PIL.UnidentifiedImageError:
        sample_image ="error"
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        if len(sample_image)==5:
            image_encoding="error"
        else:
            image_encoding="n"
            try:
                image_encoding = face_recognition.face_encodings(sample_image)[0]
            except IndexError as e:
                image_encoding="error"
            # print(len(image_encoding))
    return sample_image, image_encoding

def write_file(vid, outfolder,vidname,kata,count,akhir,awal):
    shutil.copy(vid,os.path.join(outfolder,kata+"_"+"%05d" %count+".mp4"))
    # os.system("cp "+vid+" "+os.path.join(outfolder,kata+"_"+"%05d" %count+".mp4"))                        
    f = open(os.path.join(outfolder+"/"+kata+"_"+"%05d" %count+".txt"),'w')
    f.write("VidID: "+str(vid_id))
    f.write("\n")
    f.write("VidName: "+vidname)
    f.write("\n")
    f.write("ChannelId: "+channel_id)
    f.write("\n")
    f.write("Start: "+str("%.2f"%(awal))+" End: "+str("%.2f"%(akhir)))
    f.write("\n")
    f.write("Duration: "+str("%.2f"%(akhir-awal))+" seconds ")
    f.close()


# Create arrays of known face encodings and their names
known_face_encodings = [
    # image_encoding
]
known_face_names = [
    # "User1",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# face_no = 0
vid_id = 0
channel_id = ""

df = pd.read_csv('word_filtered.csv')
dict_kata = dict()

print("\n---Process crop---")
i=0
while (i<listchild):
    listdirfold = os.listdir(os.path.join(alamatinput,str(listdir[i])))
    totalfold = len(listdirfold)
    print("-"+listdir[i]+"-")

    j=0
    while (j<totalfold):
        print("--"+listdirfold[j]+"--")
        oriaddr = os.path.join(alamatinput,listdir[i],listdirfold[j])
        # print(os.path.join(str(j+1),str(totalfold)), end='\r')

        listdirfile = os.listdir(os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j])))
        totalfile = len(listdirfile)
        k = 0
        while (k<totalfile):
            print("Processing: "+str(k+1)+"/"+str(totalfile)+"     ", end='\r')
            temp=listdirfile[k]
            vid = os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j]),listdirfile[k])

            mask = df['vidname'].str.contains(temp)
            kata = df[mask]
            try:
                awal = kata.iloc[0]['start']
            except IndexError:
                k = k+1
                continue
            akhir = kata.iloc[0]['end']
            vidname = kata.iloc[0]['vidname']
            kata = kata.iloc[0]['word']
            outfolder = os.path.join(alamatoutput_word,kata)
            if os.path.exists(outfolder)==True:
                pass
            else:
                os.mkdir(outfolder)
            sample_image, image_encoding = get_face(vid)
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                if (len(sample_image)==5) or (len(image_encoding)==5):
                    k = k+1
                    continue
            count = (len(os.listdir(outfolder)))/2+1
            write_file(vid, outfolder,vidname,kata,count,akhir,awal)
            k = k+1
        j = j+1
    i = i+1