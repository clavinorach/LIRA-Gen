import csv
import os
import pathlib
import pandas as pd
from pathlib import Path

def split_path(path):
    return [str(part) for part in Path(path).parts]

alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamatinput=os.path.join(alamatparent,"video","output_crop")

listdir = os.listdir(alamatinput)
listchild = len(listdir)

print("\n---Check parent folder input and output---")
if os.path.exists(alamatinput)==True:
    print("There is already folder output_crop")
else:
    print("There is no folder output_crop")

print("\n---Process word---")
header = ['word', 'start', 'end', 'vidname','address']

with open('word_filtered.csv', 'w', encoding='UTF8',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

df = pd.read_csv('word_draft.csv')

i =0
while (i<listchild):
    listdirfold = os.listdir(os.path.join(alamatinput,str(listdir[i])))
    totalfold = len(listdirfold)
    print("-"+listdir[i]+"-")
    j = 0
    while (j<totalfold):
        oriaddr = os.path.join(alamatinput,listdir[i],listdirfold[j])
        # subtaddr = alamatinputsubt+"/"+listdir[i]+"/"+listdirfold[j]+".txt"
        listdirfile = os.listdir(os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j])))
        totalfile = len(listdirfile)
        k = 0
        while (k<totalfile):
            temp=listdirfile[k].replace(".mp4","")
            vid = os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j]),listdirfile[k])
            # print(vid)
            # vid = vid.split("/")
            vid = split_path(vid)
            # print(vid)
            namavid = vid[len(vid)-1]
            vid = namavid
            # namavid = vid[len(vid)-1]
            # print(namavid)
            # vid = namavid.replace(".mp4","")
            # print(vid)
            # vid = vid.split("_")
            # print(vid)
            # vid = vid[0]+"_"+vid[1]+"_"+vid[2]+"_"+vid[3]+".mp4"
            # print(vid)
            kata = ""
            t1 = 0
            t2 = 1

            # print(vid)
            x = df.loc[df['vidname'] == vid]
            # print(x)
            try:
                y = x.iloc[0]
            except IndexError:
                continue

            kata = y[0]
            t11 = y[1]
            t12 = y[2]
            t21 = y[3]
            t22 = y[4]

            # with open("word_draft.csv", 'r') as file:
            #     csvreader = csv.reader(file)
            #     for row in csvreader:
            #         if(row[3]==vid):
            #             kata = row[0]
            #             t1 = row[1]
            #             t2 = row[2]
            #             break
            
            # with open("word_draft.csv", 'r') as file:
            #     csvreader = csv.reader(file)
            #     for row in csvreader:
            #         if(row[3]==vid):
            #             kata = row[0]
            #             t1 = row[1]
            #             t2 = row[2]
            #             break


            arrtemp =[]
            arrtemp.append(kata)
            # calc = float(t2)-float(t1)
            # tempcalc = calc
            # calc = 1-calc
            # calc = calc/2
            # t1 = calc
            # t2 = t1+tempcalc
            t1 = t1+(t11-t21)
            t2 = t2-(t22-t12)
            arrtemp.append(str(t1))
            arrtemp.append(str(t2))
            arrtemp.append(namavid)
            arrtemp.append(os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j]),listdirfile[k]))
            

            with open('word_filtered.csv', 'a', encoding='UTF8',newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(arrtemp)
            
            k = k+1
        j = j+1
    i =i+1

# with open("word_draft.csv", 'r') as file:
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         if(row[3]==vid):
#             print("Found")
