import pathlib
import os
import subprocess
import math
import csv
import sys
import pandas
import numpy as np

wordfrequency = sys.argv[1]
langdict = sys.argv[2]

alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamatinput=os.path.join(alamatparent,"video","inputalign")

listdir = os.listdir(alamatinput)
listchild = len(listdir)

print("\n---Check parent folder input and output---")
if os.path.exists(alamatinput)==True:
    print("There is already folder inputalign")
else:
    print("There is no folder inputalign")


print("\n---Process preparing for word analysis---")
df1 = pandas.DataFrame(columns=['word'])
indexpandas=1
i=0
while (i<listchild):
    listdirfold = os.listdir(os.path.join(alamatinput,str(listdir[i])))
    totalfold = len(listdirfold)
    print("-"+listdir[i]+"-")
    j=0
    while (j<totalfold):
        listdirfile = os.listdir(os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j])))
        totalfile = len(listdirfile)
        k = 0
        while(k<totalfile):
            textgridname = os.path.join(alamatinput,str(listdir[i]),str(listdirfold[j]),listdirfile[k])
            f = open(textgridname,"r")
            datatextgrid = f.readlines()
            f.close()
            # arrword = []
            control = 0
            line=0
            word = "item [1]"
            wordend = "item [2]"
            while(1):
                if wordend in datatextgrid[line]:
                    break
                if control==0:
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
                            with open(langdict, 'r') as fp:
                                for l_no, ln in enumerate(fp):
                                    if datatemp in ln:
                                        if indexpandas==1:
                                            dtemp = {"word": [datatemp]}
                                            df1 = pandas.DataFrame(dtemp,index=[str(indexpandas)])
                                        else:
                                            dtemp = {"word": [datatemp]}
                                            dftemp = pandas.DataFrame(dtemp,index=[str(indexpandas)])
                                            df1 = pandas.concat([df1, dftemp])
                                        indexpandas = indexpandas+1
                                        break
                line = line+1
            k = k+1
        j = j+1
    i = i+1

listdatatemp = df1["word"].unique()
datatemp = []
indexpandas = 1
for i in range(0,len(df1["word"].unique())):
    datatemp.append(listdatatemp[i])
for i in range(len(df1["word"].unique()), len(df1)):
    datatemp.append(np.nan)
df1.insert(loc=len(df1.columns), column='wordunique', value=datatemp)
df_filtered = df1.word.value_counts()
datatemp = []
indexpandas = 1
for i in range(0, len(df_filtered)):
    if df_filtered[i]>=int(wordfrequency):
        if indexpandas==1:
            dtemp = {"word": [df_filtered.index[i]],"number": [df_filtered[i]]}
            df1 = pandas.DataFrame(dtemp,index=[str(indexpandas)])
        else:
            dtemp = {"word": [df_filtered.index[i]],"number": [df_filtered[i]]}
            dftemp = pandas.DataFrame(dtemp,index=[str(indexpandas)])
            df1 = pandas.concat([df1, dftemp])
        indexpandas = indexpandas+1
# print(df1)
df1.to_csv('word_check.csv')
