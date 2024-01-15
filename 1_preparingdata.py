import warnings, sys, os, pathlib, subprocess, shutil

warnings.filterwarnings("ignore")


alamatparent = str(pathlib.Path(__file__).parent.resolve())
alamat = sys.argv[1]
lang = sys.argv[2]

def modifiedsrt():
    f = open("video."+lang+".srt","r")
    dataline = f.readlines()
    f.close()

    arr = []
    i = 0
    word = " --> "
    baris = 0
    while(i<len(dataline)):
        if word in dataline[i]:
            waktu = dataline[i].split(word)
            t1 = waktu[0].split(":")
            t1 = float(t1[0])*3600+float(t1[1])*60+float(t1[2].replace(",","."))
            t2 = waktu[1].split(":")
            t2 = float(t2[0])*3600+float(t2[1])*60+float(t2[2].replace(",","."))
            selisih = "%.2f" %(t2-t1)
            if selisih==str(0.01):
                for j in range(baris,i-2):
                    arr.append(dataline[j])
                baris = i+3
        i = i+1

    for j in range(baris,i-2):
        arr.append(dataline[j])

    dataline = arr
    arr = []

    i = 0
    word = " --> "
    compare = "\n"
    baris = 0
    while(i<len(dataline)):
        arr.append(dataline[i])
        if word in dataline[i]:
            try:
                if dataline[i+1]==compare:
                    try:
                        compare = dataline[i+2]
                    except IndexError as e:
                        pass
                    i = i+1
                elif dataline[i+1]=="\n":
                    try:
                        compare = dataline[i+2]
                    except IndexError as e:
                        pass
                    i = i+1
            except IndexError as e:
                pass
        i = i+1
    dataline = arr

    f = open("video.srt","w")
    word = " --> "
    compare = "\n"
    baris = 1
    i = 0
    while(i<len(dataline)):
        if word in dataline[i]:
            try:
                if dataline[i+1]!="\n":
                    f.write(str(baris)+"\n")
                    while 1:
                        if dataline[i]!="\n":
                            f.write(dataline[i])
                            i = i+1
                        else:
                            f.write("\n")
                            break        
                    baris = baris+1
            except IndexError as e:
                pass
        i = i+1
    f.close()

def getname(output):
    info = subprocess.check_output("yt-dlp "+output[0]+' --skip-download --print "%(channel)s - %(duration>%H:%M:%S)s - %(title)s"', shell=True)
    info = str(info)
    info = info[2:]
    temp = info.split(" ")
    temp = temp[0].split(" ")
    channel_name = temp[0]
    temp = info.split(" - ")
    temp = temp[2].split(" ")
    playlist_name = temp[0] 
    name = channel_name+"_"+playlist_name
    return name

def downloadvideo(output,lang):
    name = getname(output)
    alamatinput = os.path.join(alamatparent, "video", "input", name)
    if os.path.exists(alamatinput)==True:
        print("There is already folder input")
    else:
        print("There is no folder input")
        os.mkdir(alamatinput)
    alamatinputsubt = os.path.join(alamatparent, "video", "inputsubt", name)
    if os.path.exists(alamatinputsubt)==True:
        print("There is already folder inputsubt")
    else:
        print("There is no folder inputsubt")
        os.mkdir(alamatinputsubt)
    # print("cp "+alamatparent+"/video.mp4 "+alamatinput+"/"+name+"_"+"1"+".mp4")
    # print("cp "+alamatparent+"/video.srt "+alamatinputsubt+"/"+name+"_"+"1"+".txt")

    for i in range(0, len(output)):
        tempstr = "yt-dlp "+str(output[i])+" -S ext:mp4:m4a -o video.mp4 --socket-timeout 1 -4 --sub-lang "+lang+" --write-auto-sub --convert-subs srt --match-filter" 
        # tempstr = "yt-dlp "+str(output[i])+" -S ext:mp4:m4a -o video.mp4 --socket-timeout 1 -4 --sub-lang "+lang+" --write-auto-sub --convert-subs srt --dateafter today-6month --match-filter" 
        # subprocess.call(listtemp, stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
        listtemp = ['yt-dlp', '--rm-cache-dir']
        # subprocess.call(listtemp, stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
        os.system("yt-dlp --rm-cache-dir")
        listtemp = tempstr.split(" ")
        listtemp.append('"'+"license='Creative Commons Attribution license (reuse allowed)'"+'"')
        # print(listtemp)
        # subprocess.call(listtemp, stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
        os.system(tempstr+" "+'"'+"license='Creative Commons Attribution license (reuse allowed)'"+'"')
        # print(tempstr+" "+'"'+"license='Creative Commons Attribution license (reuse allowed)'"+'"')
        modifiedsrt()
        if os.path.exists(os.path.join(alamatparent,"video.mp4"))==True:
            shutil.copy("video.mp4", os.path.join(alamatinput, (name+"_"+str(i+1)+"."+"mp4")))
            os.remove(os.path.join(alamatparent,"video.mp4"))
        if os.path.exists(os.path.join(alamatparent,"video.srt"))==True:
            shutil.copy("video.srt", os.path.join(alamatinputsubt, (name+"_"+str(i+1)+"."+"txt")))
            os.remove(os.path.join(alamatparent,"video.srt"))


def main():
    if os.path.exists(os.path.join(alamatparent,"video","input"))==True:
        print("There is already folder input")
    else:
        print("There is no folder input")
        os.mkdir(os.path.join(alamatparent,"video","input"))
    if os.path.exists(os.path.join(alamatparent,"video","inputsubt"))==True:
        print("There is already folder inputsubt")
    else:
        print("There is no folder inputsubt")
        os.mkdir(os.path.join(alamatparent,"video","inputsubt"))

    if alamat!="off":
        output = subprocess.check_output("yt-dlp "+alamat+" --flat-playlist -j | jq -r .url", shell=True)
        output = str(output)
        output = output[2:]
        output = output.split("\\n")
        output.pop()
        downloadvideo(output,lang)
    else:
        dir = os.path.join(alamatparent,"video","input")+" "+os.path.join(alamatparent,"video","inputsubt")



if __name__ == "__main__":
    main()
