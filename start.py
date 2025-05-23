import warnings, argparse, os, pathlib, subprocess
import validators
import time

warnings.filterwarnings("ignore")

print("End-to-End Lipreading Generator")
print("Stage (1) Preparing data")
print("Stage (2) Out Sentence")
print("Stage (3) Out Align")
print("Stage (4) Word Filtered")
print("Stage (5) Out Trim After Aligned")
print("Stage (6) Out Word")
print("Stage (7) Word CSV Processing")
print("Stage (8) Face Dict")

alamatparent = str(pathlib.Path(__file__).parent.resolve())

parser = argparse.ArgumentParser(description = "End-to-End Lipreading Generator")

parser.add_argument('--videoPlaylist',         type=str, default="off",   help='Youtube playlist link')
parser.add_argument('--wordFrequency',         type=int, default="10",   help='Minimum frequency of word')
parser.add_argument('--lang',                  type=str, default="id",   help='Language')
parser.add_argument('--dictionary',            type=str, default="indonesian_words.txt",   help='Path for language dictionary')
parser.add_argument('--stage',                 type=int, default="1",   help='Start processing stage')

args = parser.parse_args()


def stage1():
    print("Preparing data (-)", end='\r')
    if args.videoPlaylist!="off":
        # Modified: Using os.system instead of subprocess to see errors
        os.system("python 1_preparingdata.py "+args.videoPlaylist+" "+args.lang)
        print("Preparing data ("+u'\u2713'+")")
    else:
        print("Preparing data ("+u'\u2713'+")")

def stage2():
    print("Sentence processing (-)", end='\r')
    # Modified: Using os.system instead of subprocess to see errors
    os.system("python 2_outsentence.py")
    print("Sentence processing ("+u'\u2713'+")")

def stage3():
    print("Align processing (-)", end='\r')
    # Modified: Using os.system instead of subprocess to see errors
    os.system("python 3_outalign.py")
    print("Align processing ("+u'\u2713'+")")

def stage4():
    print("Word filtering (-)", end='\r')
    # Modified: Using os.system instead of subprocess to see errors
    os.system(f"python 4_wordfiltered.py {args.wordFrequency} {args.dictionary}")
    print("Word filtering ("+u'\u2713'+")")

def stage5():
    print("Word trimming (-)", end='\r')
    # Modified: Using os.system instead of subprocess to see errors
    os.system("python 5_outtrimalign.py")
    print("Word trimming ("+u'\u2713'+")")

def stage6():
    print("Face cropping (-)", end='\r')
    # Already using os.system
    os.system("python 6_outwordtrim.py")
    print("Face cropping ("+u'\u2713'+")")

def stage7():
    print("Csv data (-)", end='\r')
    # Modified: Using os.system instead of subprocess to see errors
    os.system("python 7_outwordcsv.py")
    print("Csv data ("+u'\u2713'+")")

def stage8():
    print("Out word (-)", end='\r')
    # Modified: Using os.system instead of subprocess to see errors
    os.system("python 8_facerecog.py")
    print("Out word ("+u'\u2713'+")")

# Main function
def main():
    if os.path.exists(os.path.join(alamatparent, "video"))==True:
        pass
    else:
        os.mkdir(os.path.join(alamatparent, "video"))

    if os.path.exists(os.path.join(alamatparent, "video", "inputalign"))==True:
        pass
    else:
        os.mkdir(os.path.join(alamatparent, "video", "inputalign"))

    if args.videoPlaylist=="off":
        inputdata = os.path.join(alamatparent, "video", "input") + " " + os.path.join(alamatparent, "video", "inputsubt")
    else:
        if validators.url(args.videoPlaylist)==True:
            inputdata = args.videoPlaylist
        else:
            exit()
            
    if args.videoPlaylist=="off":
        print("Input: (DEFAULT) "+inputdata)
    else:
        print("Input: "+inputdata)
        if "list" in inputdata:
            pass
        else:
            print("link not youtube playlist")
            exit()
    if str(args.wordFrequency)=="10":
        print("Word Frequency: (DEFAULT) "+str(args.wordFrequency))
    else:
        print("Word Frequency: "+str(args.wordFrequency))
    if args.lang=="id":
        print("Language: (DEFAULT) "+args.lang)
    else:
        print("Language: "+args.lang)
    if args.dictionary=="indonesian_words.txt":
        print("Dictionary: (DEFAULT) "+args.dictionary)
    else:
        print("Dictionary: "+args.dictionary)
    

    yes_choices = ['yes', 'Yes','y', 'Y']
    no_choices = ['no', 'No', 'n', 'N']

    inputalign = os.path.join(alamatparent, "video", "inputalign")
    waktu = os.path.getmtime(inputalign)

    while(1):
        user_input = input('Do you like to proceed (y/n): ')
        
        if user_input.lower() in yes_choices:
            print('Program starting...')
            start = time.time()
            stage1()
            stage2()
            stage3()
            while(1):
                user_input2 = input('---\nPlease proceed by doing forced alignment using MFA with input in video/output_align_input \nCopy forced alignment result to video/inputalign \n--- \nProceed next stage? (y/n): ')
                if user_input2.lower() in yes_choices:
                    waktu2 = os.path.getmtime(inputalign)
                    if waktu==waktu2:
                        print("tidak berubah")
                    else:
                        break
                else:
                    exit(1)
            stage4()
            stage5()
            stage6()
            stage7()
            stage8()
            end = time.time()
            total_time = end - start
            print(total_time)

            break
        elif user_input.lower() in no_choices:
            print('Exitting...')
            break
        else:
            print('Type yes or no')

if __name__ == '__main__':
    main()