import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
from os import path
import pathlib

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

import dlib
from scipy.spatial import distance as dist

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=1,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath',           type=str, default="/data08/col",  help='Path for inputs, tmps and outputs')

args = parser.parse_args()

# if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
#     Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
#     cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
#     subprocess.call(cmd, shell=True, stdout=None)

# if args.evalCol == True:
# 	# The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
# 	# 	              2. extract audio, extract video frames
# 	#                 3. scend detection, face detection and face tracking
# 	#                 4. active speaker detection for the detected face clips
# 	#                 5. use iou to find the identity of each face clips, compute the F1 results
# 	# The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
# 	# The step 4 and 5 need less than 10 minutes
# 	# Need about 20G space finally
# 	# ```
# 	args.videoName = 'col'
# 	args.videoFolder = args.colSavePath
# 	args.savePath = os.path.join(args.videoFolder, args.videoName)
# 	args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
# 	args.duration = 0
# 	if os.path.isfile(args.videoPath) == False:  # Download video
# 		link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
# 		cmd = "youtube-dl -f best -o %s '%s'"%(args.videoPath, link)
# 		output = subprocess.call(cmd, shell=True, stdout=None)
# 	if os.path.isdir(args.videoFolder + '/col_labels') == False: # Download label
# 		link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
# 		cmd = "gdown --id %s -O %s"%(link, args.videoFolder + '/col_labels.tar.gz')
# 		subprocess.call(cmd, shell=True, stdout=None)
# 		cmd = "tar -xzvf %s -C %s"%(args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
# 		subprocess.call(cmd, shell=True, stdout=None)
# 		os.remove(args.videoFolder + '/col_labels.tar.gz')	
# else:
# 	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
# 	args.savePath = os.path.join(args.videoFolder, args.videoName)

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		# sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	# DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		# print("cek")
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		# sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	# alamatparent = str(pathlib.Path(__file__).parent.resolve())
	# cropFile = alamatparent+"/"+cropFile
	# print(cropFile)
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained TalkNet
	s = talkNet()
	s.loadParameters(args.pretrainModel)
	# sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		try:
			_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
			audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
			video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
			videoFeature = []
			while video.isOpened():
				ret, frames = video.read()
				if ret == True:
					face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
					face = cv2.resize(face, (224,224))
					face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
					videoFeature.append(face)
				else:
					break
			video.release()
			videoFeature = numpy.array(videoFeature)
			length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
			audioFeature = audioFeature[:int(round(length * 100)),:]
			videoFeature = videoFeature[:int(round(length * 25)),:,:]
			allScore = [] # Evaluation use TalkNet
			for duration in durationSet:
				batchSize = int(math.ceil(length / duration))
				scores = []
				with torch.no_grad():
					for i in range(batchSize):
						inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
						inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
						embedA = s.model.forward_audio_frontend(inputA)
						embedV = s.model.forward_visual_frontend(inputV)	
						embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
						out = s.model.forward_audio_visual_backend(embedA, embedV)
						score = s.lossAV.forward(out, labels = None)
						scores.extend(score)
				allScore.append(scores)
			allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
			allScores.append(allScore)	
		except (FileNotFoundError, IndexError) as e:
			pass
	return allScores

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[9]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[7]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

def detect_face(data_predictor, hasil_visualization):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(data_predictor)
	cap = cv2.VideoCapture(hasil_visualization)
	counter = 0
	face_detect = []
	calc_face_detect = 0
	while True:
		# Capture the image from the webcam
		ret, image = cap.read()
		# Convert the image color to grayscale
		try:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			rects = detector(gray, 1)
			# Detect landmarks for each face
			for rect in rects:
				# Get the landmark points
				shape = predictor(gray, rect)
				# Convert it to the NumPy Array
				shape_np = numpy.zeros((68, 2), dtype="int")
				for i in range(0, 68):
					shape_np[i] = (shape.part(i).x, shape.part(i).y)
				shape = shape_np

			if len(rects)!=0:
				landmark_detect = len(shape)
				if landmark_detect>=53: # landmark threshold
					# face_detect.append(1)
					calc_face_detect = calc_face_detect+1
				else:
					pass
					# face_detect.append(0)
			else:
				pass
				# face_detect.append(0)
		except cv2.error as e:
			pass


		counter = counter+1
		if counter==25:
			break

	cap.release()
	# print(calc_face_detect)
	if calc_face_detect>=21: # face threshold
		cap = cv2.VideoCapture(hasil_visualization)
		facevalue=1
		# print("Face detected, calc_face_detect: ", calc_face_detect)
		# Detect mouth open

		# define one constants, for mouth aspect ratio to indicate open mouth
		MOUTH_AR_THRESH = 0.6

		# grab the indexes of the facial landmarks for the mouth
		(mStart, mEnd) = (49, 68)

		counter = 0
		mouth_detect = []
		calc_mouth_detect = 0
		# loop over frames from the video stream
		while True:
			# Capture the image from the webcam
			ret, image = cap.read()
			# Convert the image color to grayscale
			try:
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				# detect faces in the grayscale frame
				rects = detector(gray, 0)

				# loop over the face detections
				for rect in rects:
					# Get the landmark points
					shape = predictor(gray, rect)
					# Convert it to the NumPy Array
					shape_np = numpy.zeros((68, 2), dtype="int")
					for i in range(0, 68):
						shape_np[i] = (shape.part(i).x, shape.part(i).y)
					shape = shape_np

					# extract the mouth coordinates, then use the
					# coordinates to compute the mouth aspect ratio
					mouth = shape[mStart:mEnd]
					mar = mouth_aspect_ratio(mouth)

					# compute the convex hull for the mouth, then
					# visualize the mouth
					mouthHull = cv2.convexHull(mouth)

					# Draw text if mouth is open
					if mar > MOUTH_AR_THRESH:
						# mouth_detect.append(1)
						calc_mouth_detect = calc_mouth_detect+1
			except cv2.error as e:
				pass
			counter = counter+1
			if counter==25:
				break
		cap.release()
		# print(counter)
		if calc_mouth_detect>=5:
			facevalue=1
			# print("Mouth detected, calc_mouth_detect: ", calc_mouth_detect)
		else:
			facevalue=0
			# print("Mouth not detected, calc_mouth_detect: ", calc_mouth_detect)
	else:
		facevalue=0
		# print("Face not detected, calc_face_detect: ", calc_face_detect)
	return facevalue


def visualization(tracks, scores, args):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]

	calc_score = []
	num = 0
	# print(len(scores))
	# print(tracks)
	for tidx, track in enumerate(tracks):
		# print(num)
		try:
			score = scores[tidx]
			calc_score.append("t"+str(num))
			# print("t"+str(num))
			for fidx, frame in enumerate(track['track']['frame'].tolist()):
				s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
				s = numpy.mean(s)
				# print(s)
				if s<=0.0:
					calc_score.append(str("0"))
				else:
					calc_score.append(str("1"))
				faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
		except IndexError as e:
			# print("asd")
			pass
		num = num+1
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
	colorDict = {0: 0, 1: 255}
# edit
	# d = {}
	# for x in range(0, len(tracks)):
	# 	d["vOut{0}".format(x)] = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only'+str(x)+'.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
	# print(calc_score)
	word = "t"
	sum_score = []
	control = 0
	for counter in range(0, len(calc_score)+1):
		if counter==len(calc_score):
			sum_score.append(summing)
			break
		sentence = calc_score[counter]
		if word in sentence:
			if control==0:
				num = int(sentence.replace("t",""))
				summing = 0
				control = 1
			elif control==1:
				num = int(sentence.replace("t",""))
				sum_score.append(summing)
				summing=0			
		else:
			summing = summing+int(sentence)
	maks = sum_score[0]
	indexscore = 0
	for counter in range(0,len(sum_score)):
		if sum_score[counter] > maks:
			maks = sum_score[counter]
			indexscore = counter

	indexscore = '{:05d}'.format(indexscore)
	# print(indexscore)
	alamatparent = pathlib.Path(__file__).parent.resolve()
	# print(args.pycropPath)
	os.system("cp "+args.pycropPath+"/"+indexscore+".avi"+" "+str(alamatparent.parent)+"/"+str(namahasil))
	# print("cp "+args.pycropPath+"/"+indexscore+".avi"+" "+str(alamatparent.parent)+"/"+namahasil)


	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			# print(clr)
			# print("==")
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)

def evaluate_col_ASD(tracks, scores, args):
	txtPath = args.videoFolder + '/col_labels/fusion/*.txt' # Load labels
	predictionSet = {}
	for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
		predictionSet[name] = [[],[]]
	dictGT = {}
	txtFiles = glob.glob("%s"%txtPath)
	for file in txtFiles:
		lines = open(file).read().splitlines()
		idName = file.split('/')[-1][:-4]
		for line in lines:
			data = line.split('\t')
			frame = int(int(data[0]) / 29.97 * 25)
			x1 = int(data[1])
			y1 = int(data[2])
			x2 = int(data[1]) + int(data[3])
			y2 = int(data[2]) + int(data[3])
			gt = int(data[4])
			if frame in dictGT:
				dictGT[frame].append([x1,y1,x2,y2,gt,idName])
			else:
				dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]	
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Load files
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]				
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		if fidx in dictGT: # This frame has label
			for gtThisFrame in dictGT[fidx]: # What this label is ?
				faceGT = gtThisFrame[0:4]
				labelGT = gtThisFrame[4]
				idGT = gtThisFrame[5]
				ious = []
				for face in faces[fidx]: # Find the right face in my result
					faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
					faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
					iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol = True)
					if iou > 0.5:
						ious.append([iou, round(face['score'],2)])
				if len(ious) > 0: # Find my result
					ious.sort()
					labelPredict = ious[-1][1]
				else:					
					labelPredict = 0
				x1 = faceGT[0]
				y1 = faceGT[1]
				width = faceGT[2] - faceGT[0]
				predictionSet[idGT][0].append(labelPredict)
				predictionSet[idGT][1].append(labelGT)
	names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas'] # Evaluate
	names.sort()
	F1s = 0
	for i in names:
		scores = numpy.array(predictionSet[i][0])
		labels = numpy.array(predictionSet[i][1])
		scores = numpy.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			# print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	# print("Average F1:%.2f"%(100 * (F1s / 5)))	  


DET = S3FD(device='cuda')
# Main function
def main():
	# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
	# ```
	# .
	# ├── pyavi
	# │   ├── audio.wav (Audio from input video)
	# │   ├── video.avi (Copy of the input video)
	# │   ├── video_only.avi (Output video without audio)
	# │   └── video_out.avi  (Output video with audio)
	# ├── pycrop (The detected face videos and audios)
	# │   ├── 000000.avi
	# │   ├── 000000.wav
	# │   ├── 000001.avi
	# │   ├── 000001.wav
	# │   └── ...
	# ├── pyframes (All the video frames in this video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...	
	# └── pywork
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     ├── scores.pckl (ASD result)
	#     └── tracks.pckl (face tracking result)
	# ```	
	alamatparent = pathlib.Path(__file__).parent.resolve()
	# alamatparent = pathlib.Path(__file__).resolve().parents[1]

	# nanti hapuslagi untuk semua txt _angka nya
	penanda = str("19")
	kontermulai = 79908
	f = open(os.path.join(alamatparent.parent, '6_temp_'+penanda+'.txt'), 'r')
	data_temp = f.read()
	f.close()
	data_temp = data_temp.split('\n')
	
	f = open(os.path.join(alamatparent.parent, '6_vid_'+penanda+'.txt'), 'r')
	data_vid = f.read()
	f.close()
	data_vid = data_vid.split('\n')
	
	f = open(os.path.join(alamatparent.parent, '6_oriaddr_'+penanda+'.txt'), 'r')
	data_oriaddr = f.read()
	f.close()
	data_oriaddr = data_oriaddr.split('\n')

	f = open(os.path.join(alamatparent.parent, '6_outaddr_'+penanda+'.txt'), 'r')
	data_outaddr = f.read()
	f.close()
	data_outaddr = data_outaddr.split('\n')

	global namahasil
	namahasil = 'hasil_'+penanda+'.mp4'
	total_data = len(data_temp)
	for i in range(kontermulai, total_data):
		args.videoName = data_temp[i]
		print(args.videoName)
		args.videoFolder = data_oriaddr[i]
		            # os.system("python "+demotalk+" --videoName "+temp+" --videoFolder "+oriaddr)

		args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
		args.savePath = os.path.join(args.videoFolder, args.videoName)
		
		# Initialization 
		args.pyaviPath = os.path.join(args.savePath, 'pyavi')
		args.pyframesPath = os.path.join(args.savePath, 'pyframes')
		args.pyworkPath = os.path.join(args.savePath, 'pywork')
		args.pycropPath = os.path.join(args.savePath, 'pycrop')
		if os.path.exists(args.savePath):
			rmtree(args.savePath)
		os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
		os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
		os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
		os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

		# Extract video
		args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
		# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
		if args.duration == 0:
			command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
				(args.videoPath, args.nDataLoaderThread, args.videoFilePath))
		else:
			command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
				(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
		subprocess.call(command, shell=True, stdout=None)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
		
		# Extract audio
		args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
		command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
			(args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
		subprocess.call(command, shell=True, stdout=None)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

		# Extract the video frames
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
			(args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg'))) 
		subprocess.call(command, shell=True, stdout=None)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))

		# Scene detection for the video frames
		scene = scene_detect(args)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))	

		# Face detection for the video frames
		faces = inference_video(args)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

		# Face tracking
		allTracks, vidTracks = [], []
		for shot in scene:
			if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
				allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

		# Face clips cropping
		for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
			try:
				vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
			except cv2.error as e:
				pass
		savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
		with open(savePath, 'wb') as fil:
			pickle.dump(vidTracks, fil)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
		fil = open(savePath, 'rb')
		vidTracks = pickle.load(fil)

		# Active Speaker Detection by TalkNet
		files = glob.glob("%s/*.avi"%args.pycropPath)
		files.sort()
		try:
			scores = evaluate_network(files, args)
		except RuntimeError as e:
			if path.isdir(data_vid[i].replace(".mp4",""))==True:
				os.system("rm -rf "+data_vid[i].replace(".mp4",""))
			continue
		savePath = os.path.join(args.pyworkPath, 'scores.pckl')
		with open(savePath, 'wb') as fil:
			pickle.dump(scores, fil)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

		if args.evalCol == True:
			evaluate_col_ASD(vidTracks, scores, args) # The columnbia video is too big for visualization. You can still add the `visualization` funcition here if you want
			quit()
		else:
			# Visualization, save the result as the new video
			# print("len(vidTracks) = ", len(vidTracks))
			# print("len(scores) = ", len(scores))
			if len(vidTracks)!=0 and len(scores)!=0:
				if len(vidTracks)==len(scores):	
					visualization(vidTracks, scores, args)	

		# join alamatparent.parent with shape_predictor_68_face_landmarks
		data_predictor = str(alamatparent.parent)+"/shape_predictor_68_face_landmarks.dat"
		
		# hasil visualization
		hasil_visualization = str(alamatparent.parent)+"/"+namahasil
		if len(vidTracks)!=0 and len(scores)!=0:
			if len(vidTracks)==len(scores):
				facevalue = detect_face(data_predictor, hasil_visualization)

		# facevalue = 1

		if path.isdir(data_vid[i].replace(".mp4",""))==True:
			os.system("rm -rf "+data_vid[i].replace(".mp4",""))
		if path.isfile(str(alamatparent.parent)+"/"+namahasil)==True:
			# print("cp "+alamatparent+"/hasil.mp4"+" "+alamatoutput_crop+"/"+str(listdir[i])+"/"+str(listdirfold[j]+"/"+listdirfile[k])+".mp4")
			if len(vidTracks)!=0 and len(scores)!=0:
				if len(vidTracks)==len(scores):
					if facevalue == 1:
						os.system("cp "+str(alamatparent.parent)+"/"+namahasil+" "+data_outaddr[i])
			os.system("rm -rf "+str(alamatparent.parent)+"/"+namahasil)

if __name__ == '__main__':
    main()