from skimage.metrics import structural_similarity as ssim
from src.templates.workerprocess import WorkerProcess 

import cv2 
import socket 
import struct



#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np
import src.data.trafficlights.trafficlights as trafficlights
import src.data.vehicletovehicle.vehicletovehicle as vehicletovehicle
from src.data.livetraffic.livetraffic import EnvironmentalHandler
from src.data.localisationssystem.locsys import LocalisationSystem
import time
import random
from threading import Thread
from multiprocessing import Pipe

#from skimage.metrics import structural_similarity as ssim

class LineDetection(WorkerProcess):
	flag = 1
	def __init__(self, inPs, outPs, inSh):
		self.flag = 1
		
		self.enableTraficLightsServer = False
		self.enableServerV2V = False
		self.enableLiveTraficServer = False
		self.enableLocalizationServer = False
		self.MyXcord = 0
		self.Myycord = 0   
		self.TraficLightSr = None
		self.CarXCord = 0
		self.CarYcord = 0
		self.ObstacleID = 0
		self.encIm = 0
		self.polEnc = 0

		self.inSh = inSh
		
		path = "imgs/"
		stop = cv2.imread(path+"stopcut.png")
		jednosmjerna = cv2.imread(path+"jednosmjernacut.png")
		prvenstvo = cv2.imread(path+"prvenstvocut.png")
		parking = cv2.imread(path+"parkingcut.png")
		pjesacki = cv2.imread(path+"pjesackicut.png")
		autoput = cv2.imread(path+"autoputcut.png")
		kraj_autoputa = cv2.imread(path+"kraj_autoputacut.png")
		kruzni = cv2.imread(path+"kruznicut.png")
		obavezno_pravo = cv2.imread(path+"obavezno_pravocut.png")
		
		self.blue = []
		
		#blue_signs = ["parking", "pjesacki", "obavezno_pravo", "kruzni", "jednosmjerna", "stop","prvenstvo", "autoput", "kraj_autoputa" ]
		self.blue.append(parking)
		self.blue.append(pjesacki)
		self.blue.append(obavezno_pravo)
		self.blue.append(kruzni)
		self.blue.append(jednosmjerna)
		self.blue.append(stop)
		self.blue.append(prvenstvo)
		self.blue.append(autoput)
		self.blue.append(kraj_autoputa)
		
		
		super(LineDetection, self).__init__(inPs, outPs, inSh)
	
	def run(self):
		self._init_socket()
		super(LineDetection, self).run()
		
	def crop_image(self, img):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)
		avg = np.average(h)
		
		return avg
		
	def compareImage(self, imageIn):
		
		h, w, _ = imageIn.shape
		
		imgC = imageIn[0:round(h/3),round(3*w/4):w]
		CopyImg = imgC.copy()
		imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
		detected_frame, have_contour, avg = self.FindContures(imgC, CopyImg)

		imgC =cv2.resize(detected_frame, (80, 80), interpolation = cv2.INTER_AREA)
		index = -1
		res = []
		
		blue_signs = ["parking", "pjesacki", "obavezno_pravo", "kruzni", "jednosmjerna","stop","prvenstvo", "autoput", "kraj_autoputa" ]
		tolerance = 5 
		if have_contour == True:
		
			for img in self.blue:
				#cv2.imshow("daj da radi", img)
				#cv2.waitKey(0)
				img =cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
				s = ssim(imgC, img[:,:,1])
				res.append(s)
			
			index = res.index(max(res))
			print("$$$$$$$$$ ", index, " $$$$$$$$$$$$$$$")
			print(res)
			if index == 5 and max(res) >  0.15:
				return blue_signs[index]
			if(max(res) < 0.20):
				return 0
			if index == 6:
				if 40 - tolerance <= avg <= 40 + tolerance:
					return blue_signs[index]
				else:
					return 0
			print("$$$$$$$$$", blue_signs[index])
			return blue_signs[index]
		else:
			return 0


		
	def FindContures(self, imgIn, copy_frame):
		copy_frame = cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)
		blur = cv2.GaussianBlur(imgIn, (27,27), 0)
		
		thresh = cv2.adaptiveThreshold(blur,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)
		thresh = cv2.bitwise_not(thresh)
		
		#cv2.imshow("blur1", thresh)
		#cv2.waitKey(1)
		detected_frame = copy_frame.copy()
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours_founded = []
		have_contour = False
		for contour in contours:  # za svaku konturu
			center, sizeS, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
			width_signs, height_signs = sizeS
			if width_signs > 25 and width_signs < 125 and height_signs > 35 and height_signs < 125 :#and abs(height_signs-width_signs) < 40:  # uslov da kontura pripada znaku
				detected_frame = thresh
				center_height = round(center[0])
				center_width = round(center[1])
				new_width = round(width_signs/2)
				new_height = round(height_signs/2)
				detected_frame = copy_frame[center_width-new_width:new_width + center_width, center_height-new_height:center_height + new_height]
				contours_founded.append(contour)
				have_contour = True
				#print("ceawhnfuh",contours_founded)
				t = time.time()
				"""
				try:
					cv2.imwrite("imgs/imgs/img"+str(t)+".png",detected_frame)
				except Exception as e:
					print(e)
				"""
		cv2.drawContours(copy_frame, contours_founded, -1, (255, 0, 0), 1)
		avg = self.crop_image(detected_frame)
		print("~~~~~~~~~~~~~~~",avg,"~~~~~~~~~~~~~~~")

		#cv2.imshow("blur", detected_frame)
		#cv2.waitKey(1)
		try:
			ret = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2GRAY)
		except Exception as e:
			print(e)
		return ret, have_contour, avg
		
	def gray(self, image):
		#print("gray")
		im = np.asarray(image)
		gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
		thresh1 = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 5)
		return thresh1
		
	def gauss(self, image):
		return cv2.GaussianBlur(image, (11,11), 0)
		
	def Canny(self, image):
		edges = cv2.Canny(image, 100 , 200)
		return edges
	
	def region(self, image):
		height, width = image.shape
		mask = np.zeros_like(image)
		h_min = int(height/2) - 50
		h_max = int(height*3/4) 
		w_min = int(width/2)
		for i in range(h_min, h_max):
			#print(i)
			for j in range (w_min, width-1):
				#print("j: ", j)
				mask[i][j] = 255
		isreg = cv2.bitwise_and(image, mask)
		return isreg	
		
	def display_lines(self, image, lines):
		lines_image = np.zeros_like(image)
		if lines is not None:
			for line in lines:
				if line is not None:
					x1, y1, x2, y2 = line.reshape(4)
					cv2.line(lines_image, (x1, y1),(x2, y2), (255, 0, 0), 10)
		return lines_image
		
	def average(self, image, lines):
		left = []
		right = []
		final_list = []
		line_det = False
		direction = 0
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			parameters = np.polyfit((x1, x2), (y1, y2), 1)
			slope = parameters[0]
			y_int = parameters[1]
			if slope < 0 :
				left.append((slope, y_int))
			else:
				right.append((slope, y_int))
		if left != []:
			left_avg = np.average(left, axis=0)
			slope, y_int = left_avg
			if abs(slope) > 0.02:
				direction = 2
				left_line = self.make_points(image, left_avg)
				final_list.append(left_line)
				line_det = True
		elif right != []:
			right_avg = np.average(right, axis=0)
			slope, y_int = right_avg
			if abs(slope) > 0.5:
				right_line = self.make_points(image, right_avg)
				if (right_line[0] < int(image.shape[1]*4/5) and right_line[2] < int(image.shape[1]*4/5)):
					direction = 4
				if  0.4 < slope < 0.9:
					direction = 1
				#print(right_line)
				final_list.append(right_line)
				line_det = True
			else:
				print("U RASKRSNICI ", slope)
		try:
			final_list = np.array(final_list)
			#print("(((((((((((((((((((((((((((((((((((", final_list)
		except:
			print("cannot convert")
		return final_list, line_det, direction
		
	def make_points(self, image, average):
		slope, y_int = average
		print(slope)
		#isLeft = 0
		#if abs(slope) > 0.5:
		y1 = int(image.shape[0]*0.5)
		y2 = int(image.shape[0]*0.75)
		x1 = int((y1 - y_int)//slope)
		x2 = int((y2 - y_int)//slope)
		"""if (x1 < int(image.shape[1]*4/5) and x2 < int(image.shape[1]*4/5)) or 0 < slope < 0.85:
			isLeft = 1
		elif slope < -0.5:
			isLeft = 2"""
		return np.array(([x1, y1,  x2, y2]))
		#else:
			#return np.array(([0, 0,  0, 0]))	
	def increase_brightness(self, img, value = 30):
		hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		h, s, v = cv2.split(hsv)
		
		lim = 255 - value
		v[v > lim] = 255
		v[v <= lim] += value
		
		final_hsv = cv2.merge((h, s, v))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
		return img
		
	def clasificate_img(self, avg_color):
		hsv = cv2.cvtColor(avg_color, cv2.COLOR_BGR2HSV)
		
		h, s, v = cv2.split(hsv)
		avg = np.average(h)
		tolerance = 5
		print("AVG: ", avg)
		
		if 133 - tolerance < avg < tolerance + 133:
			print("PRVENSTVO PROLAZA")
			return True, 0
		#elif 75 - tolerance < avg < tolerance + 75:
		#	print("PJESACKI")
		#	return True
		elif 85 - tolerance < avg < tolerance + 85: #85
			print("PARKING")
			return True, 1
		elif 118 - tolerance < avg < tolerance + 118:
			print("STOP")
			return True, 2
		elif 90 - tolerance < avg < 90 + tolerance:
			print("AUTOPUT")
			return True,4
		else:
			print("======")
			return False, 3

	def traffic_light(self, img):
		#crop image
		
		flag = True
		h, w, _ = img.shape
		w1 = w
		h = round(3*h/4) 
		w = round(2*w/3)
		w_max = w1
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		copy_frame = img.copy()
		img = img[0:h, w: w_max, :]
		img_show = copy_frame[0:h, w: w_max, :]
		new_height, _, _ = img.shape
		
		img = cv2.GaussianBlur(img, (15, 15), 0)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		_, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		
		contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_list = []
		position = -1
		
		avg = 0
		for contour in contours:
			center, size, angle = cv2.minAreaRect(contour)
			width, height = size
			#print("====================================", width, " ", height, "=======================")
			if width > 30 and width < 100 and height > 30 and height < 100 and abs(height - width)<10:
				contour_list.append(contour)
				position_y = round(center[0])
				position_x = round(center[1])
				width_new = round(width / 2) - 7
				height_new = round(height / 2) - 7
				
				detected_frame = img_show[position_x - height_new : height_new + position_x, position_y - width_new : position_y + width_new]
				hsv = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2HSV)
		
				h, s, v = cv2.split(hsv)
				avg = np.average(h)
				print("**********COLOR***********:", avg)
				
				#cv2.imshow("semafongr", detected_frame)
				#cv2.waitKey(1)
				
		cv2.drawContours(img_show, contour_list,  -1, (255,0,0), 2)
		#cv2.imshow("semafor", img_show)
		#cv2.waitKey(1)
		
		tolerance = 10
		if 60 - tolerance < avg < 60 + tolerance:
			print("UPALJENO JE : zeleno")
			flag = True
		elif 1700 - tolerance < avg < 1700 + tolerance:
			print("UPALJENO JE : crveno")
			flag = False
		else:
			print("NO TRAFFIC LIGHT")
		return flag

	def _init_threads(self):
		print("\n LaneDet thread inited \n")
		if self._blocker.is_set():
			return 
		
		StreamTh = Thread(name='LaneDetectionThread', target = self._send_thread, args= (self.inPs[0], self.outPs))
		StreamTh.daemon = True
		self.threads.append(StreamTh)
		
		StreamSh = Thread(name='ReadFromShTread', target = self._sendSH, args = ())
		StreamSh.daemon = True
		self.threads.append(StreamSh)
		
		if self.enableTraficLightsServer:
			StreamSerTL = Thread(name = 'TraficLiteThread', target = self._TraficLightServer, args = ())
			StreamSerTL.daemon = True
			self.threads.append(StreamSerTL)
			
		if self.enableServerV2V:
			StreamSerVl = Thread(name = 'Viacle2ViacleThread', target = self._ViacleToViacle, args = ())
			StreamSerVl.daemon = True
			self.threads.append(StreamSerVl)
			
		if self.enableLiveTraficServer:
			StreamSerLT = Thread(name = 'LiveTraficThread', target = self._LiveTrafic, args = ())
			StreamSerLT.daemon = True
			self.threads.append(StreamSerLT)
		
		if self.enableLocalizationServer:
			StreamserLoc = Thread(name = 'LocalizationThread', target = self._Localization, args = ())
			StreamserLoc.daemon = True
			self.threads.append(StreamserLoc)
		
		
	 # ===================================== INIT SOCKET ==================================
	def _init_socket(self):
		"""Initialize the socket client. 
		"""
		self.serverIp   =  '192.168.64.187' # PC ip
		self.port       =  2244            # com port

		self.client_socket = socket.socket()
		self.connection = None
		# Trying repeatedly to connect the camera receiver.
		try:
			while self.connection is None and not self._blocker.is_set():
				try:
					self.client_socket.connect((self.serverIp, self.port))
					self.client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
					self.connection = self.client_socket.makefile('wb') 
				except ConnectionRefusedError as error:
					time.sleep(0.5)
					pass
		except KeyboardInterrupt:
			self._blocker.set()
			pass

	def _sendSH(self):
		while True:
			#print("11111")
			self.encIm  = self.inSh[0].recv()
			try:
				self.encIm =  float(msg)
				print("gjjwedhgiuf",self.polEnc)
			except:
				msg = ""
			#print("555555")
			#print(self.polEnc)
			#print("88888")
	def _Localization(self):
		beacon = 12345 #12345
		id = 1
		#print("5555555")
		serverpublickey = 'src/data/localisationssystem/publickey_server_test.pem'
		
		gpsStR, gpsStS = Pipe(duplex = False)
		
		localisationSystem = LocalisationSystem(id, beacon, serverpublickey, gpsStS)
		localisationSystem.start()  
		#print("000000")
		time.sleep(5)
		while True:
			try:
				if(self.ObstacleID != 0):
					if gpsStR.poll():
						coora = gpsStR.recv()
					print(coora['timestamp'], coora['pos'].real, coora['pos'].imag)
					self.MyXcord = coora['pos'].real
					self.MyYcord = coora['pos'].imag
					time.sleep(1)
			except KeyboardInterrupt:
				break
			
		localisationSystem.stop()

		localisationSystem.join()
	
	def _TraficLightServer(self):
		colors = ['red','yellow','green']   
		# Get time stamp when starting tester
		start_time = time.time()
		# Create listener object
		Semaphores = trafficlights.trafficlights()
		# Start the listener
		Semaphores.start()
		# Wait until 60 seconds passed
		while (time.time()-start_time < 60):
		# Clear the screen
#print("\033c")
			print("Example program that gets the states of each semaphore from their broadcast messages\n")
			# Print each semaphore's data
			print("S1 color " + colors[Semaphores.s1_state] + ", code " + str(Semaphores.s1_state) + ".")
			print("S2 color " + colors[Semaphores.s2_state] + ", code " + str(Semaphores.s2_state) + ".")
			print("S3 color " + colors[Semaphores.s3_state] + ", code " + str(Semaphores.s3_state) + ".")
			print("S4 color " + colors[Semaphores.s4_state] + ", code " + str(Semaphores.s4_state) + ".")
			self.TraficLightSr = Semaphores
			#print("cwegyewf " + str(self.TraficLightSr.s1_state))
			time.sleep(0.5)

		Semaphores.stop()
		
	def _ViacleToViacle(self):
		# Get time stamp when starting tester
		start_time = time.time()
		# Create listener object
		#print("--------------------------")
		vehicle = vehicletovehicle.vehicletovehicle()
		# Start the listener
		#print("--------------------------")
		vehicle.start()
		#print("--------------------------")
		# Wait until 60 seconds passed
		while (time.time()-start_time < 60):
			# Clear the screen
			print("Example program that gets the info of the last car infos\n")
			# Print each received msg
			print("ID ", vehicle.ID, ", coor ", vehicle.pos, ", angle ", vehicle.ang)
			self.CarXCord = vehicle.pos.real
			self.CaryCord = vehicle.pos.imag
			#print("neduwhfviuw")
			time.sleep(0.5)
		# Stop the listener
		vehicle.stop()
			
			
	def _LiveTrafic(self):
		beacon = 23456

		id = 120
		serverpublickey = 'src/data/livetraffic/publickey_livetraffic_server_test.pem'
		clientprivatekey = 'src/data/livetraffic/privatekey_livetraffic_client_test.pem'

		#: For testing purposes, with the provided simulated livetraffic_system, the pair of keys above is used
		#       -   "publickey_livetraffic_server_test.pem"     --> Ensure by the client that the server is the actual server
		#       -   "privatekey_livetraffic_client_test.pem"    --> Ensure by the server that the client is the actual client

		#: At Bosch location during the competition 
		#       -   Use the "publickey_livetraffic_server.pem" instead of the "publickey_livetraffic_server_test.pem" 
		#       -   As for the "publickey_livetraffic_client.pem", you will have to generate a pair of keys, private and public, using the following terminal lines.  Before the competition, instruction of where to send your publickey_livetraffic_client.pem will be given.

		#: openssl genrsa -out privateckey_livetraffic_client.pem 2048 ----> Creates a private ssh key and stores it in the current dir with the given name
		#: openssl rsa -in privateckey_livetraffic_client.pem -pubout -out publickey_livetraffic_client.pem ----> Creates the corresponding public key out of the private one. 

		#:
		#: To test the functionality, 
		#       -   copy the generated public key under test/livetrafficSERVER/keys 
		#       -   rename the key using this format: "id_publickey.pem" ,where the id is the id of the key you are trying to connect with
		#       -   The given example connects with the id 120 and the same key is saved with "120_publickey.pem"
		
		gpsStR, gpsStS = Pipe(duplex = False)
		#print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
		envhandler = EnvironmentalHandler(id, beacon, serverpublickey, gpsStR, clientprivatekey)
		envhandler.start()
		#print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
		time.sleep(5)
		#for x in range(1, 10):
		while True:
			time.sleep(random.uniform(1,5))
			#a = {"obstacle_id": int(random.uniform(0,25)), "x": random.uniform(0,15), "y": random.uniform(0,15)}
			#self.ObstacleID = int(random.uniform(0,25))
			a = {"obstacle_id": self.ObstacleID , "x": self.MyXcord, "y": self.MyYcord}

			if(self.ObstacleID != 0):
				gpsStS.send(a)
				self.ObstacleId = 0
				print("MSG SEND TO ENV SERVER")
				print(a)
				print("555555555555555555")
		envhandler.stop()
		envhandler.join()
		
	def _send_thread(self, inP, outPs):
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
		flag = 1
		is_sign_clasified = False
		isStop = True
		sign = -1
		time = 0
		isRight = -1
		inParking = -1
		isTraficLight = True
		inParkingTime = 0
		isRedLight = True
		prev = -100
		pick_left_line = 0
		ignore_left_line = False
		is_priority = False
		time_p = 0
		cur_flag = 1
		parkiraj_se =  False
		ne_radi_stop = False
		mozes_prvenstvo = False
		msg = {'action': '1', 'speed': 0.12}
		while True:
			try:
				if flag == 1:
					#msg = {"action": '1', 'speed': 0.09}
					stamps, frame = inP.recv()
					lanes = frame
					copy_frame = frame.copy()
					"""
					try:
						print("#PREPOZNATI ZNAK JE: ",self.compareImage(test))
					except Exception as e :
						print(e)
					"""
					#copy_frame =  cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)	
					height_signs, width_signs, _ = frame.shape
					h, w, _ = frame.shape
					grey = self.gray(frame)
					blur = self.gauss(grey)
					"""
					try:
						isTraficLight = self.traffic_light(copy_frame)
					except:
						isTraficLight = True
						print("ITS OKAY")
					"""
					if isTraficLight == True:
						if isRedLight == True:
							msg = {'action': '1', 'speed': 0.12}
							for outP in outPs:
								outP.send(msg)
							isRedLight = False
					if isTraficLight == False and is_priority == False:
						isRedLight = True
						msg = {'action': '1', 'speed': 0.00}
						for outP in outPs:
							outP.send(msg)
					else:
						
						try:
							detecSighn = self.compareImage(frame)
						except Exception as e:
							print(e)
						print("%%%%%%%%%", detecSighn)
						
						if detecSighn == "parking":
							is_sign_clasified = True
							sign = 1
						elif detecSighn == "prvenstvo":
							is_sign_clasified = True
							sign = 0
						elif detecSighn == "stop":
							is_sign_clasified = True
							sign = 2
						else:
							is_sign_clasified = False
							sign = 3
					"""
						try:
							if  is_sign_clasified == False:
								blur_signs = cv2.GaussianBlur(copy_frame, (27, 27), 0)
								#gray_signs = blur_signs[:, :, 0]
								gray_signs = cv2.cvtColor(blur_signs, cv2.COLOR_RGB2GRAY)
								gray_signs = cv2.adaptiveThreshold(gray_signs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)
								height_signs = round(height_signs / 4) 
								width_signs = round(4 * width_signs / 5) - 20
								
								for i in range(0, h):
									for j in range(0, w):
										if i > height_signs or j < width_signs:
											gray_signs[i][j] = 1
								
								gray_signs = cv2.bitwise_not(gray_signs)
								contours, hierarchy = cv2.findContours(gray_signs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
								contours_founded = []
								for contour in contours:  # za svaku konturu
									center, sizeS, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
									width_signs, height_signs = sizeS
									if width_signs > 25 and width_signs < 90 and height_signs > 35 and height_signs < 90 and abs(height_signs-width_signs) < 40:  # uslov da kontura pripada znaku
										detected_frame = gray_signs
										center_height = round(center[0])
										center_width = round(center[1])
										new_width = round(width_signs/2)
										new_height = round(height_signs/2)
										detected_frame = copy_frame[center_width-new_width:new_width + center_width, center_height-new_height:center_height + new_height]


										detected_frame = self.increase_brightness(detected_frame)
										
										print(wd, hd, "yrghushruhfu")
										is_sign_clasified, sign = self.clasificate_img(detected_frame)
										if is_sign_clasified:
											is_sign_clasified = False
										else:
											is_sign_clasified = True

										contours_founded.append(contour)  # ova kontura pripada
										break
																
								cv2.drawContours(copy_frame, contours_founded, -1, (255, 0, 0), 1)
							else:
								is_sign_clasified = False
						except:
							print("NO SIGN")
					"""
					edges = cv2.Canny(blur, 50, 150)
					isolated = self.region(edges)
					lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 70, np.array([]), minLineLength=35, maxLineGap=5)
					print("Prije detekcije")
					print("Sign: ", sign)
					print(isStop)
					print(inParking)
					inParking = 1
					parkiraj_se = False
					if inParking == 1 and parkiraj_se == False:
						"""print("__________________111111111111 ", inParkingTime)
						if inParkingTime == 0:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
						#if inParkingTime == 1:
						#	msg = {'action': '2', 'steerAngle': 0.0}
						#	for outP in outPs:
						#		outP.send(msg)
						if 0< inParkingTime < 30:
							msg = {'action': '1', 'speed': 0.12}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 30:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 30 < inParkingTime < 83:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 83:
							msg = {'action': '1', 'speed': 0.0}
							for outP in outPs:
								outP.send(msg)
								#flag = 0
						if inParkingTime == 86:
							msg = {'action': '2', 'steerAngle': 22.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 86 < inParkingTime < 113 :
							msg = {'action': '1', 'speed': -0.12}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 113:
							msg = {'action': '2', 'steerAngle': -22.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 113 < inParkingTime < 134:
							msg = {'action': '1', 'speed': -0.12}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 134:
							msg = {'action': '1', 'speed': 0.0 }
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 135:
							msg = {'action': '2', 'steerAngle': 18.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 135 < inParkingTime < 139:
							msg = {'action': '1', 'speed': 0.12}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 139 <= inParkingTime < 144:
							msg = {'action': '1', 'speed': 0.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if  inParkingTime == 144:
							msg = {'action': '2', 'steerAngle': -22.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 144 < inParkingTime < 160:
							msg = {'action': '1', 'speed': 0.12}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 160: #154
							msg = {'action': '2', 'steerAngle': 0.0}
							inParkingTime = 0
							inParking = 0
							parkiraj_se = True
							mozes_prvenstvo = True
							for outP in outPs:
								outP.send(msg)
								flag = 0
						inParkingTime += 1"""
						self.polEnc += self.encIm
						print("in parkingggggggggggggggggggggggg")
						if self.polEnc < 50:
							msg = {'action': '2', 'steerAngle': 20.0}
							for outP in outPs:
								outP.send(msg)
						if self.polEnc < 1000:
							print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", self.polEnc)
							msg = {'action': '1', 'speed': 0.12}
							for outP in outPs:
								outP.send(msg)
					elif is_priority == True and mozes_prvenstvo == True:
						self.ObstacleID = 2
						if time_p < 42:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
						if 42 <= time_p < 92:
							msg = {'action': '2', 'steerAngle': -22.0}
							for outP in outPs:
								outP.send(msg)
						if time_p == 92:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
						if 92 < time_p  < 110:
							msg = {'action': '1', 'speed': 0.12}
							for outP in outPs:
								outP.send(msg)
						if  time_p == 110:
							is_priority = False
							mozes_prvenstvo = False
						time_p += 1
						print("########: ",time_p)
					else:
						if  sign == 2 :
							isStop = False
							print("************************")
						if isStop == False and ne_radi_stop == False:
							self.ObstacleID = 1
							time = time + 1
							print("Time: ", time)
							if time < 20:
								print("U if-u")
								msg = {'action': '1', 'speed': 0.0}
								for outP in outPs:
									print("salje")
									outP.send(msg)
									flag = 0
							elif time == 10:
								msg = {'action': '2', "steerAngle": 0.0}
								for outP in outPs:
									outP.send(msg)
							elif 10 < time < 18: #raskrsnica
								msg = {'action': '1', 'speed': 0.12}
								for outP in outPs:
									outP.send(msg)
									flag = 0
							elif  18 <= time < 19:
								msg = {'action': '2', 'steerAngle': 22.0}
								for outP in outPs:
									outP.send(msg)
							elif time == 20:
								msg = {'action': '2', 'steerAngle': 0.0}
								for outP in outPs:
									outP.send(msg)
							else:
								isStop = True
								ne_radi_stop = True
								sign = -1
								msg = {'action': '1', 'speed': 0.12}
								for outP in outPs:
									outP.send(msg)
									flag = 0
							print("-----------------------------", isStop)
						elif sign == 1:
							self.ObstacleID = 3
							inParking = 1
							flag = 0
						elif sign == 0:
							is_priority = True
						else:
							if lines is None:
								isDetected = False
							else:
								averaged_lines, isDetected, direction = self.average(copy_frame, lines)					
								#print(direction)
							if isDetected:
								black_lines = self.display_lines(copy_frame, averaged_lines)
								lanes = cv2.addWeighted(copy_frame, 0.8, black_lines, 1, 1)
								if prev == 4 and pick_left_line >= 2 and ignore_left_line == False:
									msg = {'action': '2', 'steerAngle': 18.0}
									prev = -100
									pick_left_line = 0
								else:
									if direction == 1:
										msg = {'action': '2', 'steerAngle': -22.0}
										ignore_left_line =  True
									elif direction == 2:
										if isRight == 1:
											msg = {'action': '2', 'steerAngle': 22.0}
											ignore_left_line = False
										else:
											msg = {'action': '2', 'steerAngle': 0.0}
									elif direction == 4:
										if pick_left_line < 2:
											msg = {'action': '2', 'steerAngle': -18.0}
											pick_left_line = pick_left_line + 1
									else:
										msg = {'action': '2', 'steerAngle': 0.0}
										ignore_left_line = False
										isRight = 0
									prev = direction
								for outP in  outPs:
									outP.send(msg)
									flag = 0
							else:
								lanes = copy_frame
								print("NO lANES")
								print(prev)
								if prev == 2:
									msg = {'action': '2', 'steerAngle': 20.0} #AKO JE SKRETAO LEVO I NE VIDI LINIJU, NASTAVI DA SKRECES LEVO DOK NE VIDIS LINIJU
									isRight = 1
								elif prev == 1:
									msg = {'action': '2', 'steerAngle': -22.0} #AKO JE SKRETAO DESNO I NE VIDI LINIJU, NASTAVI DA SKRECES DESNO DOK NE VIDIS LINIJU
								for outP in outPs:
									outP.send(msg)
									flag = 0
				else:
					stamps, frame = inP.recv()
					#lanes = frame
					flag = 1
				try:
					result, image = cv2.imencode('.jpg', lanes, encode_param)
					data   =  image.tobytes()
					size   =  len(data)
					#print("))))))))))))))))))))", size)
					self.connection.write(struct.pack("<L",size))
					self.connection.write(data)
				except Exception as e:
					print("except ", e)
					self.connection = None
					self._init_socket()
					#pass
			except Exception as e:
				print("EXCEPT")
				self.connection = None
				self._init_socket()
				#pass

	def stop(self):
		vehicle.stop()
		Semaphores.stop()
		print("stoppppppppp line")
		msg = {'action': '1', 'speed': 0.0}
		for outP in self.outPs:
			outP.send(msg)
			print(msg)
		super(LineDetection,self).stop()  
