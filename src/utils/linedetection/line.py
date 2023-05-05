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
		
		self.enableTraficLightsServer = True
		self.enableServerV2V = False
		self.enableLiveTraficServer = False
		self.enableLocalizationServer = False
		self.MyXcord = 0
		self.Myycord = 0   
		self.TraficLightSr = -1
		self.CarXCord = 0
		self.CarYcord = 0
		self.ObstacleID = 0
		self.encIm = 0
		self.polEnc = 0
		
		#################################################### OBAVEZNO PROVJERITI ###########################################################
		self.can_be_parking = True
		self.TraficLightSrSecond = -1
		self.can_be_priority = False
		self.down_hill = False
		
		self.lines = None
		self.lanes = None
		self.copy_frame = None
		self.isRight = -1
		self.prev = -100
		self.pick_left_line = 0
		self.ignore_left_line = False
		
		self.Dis = 0

		self.inSh = inSh
		#self.inDis = inDis
		
		path = "imgs/"
		stop = cv2.imread(path+"stopcut.png")
		stop1 = cv2.imread(path+"stopcut1.png")
		stop2 = cv2.imread(path+"stopcut2.png")
		
		parking = cv2.imread(path+"parkingcut.png")
		parking1 = cv2.imread(path+"parkingcut1.png")
		parking2 = cv2.imread(path+"parkingcut2.png")
		
		#jednosmjerna = cv2.imread(path+"jednosmjernacut.png")
		#prvenstvo = cv2.imread(path+"prvenstvocut.png")
		#pjesacki = cv2.imread(path+"pjesackicut.png")
		#autoput = cv2.imread(path+"autoputcut.png")
		#kraj_autoputa = cv2.imread(path+"kraj_autoputacut.png")
		#kruzni = cv2.imread(path+"kruznicut.png")
		#obavezno_pravo = cv2.imread(path+"obavezno_pravocut.png")
		
		self.blue = []
		
		#blue_signs = ["parking", "pjesacki", "obavezno_pravo", "kruzni", "jednosmjerna", "stop","prvenstvo", "autoput", "kraj_autoputa" ]
		self.blue.append(parking)
		self.blue.append(parking1)
		self.blue.append(parking2)
		
		self.blue.append(stop)
		self.blue.append(stop1)
		self.blue.append(stop2)
		
		#self.blue.append(kruzni)
		#self.blue.append(jednosmjerna)
		#self.blue.append(prvenstvo)
		#self.blue.append(autoput)
		#self.blue.append(kraj_autoputa)
		
		
		super(LineDetection, self).__init__(inPs, outPs, inSh)
	
	def run(self):
		self._init_socket()
		super(LineDetection, self).run()
		
	def crop_image(self, img):
		#print("**********USAO **********", img.shape)
		hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		#print("**************	NIJE PROSAO**************88*")
		h, s, v = cv2.split(hsv)
		avg = np.average(  h)
		
		return avg

	def whatId(self, sign):
		if sign == "parking":
			self.ObstacleID = 3
		elif sign == "pjesacki":
			if  self.Dis == 2:
				self.ObstacleID = 12
			else:
				self.ObstacleID = 4
		elif sign == "obavezno_pravo":
			self.ObstacleID = 8
		elif sign == "kruzni":
                        self.ObstacleID = 7
		elif sign == "jednosmjerna":
                        self.ObstacleID = 15
		elif sign == "stop":
                        self.ObstacleID = 1
		elif sign == "prvenstvo":
                        self.ObstacleID = 2
		elif sign == "autoput":
                        self.ObstacleID = 5
		elif sign == "kraj_autoputa":
                        self.ObstacleID = 6
		elif self.Dis == 1:
			self.ObstacleID = 11
		elif self.Dis == 2:
			self.ObstacleID = 12
	def compareImage(self, imageIn):
		h, w, _ = imageIn.shape
		imgC = imageIn[0:round(h/3),round(3*w/4):w, :]
		CopyImg = imgC.copy()
		#print("******************OVDJE UDJE*********************")
		try:
			imgC = cv2.cvtColor(imgC, cv2.COLOR_RGB2GRAY)
			#print("******************OVDJE UDJE*********************")
		except Exception as e:
			print("CVT COLOR IN COMPARE IMAGE: ", e)
		detected_frame, have_contour, avg = self.FindContures(imgC, CopyImg)
		imgC =cv2.resize(detected_frame, (80, 80), interpolation = cv2.INTER_AREA)
		print("============ AVG ============", avg)
		
		index = -1
		res = []
		
		blue_signs = ["parking", "parking", "parking", "stop", "stop", "stop"]
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
			"""if index == 5 and max(res) >  0.15:
				#self.whatId(blue_signs[index])
				return blue_signs[index]
			if(max(res) < 0.20):
				return 0
			if index == 6:
				if 40 - tolerance <= avg <= 40 + tolerance:
					self.whatId(blue_signs[index])
					return blue_signs[index]
				else:
					return 0
			"""
			
			if(max(res) < 0.15):
				return 0 
				
			if index == 3 or index == 4 or index == 5:
				if 115 < avg < 135:
					return blue_signs[5]
				else:
					return 0
			else:
				if 145 < avg < 175:
					return blue_signs[1]
				else:
					return 0
			#print("$$$$$$$$$", blue_signs[index])
			#self.whatId(blue_signs[index])
			
		else:
			return 0


		
	def FindContures(self, imgIn, copy_frame):
		try:
			copy_frame = cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)
			#print("***********PITAMO SE DA LI OVDJE UDJE**********")
		except Exception as e:
			print("CVT COLOR IN FIND CONTOURS: ", E)
		blur = cv2.GaussianBlur(imgIn, (27,27), 0)
		
		thresh = cv2.adaptiveThreshold(blur,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)
		thresh = cv2.bitwise_not(thresh)
		#print("***********PITAMO SE DA LI OVDJE UDJE**********")
		#cv2.imshow("blur1", thresh)
		#cv2.waitKey(1)
		detected_frame = copy_frame.copy()
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours_founded = []
		have_contour = False
		#print("***********PITAMO SE DA LI OVDJE UDJE**********")
		for contour in contours:  # za svaku konturu
			center, sizeS, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
			width_signs, height_signs = sizeS
			aditional_condition = round(height_signs/2)
			aditional_condition_width = round(width_signs/2)
			if width_signs > 25 and width_signs < 125 and height_signs > 35 and  height_signs < 125 and center[0] > aditional_condition and center[1] > aditional_condition_width: #and abs(height_signs-width_signs) < 40:  #uslov da kontura pripada znaku
				detected_frame = thresh
				center_height = round(center[0])
				center_width = round(center[1])
				new_width = round(width_signs/2) 
				new_height = round(height_signs/2)
				#print(center_height, center_width, new_height,new_width, height_signs, width_signs )
				detected_frame = copy_frame[center_width-new_width:new_width + center_width, center_height-new_height:center_height + new_height]
				contours_founded.append(contour)
				have_contour = True
				#cv2.imshow("slika", detected_frame)
				#cv2.waitKey(1)
				#print("ceawhnfuh",contours_founded)
				t = time.time()
				"""try:
					cv2.imwrite("imgs/imgs/img"+str(t)+".png",detected_frame)
				except Exception as e:
					print(e)
				"""
		#print("***********PITAMO SE DA LI OVDJE UDJE**********")
		cv2.drawContours(copy_frame, contours_founded, -1, (255, 0, 0), 1)
		#print("**************8COUNTUR FOUNDID *****************", detected_frame.shape, len(contours_founded))
		avg = self.crop_image(detected_frame)
		#print("~~~~~~~~~~~~~~~",avg,"~~~~~~~~~~~~~~~")

		#cv2.imshow("blur", detected_frame)
		#cv2.waitKey(1)
		try:
			#print("******SHAPE OF DETECTED FRAME ******", detected_frame.shape)
			ret = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2GRAY)
			#print("***********PITAMO SE DA LI OVDJE UDJE**********")
		except Exception as e:
			print("CVT COLOR IN DETECTED FRAME: ", e)
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
		h_min = int(height/2) + 50 
		h_max = int(height*3/4) + 50
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
			#print("ddddddddddddddddddddddddddd", slope)
			y_int = parameters[1]
			if slope < 0 :
				left.append((slope, y_int))
			else:
				right.append((slope, y_int))
		if left != []:
			left_avg = np.average(left, axis=0)
			slope, y_int = left_avg
			print("IDEM DESNO: ", slope)
			if abs(slope) > 0.02:
				#print("soooooooooooope:  ", slope)
				direction = 2
				left_line = self.make_points(image, left_avg)
				final_list.append(left_line)
				line_det = True
		elif right != []:
			right_avg = np.average(right, axis=0)
			slope, y_int = right_avg
			if abs(slope) > 0.5:
				right_line = self.make_points(image, right_avg)
				if (right_line[0] < int(image.shape[1]*3/5) and right_line[2] < int(image.shape[1]*3/5)):#IZMENJENO 3.5.
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
		#print(slope)
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
		
		StreamDs =Thread(name='ReadFromDistanceTread', target = self._sendDis, args = ())
		StreamDs.daemon = True
		self.threads.append(StreamDs)
		
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
		#self.serverIp   =  '192.168.88.78' # PC ip
		#self.serverIp   =  '192.168.1.224' # PC ip
		self.serverIp   =  '192.168.220.149' # PC ip
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

	def _sendDis(self):
		while True:
			self.Dis = self.outPs[0].recv()
			if self.Dis == 3:
				self.can_be_parking = True
			elif self.Dis == 4:
				self.down_hill = True

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
		while True:
		# Clear the screen
#print("\033c")
			#print("Example program that gets the states of each semaphore from their broadcast messages\n")
			# Print each semaphore's data
			#print("S1 color " + colors[Semaphores.s1_state] + ", code " + str(Semaphores.s1_state) + ".")
			#print("S2 color " + colors[Semaphores.s2_state] + ", code " + str(Semaphores.s2_state) + ".")
			#print("S3 color " + colors[Semaphores.s3_state] + ", code " + str(Semaphores.s3_state) + ".")
			#print("S4 color " + colors[Semaphores.s4_state] + ", code " + str(Semaphores.s4_state) + ".")
			self.TraficLightSr = Semaphores.s3_state #START
			self.TraficLightSrSecond = Semaphores.s1_state
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
		while True:
			# Clear the screen
			print("Example program that gets the info of the last car infos\n")
			# Print each received msg
			print("ID ", vehicle.ID, ", coor ", vehicle.pos)
			self.CarXCord = vehicle.pos.real
			self.CaryCord = vehicle.pos.imag
			#print("neduwhfviuw")
			time.sleep(1)
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
		
	def lane_keeping(self):
		idi_desno = -1
		isao_levo = -1
		idi_duze = False
		idi_duze_lijevo = False
		iterator = 0
		lijevo = 0
		msg = {'action': '1', 'speed': 0.12}
		#print("++++++++++++++++++++++++++++++U LANE KEEPING")
		if self.lines is None:
			isDetected = False
		else:
			#print(self.lines)
			averaged_lines, isDetected, direction = self.average(self.copy_frame, self.lines)
			#print(self.copy_frame.shape)
			#print(averaged_lines)
			print("#########################",direction)
		if isDetected:
			black_lines = self.display_lines(self.copy_frame, averaged_lines)
			#print(black_lines)
			self.lanes = cv2.addWeighted(self.copy_frame, 0.8, black_lines, 1, 1)
			print(self.lanes.shape)
			if self.prev == 4 and self.pick_left_line >= 3 and self.ignore_left_line == False:
				msg = {'action': '2', 'steerAngle': 18.0}
				self.prev = -100
				self.pick_left_line = 0
			else:
				if direction == 1:
					msg = {'action': '2', 'steerAngle': -9.0}
					isao_levo = 1
					idi_duze_lijevo = True
					self.ignore_left_line =  True
				elif direction == 2:
					if self.isRight == 1:
						msg = {'action': '2', 'steerAngle': 9.0}
						self.ignore_left_line = False
						idi_duze = True
						print("IDI DUZE IDI DUZE IDI DUZE IDI DUZE")
					else:
						msg = {'action': '2', 'steerAngle': 0.0}
				elif direction == 4:
					if isao_levo == 1:
						print("PROBLEEEEEEEEEEEEEEEEM")
						msg = {'action': '2', 'steerAngle': 22.0}
						#for outP in self.outPs:
						#	outP.send(msg)
						#time_start = time.time()
						#time_end = time.time()
						#print("----------TIME TIME TIME TIME TIME:", time_end - time_start)
						#while time_end -time_start < 0.20:
						#	time_end = time.time()
						isao_levo = 0
						#self.flag = 0 
					else:
						if self.pick_left_line < 3:
							msg = {'action': '2', 'steerAngle': -18.0}
							self.pick_left_line = self.pick_left_line + 1
				else:
					msg = {'action': '2', 'steerAngle': 0.0}
					self.ignore_left_line = False
					self.isRight = 0
			self.prev = direction
			if idi_duze == True:
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				time_start = time.time()
				time_end = time.time()
				print("----------TIME TIME TIME TIME TIME:", time_end - time_start)
				while time_end -time_start < 0.20:
					time_end = time.time()
				print("TIME TIME TIME TIME TIME:", time_end - time_start)
				iterator = -100
			if idi_duze_lijevo == True:
				time_start = time.time()
				time_end = time.time()
                            	#print("----------TIME TIME TIME TIME TIME:", time_end - time_start)
				while time_end -time_start < 0.20:
					time_end = time.time()
                                #print("TIME TIME TIME TIME TIME:", time_end - time_start)
				lijevo = -100
			if idi_duze == False:
				for outP in  self.outPs:
					outP.send(msg)
					self.flag = 0
			elif idi_duze_lijevo == True:
				if lijevo == -100:
					for outP in  self.outPs:
						outP.send(msg)
						self.flag = 0
						lijevo = False
			else:
				if iterator == -100:
					for outP in self.outPs:
						outP.send(msg)
						self.flag = 0
						idi_duze = False
		else:
			print("copy_frame: ",  self.copy_frame.shape)
			self.lanes = self.copy_frame
			print("NO lANES")
			print(self.prev)
			if self.prev == 2:
				msg = {'action': '2', 'steerAngle': 20.0} #AKO JE SKRETAO LEVO I NE VIDI LINIJU, NASTAVI DA SKRECES LEVO DOK NE VIDIS LINIJU
				self.isRight = 1
			elif self.prev == 1:
				msg = {'action': '2', 'steerAngle': -22.0} #AKO JE SKRETAO DESNO I NE VIDI LINIJU, NASTAVI DA SKRECES DESNO DOK NE VIDIS LINIJU
			else:
				msg = {'action': '2', 'steerAngle': 20.0}
			for outP in self.outPs:
				outP.send(msg)
				self.flag = 0
		
	def _send_thread(self, inP, outPs):
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
		#flag = 1
		is_sign_clasified = False
		isStop = True
		finish = False
		sign = -1
		time = 0
		isRight = -1
		inParking = -1
		isTraficLight = self.enableTraficLightsServer
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
		start = False
		ramp_pass = False
		detecSighn = "glupost"
		msg = {'action': '1', 'speed': 0.12}
		#for outP in outPs:
			#outP.send(msg)
		while True:
			try:
				if self.flag == 1:
					#msg = {"action": '1', 'speed': 0.09}
					stamps, frame = inP.recv()
					self.lanes = frame
					self.copy_frame = frame.copy()
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
					############################################ OBAVEZNO MJENJATI #####################################################
					self.TraficLightSr =2
					print("SERVER IS:", isTraficLight , "STANJE SEMAFORA" ,self.TraficLightSr)
					if isTraficLight == True:
						if self.TraficLightSr == 0:
							msg = {'action': '1', 'speed': 0.00}
							for outP in outPs:
								outP.send(msg)
						else:
							msg = {'action': '1', 'speed': 0.12}
							for outP in outPs:
								outP.send(msg)
							isTraficLight = False
							
						

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
						"""
					else:
						isTraficLight = False
						try:
							detecSighn = self.compareImage(self.copy_frame)
						except Exception as e:
							print("***ZNAKOVI: ",e)
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
					self.lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 80, np.array([]), minLineLength=25, maxLineGap=10)
					print("Prije detekcije")
					print("Sign: ", sign)
					print(isStop)
					print(inParking)
					#inParking = -100
					#parkiraj_se = False #ovim se bl;okira DA NE UPADA VISE PUTA U PARKING
					#tmp = -1
					is_priority = False
					if inParking == 1 and parkiraj_se == False and self.can_be_parking == True:
						print("||||||||||||||||||||||||||||||||||||||||||||", self.Dis)
						print("__________________111111111111 ", inParkingTime)
						if self.Dis != 0 and self.Dis != 1:
							self.Dis = 0
						if self.Dis == 0:
							if inParkingTime == 0:
								msg = {'action': '2', 'steerAngle': 0.0}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							#if inParkingTime == 1:
							#	msg = {'action': '2', 'steerAngle': 0.0}
							#	for outP in outPs:
							#		outP.send(msg)
							if 0< inParkingTime < 50:
								"""msg = {'action': '1', 'speed': 0.12}
								for outP in outPs:
									outP.send(msg)
									flag = 0"""
								#self.lane_keeping()
								#self.polEnc += self.encIm
								if self.polEnc < 750:
									self.lane_keeping()
								else:
									msg = {'action': '1', 'speed': 0.0}
									for outP in outPs:
										outP.send(msg)
									self.flag = 0
							if inParkingTime == 50:
								print("303030303030303030303030303030")
								msg = {'action': '2', 'steerAngle': 0.0}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							"""if 30 < inParkingTime < 83:
								msg = {'action': '2', 'steerAngle': 0.0}
								for outP in outPs:
									outP.send(msg)
									self.flag = 0"""
							if inParkingTime == 51:
								print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
								msg = {'action': '1', 'speed': 0.0}
								for outP in outPs:
									#print(msg)
									outP.send(msg)
								self.flag = 0
							if inParkingTime == 52:
								msg = {'action': '2', 'steerAngle': 22.0}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if 52 < inParkingTime < 73 :
								msg = {'action': '1', 'speed': -0.12}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if inParkingTime == 73:
								msg = {'action': '2', 'steerAngle': -22.0}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if 73 < inParkingTime < 83:
								msg = {'action': '1', 'speed': -0.12}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if inParkingTime == 83:
								msg = {'action': '1', 'speed': 0.0 }
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if inParkingTime == 84:
								msg = {'action': '2', 'steerAngle': 18.0}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if 84 < inParkingTime < 88:
								msg = {'action': '1', 'speed': 0.12}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if 88 <= inParkingTime < 94:
								msg = {'action': '1', 'speed': 0.0}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if  inParkingTime == 94:
								msg = {'action': '2', 'steerAngle': -22.0}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if 94 < inParkingTime < 100:
								msg = {'action': '1', 'speed': 0.12}
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							if inParkingTime == 100: #154
								msg = {'action': '2', 'steerAngle': 0.0}
								inParkingTime = 0
								inParking = 0
								self.can_be_parking = False
								parkiraj_se = True
								mozes_prvenstvo = True
								for outP in outPs:
									outP.send(msg)
								self.flag = 0
							inParkingTime += 1
							#print(msg)
							self.flag = 0
						elif self.Dis == 1 and inParkingTime < 35:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
							self.flag = 0
							#msg = {'action': '1', 'speed': 0.12}
							#for outP in outPs:
							#		outP.send(msg)
							#		self.flag = 0
						else:
							self.Dis = 0

					elif is_priority == True and mozes_prvenstvo == True and self.can_be_priority == True:
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
						if time_p == 110:
							if self.TraficLightSrSecond == 0:
								msg = {'action': '1', 'speed': 0.0}
								for outP in outPs:
									outP.send(msg)
							else:
								self.lane_keeping()
						if  time_p == 110:
							is_priority = False
							mozes_prvenstvo = False
							self.polEnc = 0
							finish = True
						self.flag = 0
						time_p += 1
						print("########: ",time_p)
					
					elif finish == True:
						print("!!!!!!IN FINISH: ", self.polEnc)
						self.polEnc += self.encIm
						if self.polEnc < 2600:
							self.lane_keeping()
						elif 2600 <= self.polEnc < 3100:
							msg = {'action': '2', "steerAngle": -22.0}
							for outP in outPs:
								outP.send(msg)
							self.flag = 0
						elif 3100 <= self.polEnc < 3300:
							self.lane_keeping()
						elif 3300 <= self.polEnc < 3400:
							if self.TraficLightSrSecond == 0:
								msg = {'action': '1', 'speed': 0.0}
								for outP in outPs:
									#print("salje")
									outP.send(msg)
								self.flag = 0
							else:
								msg = {'action': '1', 'speed': 0.12}
								for outP in outPs:
									#print("salje")
									outP.send(msg)
								self.lane_keeping()
						elif 3400 <= self.polEnc < 6400:
							self.lane_keeping()
						elif 6400 <= self.polEnc < 7100:
							msg = {'action': '2', "steerAngle": -22.0}
							for outP in outPs:
								outP.send(msg)
							self.flag = 0
						else:
							self.lane_keeping()
							finish = False
					
					else:
						if  sign == 2 :
							isStop = False
							print("************************")
						if isStop == False: # and ne_radi_stop == False:
							self.ObstacleID = 1
							time = time + 1
							print("Time: ", time)
							if time < 20:
								print("U if-u")
								msg = {'action': '1', 'speed': 0.0}
								for outP in outPs:
									print("salje")
									outP.send(msg)
									self.flag = 0
							elif time == 10:
								msg = {'action': '2', "steerAngle": 0.0}
								for outP in outPs:
									outP.send(msg)
							elif 10 < time < 18: #raskrsnica
								msg = {'action': '1', 'speed': 0.12}
								for outP in outPs:
									outP.send(msg)
									self.flag = 0
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
									self.flag = 0
							print("-----------------------------", isStop)
						elif sign == 1:
							self.ObstacleID = 3
							inParking = 1
							self.flag = 0
						elif sign == 0:
							is_priority = True
						elif self.down_hill == True and self.Dis == 0:
							self.polEnc = 0
							self.down_hill = False
							finish = True
						else:
							print("ELSE")
							self.lane_keeping()
							"""if lines is None:
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
									flag = 0"""
						
						
				else:
					stamps, frame = inP.recv()
					#lanes = frame
					self.flag = 1
				try:
					result, image = cv2.imencode('.jpg', self.lanes, encode_param)
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
				print("EXCEPT", e)
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
