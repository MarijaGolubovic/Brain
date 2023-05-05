

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

class Speed(WorkerProcess):
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
		self.Dis = 0
		
		self.lines = None
		self.lanes = None
		self.copy_frame = None
		self.isRight = -1
		self.prev = -100
		self.pick_left_line = 0
		self.ignore_left_line = False
		
		self.Flag = True
		
		self.inSh = inSh

		

		
		super(Speed, self).__init__(inPs, outPs, inSh)
	
	def run(self):
		self._init_socket()
		super(Speed, self).run()
		
	
		
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
		
		EncTh = Thread(name='encoderThread', target = self._send_enc, args= ())
		EncTh.daemon = True
		self.threads.append(EncTh)
		
		MsgFrSe = Thread(name='msgFromSensors', target = self._rec_msg, args= ())
		MsgFrSe.daemon = True
		self.threads.append(MsgFrSe)
		
		
	 # ===================================== INIT SOCKET ==================================
	def _init_socket(self):
		"""Initialize the socket client. 
		"""
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

	

		
	def lane_keeping(self):
		idi_desno = -1
		isao_levo = -1
		idi_duze = False
		idi_duze_lijevo = False
		iterator = 0
		lijevo = 0
		msg = {'action': '1', 'speed': 0.35}
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
		
	def _rec_msg(self):
		while True:
			while True:
				self.Dis = self.outPs[0].recv()
				if self.Dis == 1:
					self.Flag = False
	
	def _send_enc(self):
		while True:
			self.encIm  = self.inSh[0].recv()
			try:
				self.encIm =  float(msg)
				#print("gjjwedhgiuf",self.polEnc)
			except:
				msg = ""
		
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
		detecSighn = "glupost"
		
		num_frame = 0
		msg = {'action': '1', 'speed': 0.35}
		for outP in outPs:
			outP.send(msg)
		encLenght = 0
		Flag_pom = True
		while True:
			try:
				msg = {'action': '1', 'speed': 0.35}
				for outP in outPs:
					outP.send(msg)
				if self.flag == 1:
					
					#msg = {"action": '1', 'speed': 0.09}
					print("!!!!!!!!!!!!!!!!!!!!!!!:  ", num_frame)
					stamps, frame = inP.recv()
					self.lanes = frame
					self.copy_frame = frame.copy()
					
					#copy_frame =  cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)	
					#height_signs, width_signs, _ = frame.shape
					#h, w, _ = frame.shape
					if self.Flag == True:
						if Flag_pom == True:
							encLenght = self.encIm + encLenght
							if encLenght  < 200:
								msg = {'action': '1', 'speed': 0.35}
								for outP in self.outPs:
									outP.send(msg)
								self.flag = 0
							elif 200 <= encLenght < 300:
								msg = {'action': '2', 'steerAngle': 22.0}
								for outP in self.outPs:
									outP.send(msg)
								self.flag = 0
								encLenght = 0
								Flag_pom = False
						else :
							grey = self.gray(frame)
							blur = self.gauss(grey)
					
							edges = cv2.Canny(blur, 50, 150)
							isolated = self.region(edges)
							self.lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 70, np.array([]), minLineLength=35, maxLineGap=5)
							self.lane_keeping()
					elif self.Flag == False and self.Dis == 0:
						num_frame = num_frame + 1
						encLenght = self.encIm + encLenght
						print("encLenght:  ", encLenght)
						grey = self.gray(frame)
						blur = self.gauss(grey)
						edges = cv2.Canny(blur, 50, 150)
						isolated = self.region(edges)
						self.lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 70, np.array([]), minLineLength=35, maxLineGap=5)
						if num_frame == 1: 
							msg = {'action': '1', 'speed': 0.0}
							for outP in self.outPs:
								outP.send(msg)
							self.flag = 0
						elif num_frame == 2:
							msg = {'action': '1', 'speed': 0.35}
							for outP in self.outPs:
								outP.send(msg)
							self.flag = 0
						else:
							if encLenght < 700:
								self.lane_keeping()
							elif 700 <= encLenght < 1150:
								msg = {'action': '2', 'steerAngle': -22.0}
								for outP in self.outPs:
									outP.send(msg)
								self.flag = 0
							elif 1150 <= encLenght < 2600:
								msg = {'action': '2', 'steerAngle': 0.0}
								for outP in self.outPs:
									outP.send(msg)
								self.flag = 0
							elif 2600 <= encLenght < 3700:
								self.lane_keeping()
							else:
								msg = {'action': '1', 'speed': 0.0}
								for outP in self.outPs:
									outP.send(msg)
								self.flag = 0
					else:
						msg = {'action': '1', 'speed': 0.35}
						for outP in self.outPs:
							outP.send(msg)
						self.flag = 0
							
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
