from threading import Thread
from src.templates.workerprocess import WorkerProcess
import cv2
import numpy as np
import time
import socket
import struct

class TraficSignDetection(WorkerProcess):
	def __init__(self, inPs, outPs):
		
		super(TraficSignDetection, self).__init__(inPs, outPs)
		
	def run(self):
		self._init_socket()
		super(TraficSignDetection, self).run()
		
	def _init_socket(self):
		"""Initialize the socket client. 
		"""
		self.serverIp   =  '192.168.0.101' # PC ip
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

		
	def _init_threads(self):
		print("\n Trafic thread inited \n")
		if self._blocker.is_set():
			return 
		StreamTh = Thread(name='TraficDetectionThread', target = self._send_thread, args= (self.inPs[0], self.outPs))
		StreamTh.daemon = True
		self.threads.append(StreamTh)
		
	def increase_brightness(self, img, value = 30):
		hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
		h, s, v = cv2.split(hsv)
		
		lim = 255 - value
		v[v > lim] = 255
		v[v <= lim] += value
		
		final_hsv = cv2.merge((h, s, v))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
		return img
	
	def calculate_average_color(self, image):
		height, width, _ = image.shape
		img = image[20:height - 20, 20:width - 20]
		avg_color_row = np.average(img, axis=0)
		avg_color = np.average(avg_color_row, axis=0)
		return avg_color


	def clasificate_img(self, avg_color):
		hsv = cv2.cvtColor(avg_color, cv2.COLOR_BGR2HSV)
		
		h, s, v = cv2.split(hsv)
		avg = np.average(h)
		print("**************************************")
		tolerence = 5
		print("AVG = ", avg)
		
		if 40 - tolerence < avg < 40 + tolerence:
			print("PRVENSTVO PROLAZE")
			return True
		elif 109 - tolerence < avg < 109 + tolerence:
			print("PESACKI")
			return True
		else:
			print("======")
			return False
		"""red = avg_color[0]
		green = avg_color[1]
		blue = avg_color[2]

		red_procentage_pp = False
		green_procentage_pp = False
		blue_procentage_pp = False

		tolerance = 10
		# Prvenstvo prolaza [149.5625  92.61    40.0075] 
		if 149 - tolerance < red < 149+ tolerance:
			red_procentage_pp = True
		if 92 - tolerance < green < 92 + tolerance:
			green_procentage_pp = True
		if 40 - tolerance < blue < 40 + tolerance:
			blue_procentage_pp = True

		red_procentage_stop = False
		green_procentage_stop = False
		blue_procentage_stop = False

		# STOP  [133.9775  48.96    59.02  ]
		if 129 - tolerance < red < 129 + tolerance:
			red_procentage_stop = True
		if 45 - tolerance < green < 45 + tolerance:
			green_procentage_stop = True
		if 55 - tolerance < blue < 55 + tolerance:
			blue_procentage_stop = True

		red_procentage_pjesacki = False
		green_procentage_pjesacki = False
		blue_procentage_pjesacki = False

		# Pjesacki  [106.3125  90.6375  96.9 ]
		if 107 - tolerance < red < 107 + tolerance:
			red_procentage_pjesacki = True
		if 90 - tolerance < green < 90 + tolerance:
			green_procentage_pjesacki = True
		if 96 - tolerance < blue < 96 + tolerance:
			blue_procentage_pjesacki = True

		red_procentage_parking = False
		green_procentage_parking = False
		blue_procentage_parking = False

		#Parking [103.1175  95.4175 121.2925]
		if 103 - tolerance < red <  103 + tolerance:
			red_procentage_parking =  True
		if 95 - tolerance < green < 95 + tolerance:
			green_procentage_parking = True
		if 121 - tolerance < blue < 121 + tolerance:
			blue_procentage_parking = True

		if red_procentage_pp and green_procentage_pp and blue_procentage_pp:
			print("PRVENSTVO PROLAZA")
			return  True
		elif red_procentage_stop and green_procentage_stop and blue_procentage_stop:
			print("STOP")
			return  True
		elif red_procentage_pjesacki and green_procentage_pjesacki and blue_procentage_pjesacki:
			print("PJESACKI")
			return  True
		elif red_procentage_parking and green_procentage_parking and blue_procentage_parking:
			print("PARKING")
			return True
		else:
			print("============")
			return False
		"""
		

			
	def _send_thread(self, inP, outPs):
		is_sign_clasified = False
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
		while True:
			try:
				print("++++++++++++")
				stamps, frame = inP.recv()
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				#cv2.imshow("framerc", frame)
				#cv2.waitKey(1)
				copy_frame = frame.copy()
				copy_frame =  cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)	
				
				if  is_sign_clasified == False:
					
					height, width, _ = frame.shape
					h, w, _ = frame.shape
					frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
					blur = cv2.GaussianBlur(frame, (15, 15), 0)
					gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
					gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 0)
					#cv2.imshow("framerc",gray)
					#cv2.waitKey(1)
					
					# add padding on image - crop image
					height = round(height / 3)
					width = round(width / 2)
					for i in range(0, h):
						for j in range(0, w):
							if i > height or j < width:
								gray[i][j] = 255

					#gray = cv2.bitwise_not(gray)
					
					contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
					# cv2.drawContours(copy_frame, contours, -1, (255, 0, 0), 1)
					
					contours_founded = []
					for contour in contours:  # za svaku konturu
						center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
						width, height = size
						if width > 20 and width < 90 and height > 20 and height < 90:  # uslov da kontura pripada bar-kodu
							detected_frame = gray
							center_height = round(center[0])
							center_width = round(center[1])
							detected_frame = copy_frame[center_width - 20:center_width + 20, center_height - 20:center_height + 20]
							#cv2.imshow("framerc",detected_frame)
							#cv2.waitKey(1)

							detected_frame = self.increase_brightness(detected_frame)
							#cv2.imshow("framerc",detected_frame)
							#cv2.waitKey(1)

							#avg = self.calculate_average_color(detected_frame)
							#print(avg)
							is_sign_clasified = self.clasificate_img(detected_frame)
							
							if is_sign_clasified:
								is_sign_clasified = False
							else:
								is_sign_clasified = True

							contours_founded.append(contour)  # ova kontura pripada
							break
						
					cv2.drawContours(copy_frame, contours_founded, -1, (255, 0, 0), 1)
					
				else:
					is_sign_clasified = False
				try:
					result, image = cv2.imencode('.jpg', copy_frame, encode_param)
					data   =  image.tobytes()
					size   =  len(data)
					self.connection.write(struct.pack("<L",size))
					self.connection.write(data)
					#print("line send")
				except Exception as e:
					self.connection = None
					self._init_socket()
					pass
				
				
			except:
				print("No Stream")
				self.connection = None
				#self._init_socket()
				pass
