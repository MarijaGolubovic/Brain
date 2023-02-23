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
		self.serverIp   =  '192.168.0.102' # PC ip
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
		hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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
		#print("**************************************")
		tolerance = 3
		print("AVG: ", avg)
		
		if 104 - tolerance < avg < tolerance + 104:
			print("PRVENSTVO PROLAZA")
			return True
		#elif 75 - tolerance < avg < tolerance + 75:
		#	print("PJESACKI")
		#	return True
		elif 62 - tolerance < avg < tolerance + 62:
			print("PARKING")
			return True
		elif 115 - tolerance < avg < tolerance + 115:
			print("STOP")
			return True
		else:
			print("======")
			return False


	def _send_thread(self, inP, outPs):
		is_sign_clasified = False
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
		while True:
			try:
				#print("++++++++++++")
				stamps, frame = inP.recv()
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				#cv2.imshow("framerc", frame)
				#cv2.waitKey(1)
				copy_frame = frame.copy()
				copy_frame =  cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)	
				if  is_sign_clasified == False:
					
					height, width, _ = frame.shape
					h, w, _ = frame.shape
					#frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
					blur = cv2.GaussianBlur(frame, (27, 27), 0)
					#blur = self.increase_brightness(blur)
					#gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
					gray = blur[:, :, 0]
					gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)
					#cv2.imshow("framerc",gray)
					#cv2.waitKey(1)
					
					# add padding on image - crop image
					height = round(height / 4) 
					width = round(4 * width / 5)
					for i in range(0, h):
						for j in range(0, w):
							if i > height or j < width:
								gray[i][j] = 1
					
					#gray_white = gray
					gray = cv2.bitwise_not(gray)
					#gray = cv2.bitwise_and(gray_white, gray_black)
					contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
					# cv2.drawContours(copy_frame, contours, -1, (255, 0, 0), 1)
					
					contours_founded = []
					for contour in contours:  # za svaku konturu
						center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
						width, height = size
						#x, y, height, width = cv2.boundingRect(contour)
						if width > 35 and width < 90 and height > 30 and height < 90 and abs(height-width) < 20:  # uslov da kontura pripada znaku
							detected_frame = gray
							center_height = round(center[0])
							center_width = round(center[1])
							new_width = round(width/2)
							new_height = round(height/2)
							detected_frame = copy_frame[center_width-new_width:new_width + center_width, center_height-new_height:center_height + new_height]
							#cv2.imshow("framerc",detected_frame)
							#cv2.waitKey(1)

							detected_frame = self.increase_brightness(detected_frame)
							t = time.time
							#t = time.strftime(t)
							#local = '/home/schobot/test_image/img' + t + '.jpg'
							#cv2.imwrite(local, detected_frame)
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
