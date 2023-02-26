from src.templates.workerprocess import WorkerProcess
import cv2
import socket
import struct
#import matplotlib.pyplot as plt
import numpy as np
import time
from threading import Thread

class LineDetection(WorkerProcess):
	flag = 1
	def __init__(self, inPs, outPs):
		self.flag = 1
		super(LineDetection, self).__init__(inPs, outPs)
	
	def run(self):
		self._init_socket()
		super(LineDetection, self).run()
		
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
		h_min = int(height/2)
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
		#right_line = np.array(([0, 0,  0, 0]))
		if lines is None:
			return final_list, line_det
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			parameters = np.polyfit((x1, x2), (y1, y2), 1)
			slope = parameters[0]
			y_int = parameters[1]
			if slope < 0 :
				left.append((slope, y_int))
			else:
				right.append((slope, y_int))
		if right != []:
			right_avg = np.average(right, axis=0)
			slope, y_int = right_avg
			#print("_______________", slope)
			if abs(slope) > 0.5:
				right_line = self.make_points(image, right_avg)
				#print(right_line)
				final_list.append(right_line)
				line_det = True
		if left != []:
			left_avg = np.average(left, axis=0)
			slope, y_int = left_avg
			if abs(slope) > 0.5:
				left_line = self.make_points(image, left_avg)
				final_list.append(left_line)
				line_det = True
		try:
			final_list = np.array(final_list)
		except:
			print("cannot convert")
		return final_list, line_det
		
	def make_points(self, image, average):
		slope, y_int = average
		#if abs(slope) > 0.5:
		y1 = int(image.shape[0]*0.5)
		y2 = int(image.shape[0]*0.75)
		x1 = int((y1 - y_int)//slope)
		x2 = int((y2 - y_int)//slope)
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
		
	def _init_threads(self):
		print("\n LaneDet thread inited \n")
		if self._blocker.is_set():
			return 
		StreamTh = Thread(name='LaneDetectionThread', target = self._send_thread, args= (self.inPs[0], self.outPs))
		StreamTh.daemon = True
		self.threads.append(StreamTh)
		
	 # ===================================== INIT SOCKET ==================================
	def _init_socket(self):
		"""Initialize the socket client. 
		"""
		self.serverIp   =  '192.168.100.149' # PC ip
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

		
	def _send_thread(self, inP, outPs):
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
		flag = 1
		is_sign_clasified = False
		while True:
			try:
				if flag == 1:
					stamps, frame = inP.recv()
					copy_frame = frame.copy()
					#copy_frame =  cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)	
					height_signs, width_signs, _ = frame.shape
					h, w, _ = frame.shape
					grey = self.gray(frame)
					blur = self.gauss(grey)
					
					try:
						if  is_sign_clasified == False:
							blur_signs = cv2.GaussianBlur(frame, (27, 27), 0)
							gray_signs = blur_signs[:, :, 0]
							gray_signs = cv2.adaptiveThreshold(gray_signs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)
							height_signs = round(height_signs / 4) 
							width_signs = round(4 * width_signs / 5) - 20
							
							for i in range(0, h):
								for j in range(0, w):
									if i > height_signs or j < width_signs:
										gray_signs[i][j] = 1
							
							gray_signs = cv2.bitwise_not(gray_signs)
							contours, hierarchy = cv2.findContours(gray_signs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
							contours_founded = []
							for contour in contours:  # za svaku konturu
								center, sizeS, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
								width_signs, height_signs = sizeS
								if width_signs > 35 and width_signs < 90 and height_signs > 30 and height_signs < 90 and abs(height_signs-width_signs) < 20:  # uslov da kontura pripada znaku
									detected_frame = gray_signs
									center_height = round(center[0])
									center_width = round(center[1])
									new_width = round(width_signs/2)
									new_height = round(height_signs/2)
									detected_frame = copy_frame[center_width-new_width:new_width + center_width, center_height-new_height:center_height + new_height]


									detected_frame = self.increase_brightness(detected_frame)

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
					except:
						print("NO SIGN")
					
					edges = cv2.Canny(blur, 50, 150)
					isolated = self.region(edges)
					lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 70, np.array([]), minLineLength=40, maxLineGap=5)
					averaged_lines, isDetected = self.average(copy_frame, lines)
					black_lines = self.display_lines(copy_frame, averaged_lines)
					lanes = cv2.addWeighted(copy_frame, 0.8, black_lines, 1, 1)
					if isDetected: #and averaged_lines != None:
						msg = {'action': '2', 'steerAngle': 0.0}
						for outP in  outPs:
							outP.send(msg)
							flag = 0
					else:
						lanes = copy_frame
						print("NO lANES")
						msg = {'action': '2', 'steerAngle': 22.0}
						for outP in outPs:
							outP.send(msg)
							flag = 0
				else:
					stamps, frame = inP.recv()
					flag = 1
				try:
					result, image = cv2.imencode('.jpg', lanes, encode_param)
					data   =  image.tobytes()
					size   =  len(data)
					self.connection.write(struct.pack("<L",size))
					self.connection.write(data)
					#print("line send")
				except Exception as e:
					self.connection = None
					self._init_socket()
					pass
			except Exception as e:
				self.connection = None
				#self._init_socket()
				pass

	def stop(self):
		print("stoppppppppp line")
		msg = {'action': '1', 'speed': 0.0}
		for outP in self.outPs:
			outP.send(msg)
			print(msg)
		super(LineDetection,self).stop()  
