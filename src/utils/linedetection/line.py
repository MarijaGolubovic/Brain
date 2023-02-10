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
		h_max = height-1
		for i in range(h_min, h_max):
			#print(i)
			for j in range (0, width-1):
				#print("j: ", j)
				mask[i][j] = 255
		isreg = cv2.bitwise_and(image, mask)
		return isreg	
		
	def display_lines(self, image, lines):
		lines_image = np.zeros_like(image)
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line.reshape(4)
				cv2.line(lines_image, (x1, y1),(x2, y2), (255, 0, 0), 10)
		return lines_image
		
	def average(self, image, lines):
		left = []
		right = []
		if lines is None:
			return None
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			parameters = np.polyfit((x1, x2), (y1, y2), 1)
			slope = parameters[0]
			y_int = parameters[1]
			if slope < 0 :
				left.append((slope, y_int))
			else:
				right.append((slope, y_int))
		try:
			right_avg = np.average(right, axis=0)
			left_avg = np.average(left, axis=0)
			left_line = self.make_points(image, left_avg)
			right_line = self.make_points(image, right_avg)
			return np.array([right_line])
		except:
			print("no line")
		
	def make_points(self, image, average):
		slope, y_int = average
		y1 = image.shape[0]
		y2 = int(y1*(0.6))
		x1 = int((y1 - y_int)//slope)
		x2 = int((y2 - y_int)//slope)
		return np.array(([x1, y1,  x2, y2]))
		
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
		self.serverIp   =  '192.168.100.83' # PC ip
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
		while True:
			try:
				if flag == 1:
					print("flag: ", flag)
					stamps, frame = inP.recv()
					grey = self.gray(frame)
					blur = self.gauss(grey)
					edges = cv2.Canny(blur, 50, 150)
					isolated = self.region(edges)
					lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 70, np.array([]), minLineLength=40, maxLineGap=5)
					try:
						averaged_lines = self.average(frame, lines)
						black_lines = self.display_lines(frame, averaged_lines)
						lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
						msg = {'action': '2', 'steerAngle': 0.0}
						for outP in  outPs:
							outP.send(msg)
							flag = 0
					except:
						lanes = frame
						print("NO lANES")
						msg = {'action': '2', 'steerAngle': 22.0}
						for outP in outPs:
							outP.send(msg)
							flag = 0
				else:
					print("ne radi nista ", flag)
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
