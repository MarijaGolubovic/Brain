from src.templates.workerprocess import WorkerProcess
import cv2
import socket
import struct
#import matplotlib.pyplot as plt
import numpy as np
import time
from threading import Thread

class LineDetection(WorkerProcess):
	flag = 0
	def __init__(self, inPs, outPs):
		self.flag = 0
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
		#print(image.shape)
		#polygon = np.array([[(0, 439), (475, 140), (width, 439)]])
		#mask = np.zeros_like(image)
		#mask = cv2.fillPoly(mask, polygon, 255)
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
		#print('dl', lines)
		lines_image = np.zeros_like(image)
		if lines is not None:
			for line in lines:
				#print('dl for ',line)
				x1, y1, x2, y2 = line.reshape(4)
				#x2, y2 = line.reshape(2)
				#print('x2 y2 ', x2, y2)
				cv2.line(lines_image, (x1, y1),(x2, y2), (255, 0, 0), 10)
				#cv2.line(lines_image, (x2, y2), (255, 0, 0), 10)
		#print("display lines")
		return lines_image
		
	def average(self, image, lines):
		#print("avg start")
		left = []
		right = []
		if lines is None:
			#print("lines is none")
			return None
		for line in lines:
			#print('line ', line)
			x1, y1, x2, y2 = line.reshape(4)
			#print(x1, x2, y1, y2)
			parameters = np.polyfit((x1, x2), (y1, y2), 1)
			#print('par ',parameters)
			slope = parameters[0]
			y_int = parameters[1]
			if slope < 0 :
				left.append((slope, y_int))
			else:
				right.append((slope, y_int))
		try:
			right_avg = np.average(right, axis=0)
			left_avg = np.average(left, axis=0)
			# create lines based on averages calculates
			left_line = self.make_points(image, left_avg)
			right_line = self.make_points(image, right_avg)
			return np.array([right_line])
		except:
			print("no line")
		
	def make_points(self, image, average):
		#print('make points')
		slope, y_int = average
		y1 = image.shape[0]
		y2 = int(y1*(0.6))
		x1 = int((y1 - y_int)//slope)
		x2 = int((y2 - y_int)//slope)
		#print('p ', x1, x2, y1, y2)
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
		self.serverIp   =  '192.168.112.83' # PC ip
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
		while True:
			try:
				#cv2.waitKey(1)
				#print("radi nesto \n")
				stamps, frame = inP.recv()
				#print(frame)
				#cv2.imshow('frame', frame)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()
				grey = self.gray(frame)
				#cv2.imshow('bin', grey)
				#cv2.waitKey(1)
				blur = self.gauss(grey)
				edges = cv2.Canny(blur, 50, 150)
				isolated = self.region(edges)
				lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 70, np.array([]), minLineLength=40, maxLineGap=5)
				#print(lines(
				
				try:
					#print("tryfff")
					averaged_lines = self.average(frame, lines)
					#print('al', averaged_lines)
					black_lines = self.display_lines(frame, averaged_lines)
					#cv2.imshow('lines', black_lines);
					#cv2.waitKey(1)
					# taking wighted sum of original image and lane lines image
					lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
					msg = {'action': '2', 'steerAngle': 0.0}
					for outP in  outPs:
						outP.send(msg)
				except:
					lanes = frame
					print("NO lANES")
					#for i in range(0,10):
					msg = {'action': '2', 'steerAngle': 22.0}
					for outP in outPs:
						outP.send(msg)
				"""
				if self.flag == 1:
					print("++++++++++++++++")
					msg = {'action': '1', 'speed': 0.0}
				for outP in outPs:
					outP.send(msg)
					print(msg)
				"""
				#print("ne mogu da nacrtam")
				#cv2.imshow('lane', lanes)
				#cv2.waitKey(1)
				#result, image = cv2.imencode('.jpg', lanes, encode_param)
				#data = lanes.tobytes()
				#size = len(data)

				#self.connection.write(struct.pack("<L",size))
				#self.connection.write(data)
				
				#result.write(lanes)
				#for outP in self.outPs:
				#	print("poslao")
				#	outP.send([lanes])
				try:
					result, image = cv2.imencode('.jpg', lanes, encode_param)
					data   =  image.tobytes()
					size   =  len(data)
					self.connection.write(struct.pack("<L",size))
					self.connection.write(data)
					#print("line send")
				except Exception as e:
					#print("CameraStreamer failed to stream images:",e,"\n")
					# Reinitialize the socket for reconnecting to client.  
					self.connection = None
					self._init_socket()
					pass
				"""
				try:
					while True:

						# decode image
						image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
						bts = self.connection.read(image_len)

						# ----------------------- read image -----------------------
						image = np.frombuffer(bts, np.uint8)
						image = cv2.imdecode(image, cv2.IMREAD_COLOR)
						image = np.reshape(image, self.imgSize)
						image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

						# ----------------------- show images -------------------
						cv2.imshow('Image', image) 
						cv2.waitKey(1)
				"""
				
			except Exception as e:
				#print("CameraStreamer failed to stream images:",e,"\n")
				# Reinitialize the socket for reconnecting to client.  
				self.connection = None
				#self._init_socket()
				pass
				
				
				"""
				frame = np.frombuffer(frame, dtype=np.uint8)
				frame = np.reshape(frame, (480, 640,3))
				for outP in self.outPs:
					outP.send([[stamps],frame])
				"""
	def stop(self):
		print("stoppppppppp line")
		msg = {'action': '1', 'speed': 0.0}
		for outP in self.outPs:
			outP.send(msg)
			print(msg)
#self.inPs[0] = msg
#writeTh = WriteThread(self.inPs[0], self.serialCom, self.historyFile)
#self.threads.append(writeTh) 
		super(LineDetection,self).stop()  
