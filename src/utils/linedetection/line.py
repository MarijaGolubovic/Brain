from src.templates.workerprocess import WorkerProcess
import cv2
import socket
import struct
#import matplotlib.pyplot as plt
import numpy as np
import time
from threading import Thread

class LineDetection(WorkerProcess):
	def __init__(self, inPs, outPs):
		super(LineDetection, self).__init__(inPs, outPs)
	
	def run(self):
		self._init_socket()
		super(LineDetection, self).run()
		
	def gray(self, image):
		im = np.asarray(image)
		gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
		ret, thresh1 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return thresh1
		
	def gauss(self, image):
		return cv2.GaussianBlur(image, (5,5), 0)
		
	def Canny(self, image):
		edges = cv2.Canny(image, 100 , 200)
		return edges
	
	def region(self, image):
		height, width = image.shape
		triangle = np.array([[(0, height), (475, 140), (width, height)]])
		mask = np.zeros_like(image)
		mask = cv2.fillPoly(mask, triangle, 255)
		isreg = cv2.bitwise_and(image, mask)
		return isreg	
		
	def display_lines(self, image, lines):
		lines_image = np.zeros_like(image)
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line
				cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
		return lines_image
		
	def average(self, image, lines):
		left = []
		right = []
		if lines is not None:
			for line in lines:
				#print(line)
				x1, y1, x2, y2 = line.reshape(4)
				parameters = np.polyfit((x1, x2), (y1, y2), 1)
				#print(parameters)
				slope = parameters[0]
				y_int = parameters[1]
				if slope <0 :
					left.append((slope, y_int))
				else:
					right.append((slope, y_int))
		right_avg = np.average(right, axis=0)
		left_avg = np.average(left, axis=0)
		left_line = self.make_points(image, left_avg)
		right_line = self.make_points(image, right_avg)
		return np.array([left_line, right_line])
		
	def make_points(self, image, average):
		#print(average)
		slope, y_int = average
		y1 = image.shape[0]
		y2 = int(y1*(3/5))
		x1 = int((y1 - y_int)//slope)
		x2 = int((y2 - y_int)//slope)
		return np.array(([x1, y1, x2, y2]))
		
	def _init_threads(self):
		print("\n LaneDet thread inited \n")
		if self._blocker.is_set():
			return 
		StreamTh = Thread(name='LaneDetectionThread', target = self._send_thread, args= (self.inPs[0], ))
		StreamTh.daemon = True
		self.threads.append(StreamTh)
		
	 # ===================================== INIT SOCKET ==================================
	def _init_socket(self):
		"""Initialize the socket client. 
		"""
		self.serverIp   =  '192.168.112.199' # PC ip
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

		
	def _send_thread(self, inP):
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
				edges = cv2.Canny(blur, 50, 100)
				isolated = self.region(edges)
				lines = cv2.HoughLinesP(isolated, 1, np.pi/180, 10, np.array([]), minLineLength=3, maxLineGap=10)
				if lines is not None:
					averaged_lines = self.average(frame, lines)
					black_lines = self.display_lines(frame, averaged_lines)
					lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
				else:
					lanes = frame
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
				except Exception as e:
					print("CameraStreamer failed to stream images:",e,"\n")
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
				print("CameraStreamer failed to stream images:",e,"\n")
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
