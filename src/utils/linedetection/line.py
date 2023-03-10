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
		h_min = int(height/2) + 50
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
			#print("((((((((((((((((((((((((((((((((((((", final_list)
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
		
		if 5 - tolerance < avg < tolerance - 10:
			print("PRVENSTVO PROLAZA")
			return True, 0
		#elif 75 - tolerance < avg < tolerance + 75:
		#	print("PJESACKI")
		#	return True
		elif 77 - tolerance < avg < tolerance + 77:
			print("PARKING")
			return True, 1
		elif 118 - tolerance < avg < tolerance + 118:
			print("STOP")
			return True, 2
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
		elif 170 - tolerance < avg < 170 + tolerance:
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
		
	 # ===================================== INIT SOCKET ==================================
	def _init_socket(self):
		"""Initialize the socket client. 
		"""
		self.serverIp   =  '192.168.30.149' # PC ip
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
		parkiraj_se =  False
		msg = {'action': '1', 'speed': -0.2}
		while True:
			try:
				if flag == 1:
					#msg = {"action": '1', 'speed': 0.09}
					stamps, frame = inP.recv()
					lanes = frame
					copy_frame = frame.copy()
					#copy_frame =  cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)	
					height_signs, width_signs, _ = frame.shape
					h, w, _ = frame.shape
					grey = self.gray(frame)
					blur = self.gauss(grey)
					try:
						isTraficLight = self.traffic_light(copy_frame)
					except:
						isTraficLight = True
						print("ITS OKAY")
					if isTraficLight == True:
						if isRedLight == True:
							msg = {'action': '1', 'speed': 0.09}
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

					edges = cv2.Canny(blur, 50, 150)
					isolated = self.region(edges)
					lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 70, np.array([]), minLineLength=35, maxLineGap=5)
					print("Prije detekcije")
					print("Sign: ", sign)
					print(isStop)
					print(inParking)
					if inParking == 1 and parkiraj_se == False:
						print("__________________111111111111 ", inParkingTime)
						if inParkingTime == 0:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
						if inParkingTime == 1:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
						if 1 < inParkingTime < 32:
							msg = {'action': '1', 'speed':- 0.2}
							for outP in outPs:
								outP.send(msg)
								#flag = 0
						if inParkingTime == 32:
							msg = {'action': '1', 'speed': 0.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 33:
							msg = {'action': '2', 'steerAngle': 22.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 33 < inParkingTime < 47:
							msg = {'action': '1', 'speed': -0.09}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 47:
							msg = {'action': '2', 'steerAngle': -22.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 47 < inParkingTime < 56:
							msg = {'action': '1', 'speed': -0.09}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 56:
							msg = {'action': '1', 'speed': 0.0 }
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 57:
							msg = {'action': '2', 'steerAngle': 18.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if 57 < inParkingTime < 59:
							msg = {'action': '1', 'speed': 0.09}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						if inParkingTime == 59:
							inParkingTime = 0
							inParking = 0
							parkiraj_se = True
							msg = {'action': '1', 'speed': 0.0}
							for outP in outPs:
								outP.send(msg)
								flag = 0
						inParkingTime += 1
					elif is_priority == True:
						if time_p < 14:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
						if 14 <= time_p < 36:
							msg = {'action': '2', 'steerAngle': -22.0}
							for outP in outPs:
								outP.send(msg)
						if time_p == 36:
							msg = {'action': '2', 'steerAngle': 0.0}
							for outP in outPs:
								outP.send(msg)
						if  time_p == 37:
							is_priority = False
						time_p += 1
						print("########: ",time_p)
					else:
						if  sign == 2 :
							isStop = False
							print("************************")
						if isStop == False:
							time = time + 1
							print("Time: ", time)
							if time < 10:
								print("U if-u")
								msg = {'action': '1', 'speed': 0.0}
								for outP in outPs:
									print("salje")
									outP.send(msg)
									flag = 0
							elif 10 <= time < 37: #raskrsnica
								msg = {'action': '1', 'speed': 0.09}
								for outP in outPs:
									outP.send(msg)
									flag = 0
							else:
								isStop = True
								sign = -1
								msg = {'action': '1', 'speed': 0.09}
								for outP in outPs:
									outP.send(msg)
									flag = 0
							print("-----------------------------", isStop)
						elif sign == 1:
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
		print("stoppppppppp line")
		msg = {'action': '1', 'speed': 0.0}
		for outP in self.outPs:
			outP.send(msg)
			print(msg)
		super(LineDetection,self).stop()  
