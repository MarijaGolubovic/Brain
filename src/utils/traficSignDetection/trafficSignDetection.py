from threading import Thread
from src.templates.workerprocess import WorkerProcess
import cv2
import numpy as np
import time

class TraficSignDetection(WorkerProcess):
	def __init__(self, inPs, outPs):
		
		super(TraficSignDetection, self).__init__(inPs, outPs)
		
	def run(self):
		super(TraficSignDetection, self).run()
		
	def _init_threads(self):
		print("\n Trafic thread inited \n")
		if self._blocker.is_set():
			return 
		StreamTh = Thread(name='TraficDetectionThread', target = self._send_thread, args= (self.inPs[0], self.outPs))
		StreamTh.daemon = True
		self.threads.append(StreamTh)
		
	def calculate_average_color(self, image):
		height, width, _ = image.shape
		img = image[20:height - 20, 20:width - 20]
		avg_color_row = np.average(img, axis=0)
		avg_color = np.average(avg_color_row, axis=0)
		return avg_color


	def clasificate_img(self, avg_color):
		red = avg_color[0]
		green = avg_color[1]
		blue = avg_color[2]

		red_procentage_pp = False
		green_procentage_pp = False
		blue_procentage_pp = False

		tolerance = 10
		# Prvenstvo prolaza [87.195 78.28  58.645]
		if 87 - tolerance < red < 87+ tolerance:
			red_procentage_pp = True
		if 78 - tolerance < green < 78 + tolerance:
			green_procentage_pp = True
		if 58 - tolerance < blue < 58 + tolerance:
			blue_procentage_pp = True

		red_procentage_stop = False
		green_procentage_stop = False
		blue_procentage_stop = False

		# STOP [20.19333333 21.26777778 53.42333333]
		if 20 - tolerance < red < 20 + tolerance:
			red_procentage_stop = True
		if 21 - tolerance < green < 21 + tolerance:
			green_procentage_stop = True
		if 53 - tolerance < blue < 53 + tolerance:
			blue_procentage_stop = True

		red_procentage_pjesacki = False
		green_procentage_pjesacki = False
		blue_procentage_pjesacki = False

		# Pjesacki [86.775 75.335 59.34 ]
		if 140 - tolerance < red < 140 + tolerance:
			red_procentage_pjesacki = True
		if 123 - tolerance < green < 123 + tolerance:
			green_procentage_pjesacki = True
		if 107 - tolerance < blue < 107 + tolerance:
			blue_procentage_pjesacki = True


		if red_procentage_pp and green_procentage_pp and blue_procentage_pp:
			print("PRVENSTVO PROLAZA")
			return  True
		elif red_procentage_stop and green_procentage_stop and blue_procentage_stop:
			print("STOP")
			return  True
		elif red_procentage_pjesacki and green_procentage_pjesacki and blue_procentage_pjesacki:
			print("PJESACKI")
			return  True
		else:
			print("============")
			return False


			
	def _send_thread(self, inP, outPs):
		is_sign_clasified = False
		while True:
			try:
				
				stamps, frame = inP.recv()
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				cv2.imshow("framerc", frame)
				cv2.waitKey(1)
				copy_frame = frame.copy()
				
				if  is_sign_clasified == False:
					
					height, width, _ = frame.shape
					h, w, _ = frame.shape
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
					#print("**************************************")
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
							detected_frame = copy_frame[center_width - 30:center_width + 30, center_height - 30:center_height + 30]
							#cv2.imshow("framerc",detected_frame)
							#cv2.waitKey(1)

							avg = self.calculate_average_color(detected_frame)
							print(avg)
							is_sign_clasified = self.clasificate_img(avg)
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
				print("No Stream")
