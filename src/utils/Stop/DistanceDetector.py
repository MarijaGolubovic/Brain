from src.templates.workerprocess import WorkerProcess
import pigpio 
import time

import sys
sys.path.append('.')
import RTIMU
import os.path
import time
import math

import numpy as np

class Distance(WorkerProcess):
	def __init__(self, inPs, outPs):
		self.pi = pigpio.pi()
		self.GPIO_TRIGER1  = 13
		self.GPIO_ECHO1  = 16
		self.pi.set_mode(self.GPIO_TRIGER1, pigpio.OUTPUT)
		self.pi.set_mode(self.GPIO_ECHO1, pigpio.INPUT)
		
		self.GPIO_TRIGER2  = 6
		self.GPIO_ECHO2  = 5
		self.pi.set_mode(self.GPIO_TRIGER2, pigpio.OUTPUT)
		self.pi.set_mode(self.GPIO_ECHO2, pigpio.INPUT)
		
		self.GPIO_TRIGER3  = 12
		self.GPIO_ECHO3  = 18
		self.pi.set_mode(self.GPIO_TRIGER3, pigpio.OUTPUT)
		self.pi.set_mode(self.GPIO_ECHO3, pigpio.INPUT)
		
		self.SETTINGS_FILE = "RTIMULib"
		print("Using settings file " + self.SETTINGS_FILE + ".ini")
		if not os.path.exists(self.SETTINGS_FILE + ".ini"):
			print("Settings file does not exist, will be created")
		self.s = RTIMU.Settings(self.SETTINGS_FILE)
		self.imu = RTIMU.RTIMU(self.s)
		print("IMU Name: " + self.imu.IMUName())
		if (not self.imu.IMUInit()):
			print("IMU Init Failed")
			self.stop()
			sys.exit(1)
		else:
			print("IMU Init Succeeded")
		self.imu.setSlerpPower(0.02)
		self.imu.setGyroEnable(True)
		self.imu.setAccelEnable(True)
		self.imu.setCompassEnable(True)

		

		self.poll_interval = self.imu.IMUGetPollInterval()
		print("Recommended Poll Interval: %dmS\n" % self.poll_interval)
		
		super(Distance, self).__init__(inPs, outPs)
	
	def distance(self, dist, GPIO_TRIGER,  GPIO_ECHO):
		distance = 0
		
		self.pi.write(GPIO_TRIGER,1)
		time.sleep(0.00001)
		self.pi.write(GPIO_TRIGER,0)
		
		StartTime = time.time()
		StopTime = time.time()
		
		pom = 0
		while self.pi.read(GPIO_ECHO) == 0:
			StartTime = time.time()
			pom = pom + 1
			if pom == 25:
				pom = 0
				break
			
		while self.pi.read(GPIO_ECHO) == 1:
			StopTime = time.time()
			pom = pom + 1
			if pom == 25:
				pom = 0
				break
			

		TimeElapsed = StopTime - StartTime
		distance = TimeElapsed * 34300 / 2
		
		if distance < 0 :
			"""
			myArr = np.nonzero(dist)
			for k in myArr:
				dista[k] = dist[k]
			distance = np.average(dista)"""
			distance = np.average(dist)
			
		return distance
		
	def run(self):
		self._init_mesure()
		super(Distance, self).run()
		
	def imuMesurment(self):
		#print("555555555")
		if self.imu.IMURead():
				self.data = self.imu.getIMUData()
				self.fusionPose = self.data["fusionPose"]
				self.accel = self.data["accel"]
				self.roll  =  math.degrees(self.fusionPose[0])
				self.pitch =  math.degrees(self.fusionPose[1])
				self.yaw   =  math.degrees(self.fusionPose[2])
				self.accelx =  self.accel[0]
				self.accely =  self.accel[1]
				self.accelz =  self.accel[2]
				print("roll = %f pitch = %f yaw = %f" %(self.roll, self.pitch, self.yaw))
				time.sleep(self.poll_interval * 1.0 / 1000.0)
				return self.pitch
	
	def _init_mesure(self):
		block = 0;
		num_of_ditections = 0
		com_num = 0
		while True:
			pommsg = 0
			flagBokDis = True
			command = self.inPs[0].recv()
			if command == {'action': '1', 'speed': -0.12}:
				flagBokDis = False
			try:
				pitch = self.imuMesurment()
				#pitch = 0
			except Exception as e:
				pitch = 0
				print("IMU EXCEPTION: ", E)
			if num_of_ditections == 10:
				if 0 <= com_num < 8:
					com_num = com_num +1
					command = {'action': '1', 'speed': -0.13}
				if com_num > 7 and com_num < 11:
					com_num = com_num +1
					command = {'action': '1', 'speed': 0.0}
				if com_num == 11:
					com_num = com_num + 1
					command = {'action': '2', 'steerAngle': -22.0}
				elif com_num > 11 and com_num < 22:
					com_num = com_num + 1
					command = {'action': '1', 'speed': 0.13}
				elif com_num == 22:
					com_num = com_num + 1
					command = {'action': '2', 'steerAngle': 0.0}
				elif com_num > 22 and com_num < 33:
					com_num = com_num + 1
					command = {'action': '1', 'speed': 0.13}
				elif  com_num == 33:
					com_num = com_num + 1
					command = {'action': '2', 'steerAngle': 22.0}
				elif com_num > 33 and com_num < 62:
					command = {'action': '1', 'speed': 0.13}
					com_num = com_num + 1
				elif com_num == 62 :
					command = {'action': '2', 'steerAngle': -22.0}
					com_num = com_num + 1
				elif com_num > 62 and com_num < 67:
					command = {'action': '1', 'speed': 0.13}
					com_num = com_num + 1
				elif com_num == 67:
					command = {'action': '2', 'steerAngle': 0.0}
					com_num = 0 
					num_of_ditections = 0
					pommsg = 3
				else:
					print("Ne definisano stanje je ", com_num)
			
			elif pitch < -10:
				command = {'action': '1', 'speed': 0.09}
				pommsg = 4
			elif pitch > 10:
				command = {'action': '1', 'speed': 0.15}
			else :
				try:
					dist = np.zeros(4)
					for i in range (0, 4):
						dist[i] = self.distance(dist, self.GPIO_TRIGER2, self.GPIO_ECHO2)
					dista1 = np.average(dist)
					#time.sleep(0.2)
					print("Udaljenost dva je = %.1f cm" % dista1)
				except Exception as e:
					print("Greska u senozru 2 !!!!!!!", e)
				try:
					dist = np.zeros(4)
					for i in range (0, 4):
						dist[i] = self.distance(dist, self.GPIO_TRIGER3, self.GPIO_ECHO3)
					dista2 = np.average(dist)
					#time.sleep(0.2)
					print("Udaljenost je tri = %.1f cm" % dista2)
				except Exception as e:
					print("Greska u senozru 3 !!!!!!!", e)
				if dista1 < 10 or dista2 < 10:
					pommsg = 1
				else:
					pommsg = 0
					
				
			
				try:
					dist = np.zeros(4)
					for i in range (0, 4):
						dist[i] = self.distance(dist, self.GPIO_TRIGER1, self.GPIO_ECHO1)
					dista = np.average(dist)
					time.sleep(0.2)
					print("Udaljenost je = %.1f cm" % dista)
					if dista < 25 :
						block = 1
						pommsg = 2
						num_of_ditections = num_of_ditections + 1
						command = {'action': '1', 'speed': 0.00}
					else :
						if block == 1:
							command = {'action': '1', 'speed': 0.12}
						block = 0;
						#pommsg = 0
						num_of_ditections = 0
					#print(command)
					"""for outP in self.outPs:
						outP.send(command)"""
					#print("55555555")
				except Exception as e:
					block = 0
					print("Greska u senzoru 1!!!!!!!!!!! ",e)
					#pommsg = 0
					#print("*****************")
			for outP in self.outPs:
				outP.send(command)
			for outP in self.inPs:
				outP.send(pommsg)
			print("SEND DATA IS : ",pommsg)
	def stop(self):
		self.pi.stop()
