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
		self.GPIO_TRIGER  = 13
		self.GPIO_ECHO  = 16
		self.pi.set_mode(self.GPIO_TRIGER, pigpio.OUTPUT)
		self.pi.set_mode(self.GPIO_ECHO, pigpio.INPUT)
		
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
	
	def distance(self, dist):
		distance = 0
		
		self.pi.write(self.GPIO_TRIGER,1)
		time.sleep(0.00001)
		self.pi.write(self.GPIO_TRIGER,0)
		
		StartTime = time.time()
		StopTime = time.time()
		
		pom = 0
		while self.pi.read(self.GPIO_ECHO) == 0:
			StartTime = time.time()
			pom = pom + 1
			if pom == 25:
				pom = 0
				break
			
		while self.pi.read(self.GPIO_ECHO) == 1:
			StopTime = time.time()
			

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
		while True:
			command = self.inPs[0].recv()
			pitch = self.imuMesurment()
			
			if pitch < -10:
				command = {'action': '1', 'speed': 0.09}
			elif pitch > 10:
				command = {'action': '1', 'speed': 0.09}
			else:
				try:
					dist = np.zeros(4)
					for i in range (0, 4):
						dist[i] = self.distance(dist)
					dista = np.average(dist)
					time.sleep(0.2)
					print("Udaljenost je = %.1f cm" % dista)
					if dista < 20 :
						block = 1;
						command = {'action': '1', 'speed': 0.05}
					else :
						if block == 1:
							command = {'action': '1', 'speed': 0.15}
						block = 0;
					#print(command)
					"""for outP in self.outPs:
						outP.send(command)"""
					#print("55555555")
				except:
					block = 0
					#print("*****************")
			for outP in self.outPs:
				outP.send(command)
	def stop(self):
		self.pi.stop()
