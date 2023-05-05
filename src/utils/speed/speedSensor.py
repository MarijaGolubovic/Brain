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

class SpeedSensor(WorkerProcess):
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
		
		super(SpeedSensor, self).__init__(inPs, outPs)

	def run(self):
		self._init_mesure()
		super(SpeedSensor, self).run()
		
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
			#print("RADI BILO STA ..............................")
			try:
				pitch = self.imuMesurment()
				#pitch = 0
			except Exception as e:
				pitch = 0
				print("IMU EXCEPTION: ", E)
			
			if pitch < -5:
				#command = {'action': '1', 'speed': 0.09}
				pommsg = 0
			elif pitch > 5:
				pommsg = 1
				#command = {'action': '1', 'speed': 0.15}
			
			for outP in self.outPs:
				outP.send(command)
			for outP in self.inPs:
				outP.send(pommsg)
			print("SEND DATA IS : ",pommsg)
	def stop(self):
		self.pi.stop()
