from src.templates.workerprocess import WorkerProcess
import RPi.GPIO as GPIO
import time

class Distance(WorkerProcess):
	def __init__(self, inPs, outPs):
		GPIO.setmode(GPIO.BCM)
		self.GPIO_TRIGER  = 19
		self.GPIO_ECHO  = 26
		GPIO.setup(self.GPIO_TRIGER, GPIO.OUT)
		GPIO.setup(self.GPIO_ECHO, GPIO.IN)
		super(Distance, self).__init__(inPs, outPs)
	
	def distance(self):
		GPIO.output(self.GPIO_TRIGER,True)
		time.sleep(0.00001)
		GPIO.output(self.GPIO_TRIGER,False)
		StartTime = time.time()
		StopTime = time.time()
		while GPIO.input(self.GPIO_ECHO) == 0:
			StartTime = time.time()
		
		while GPIO.input(self.GPIO_ECHO) == 1:
			StopTime = time.time()
		
		TimeElapsed = StopTime - StartTime
		
		distance = TimeElapsed * 34300 / 2
		
		return distance
		
	def run(self):
		self._init_mesure()
		super(Distance, self).run()
		
	def _init_mesure(self):
		block = 0;
		while True:
			command = self.inPs[0].recv()
			dis = self.distance()
			time.sleep(0.1)
			#print("Udaljenost je = %.1f cm" % dis)
			if dis < 20:
				block = 1;
				command = {'action': '1', 'speed': 0.00}
				#for outP in self.outPs:
					#outP.send(command)
			else :
				if block == 1:
					command = {'action': '1', 'speed': 0.08}
				#for outP in self.outPs:
					#outP.send(command)
				block = 0;
			for outP in self.outPs:
				outP.send(command)
