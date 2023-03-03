from src.templates.workerprocess import WorkerProcess
import pigpio 
import time

class Distance(WorkerProcess):
	def __init__(self, inPs, outPs):
		self.pi = pigpio.pi()
		self.GPIO_TRIGER  = 13
		self.GPIO_ECHO  = 16
		self.pi.set_mode(self.GPIO_TRIGER, pigpio.OUTPUT)
		self.pi.set_mode(self.GPIO_ECHO, pigpio.INPUT)
		super(Distance, self).__init__(inPs, outPs)
	
	def distance(self):
		distance = -1
		self.pi.write(self.GPIO_TRIGER,1)
		time.sleep(0.00001)
		self.pi.write(self.GPIO_TRIGER,0)
		StartTime = 0
		StopTime = 0
		pom = 0
		while self.pi.read(self.GPIO_ECHO) == 0:
			StartTime = time.time()
			pom = pom + 1
			if pom == 15:
				pom = 0
				break
			
		while self.pi.read(self.GPIO_ECHO) == 1:
			StopTime = time.time()
		if StartTime != 0 and StopTime != 0 :
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
			try:
				
				dis = self.distance()
				time.sleep(0.2)
				print("Udaljenost je = %.1f cm" % dis)
				if dis < 20 and dis > 0:
					block = 1;
					command = {'action': '1', 'speed': 0.00}
					#for outP in self.outPs:
						#outP.send(command)
				else :
					if block == 1:
						command = {'action': '1', 'speed': 0.09}
					#for outP in self.outPs:
						#outP.send(command)
					block = 0;
				for outP in self.outPs:
					outP.send(command)
			except:
				for outP in self.outPs:
					outP.send(command)
				block = 0
				print("*****************")
	def stop(self):
		self.pi.stop()
