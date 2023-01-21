import src.data.trafficlights.trafficlights as trafficlights
import time
from threading import Thread
from src.templates.workerprocess import WorkerProcess

class TraficDetector(WorkerProcess):
	def __init__(self, inPs, outPs):
		super(TraficDetector, self).__init__(inPs, outPs)
	
	def run(self):
		#self._init_socket()
		super(TraficDetector, self).run()
		
	def _init_threads(self):
		print("\n LaneDet thread inited \n")
		if self._blocker.is_set():
			return 
		TrafTh = Thread(name='LaneDetectionThread', target = self._send_thread, args= (self.inPs,self.outPs))
		TrafTh.daemon = True
		self.threads.append(TrafTh)
	
	def _send_thread(self,inPs,outPs):
		colors = ['red','yellow','green']   
		# Get time stamp when starting tester
		start_time = time.time()
		# Create listener object
		Semaphores = trafficlights.trafficlights()
		# Start the listener
		Semaphores.start()
		# Wait until 60 seconds passed
		while (time.time()-start_time < 60):
		# Clear the screen
#print("\033c")
			print("Example program that gets the states of each\nsemaphore from their broadcast messages\n")
			# Print each semaphore's data
			print("S1 color " + colors[Semaphores.s1_state] + ", code " + str(Semaphores.s1_state) + ".")
			print("S2 color " + colors[Semaphores.s2_state] + ", code " + str(Semaphores.s2_state) + ".")
			print("S3 color " + colors[Semaphores.s3_state] + ", code " + str(Semaphores.s3_state) + ".")
			print("S4 color " + colors[Semaphores.s4_state] + ", code " + str(Semaphores.s4_state) + ".")
			time.sleep(0.5)
			if Semaphores.s1_state == 1:
				command = {'action': '1', 'speed': 0.30}
			else :
				command = {'action': '1', 'speed': 0.00}
			for outP in outPs:
				outP.send(command)
		Semaphores.stop()
