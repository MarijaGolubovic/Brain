import sys
sys.path.append('.')
import RTIMU
import os.path
import time
import math
from threading import Thread
from src.templates.workerprocess import WorkerProcess

class imu(WorkerProcess):
    def __init__(self, inPs, outPs): 
        #Thread.__init__(self)
        self.running = True
        
        #print(sys.path)

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
        
        super(imu, self).__init__(inPs, outPs)
    
    def _init_threads(self):
        if self._blocker.is_set():
            return
 
        StreamTh = Thread(name='IMUDetectionThread', target = self._send_thread, args= ())
        StreamTh.daemon = True
        self.threads.append(StreamTh)
        
    def run(self):
        super(imu,self).run()
        
    def _send_thread(self):
        while self.running == True:
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
                
                
                print("yaw = %f" % (self.yaw))
            #print("x = %f y = %f z = %f" %((self.accelx,self.accely,self.accelz)))
            #f.close()
                time.sleep(self.poll_interval*1.0/10.0)

    def stop(self): 
        self.running = False
        super(imu,self).stop()
