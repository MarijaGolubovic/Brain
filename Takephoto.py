from picamera2 import Picamera2
from libcamera import controls
import os
import cv2
import numpy as np

picam = Picamera2()

picam.resolution = (1640,1232)
picam.framerate = 25
picam.brightness = 55
picam.shutter_speed = 1200
picam.iso = 0

camera_con = picam.create_still_configuration(main = {"size": (640,480)}, lores = {"size": (640,480)}, display = "lores")
picam.configure(camera_con)

signs = ["prvenstvo", "jednosmjerna", "stop", "parking", "autoput", "pjesacki", "kraj_autoputa", "kruzni", "obavezno_pravo"]

for i in range(0,9):
	print(signs[i])
	input("Prees to take photo...")
	picam.start()
	picam.capture_file("test.jpg")
	data = cv2.imread("test.jpg")
	data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

	picam.stop()

	data = np.frombuffer(data, dtype = np.uint8)
	data = np.reshape(data,(480,640,1))
	cv2.imwrite("imgs/"+signs[i]+".png",data)
	h, w, _ = data.shape
	data = data[0:round(h/3),round(3*w/4)+15:w, :]
	#data = data[20:round(h/3)-20,round(3*w/4)+40:w-20, :]
	
	data =cv2.resize(data, (80, 80), interpolation = cv2.INTER_AREA)
	#print(data)
	cv2.imwrite("imgs/"+signs[i]+"cut"+".png",data)
	
	#cv2.waitKey(0)


"""
os.system("v4l2-ctl --set-ctrl wide_dynamic_range=1 -d /dev/v4l-subdev0")
print("Setting HDR to ON")
picam2.start(show_preview=True)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfSpeed": controls.AfSpeedEnum.Fast})
picam2.start_and_capture_files("HDRfastfocus{:d}.jpg", num_files=1, delay=1)
picam2.stop_preview()
picam2.stop()
print("Setting HDR to OFF")
os.system("v4l2-ctl --set-ctrl wide_dynamic_range=0 -d /dev/v4l-subdev0")
"""
