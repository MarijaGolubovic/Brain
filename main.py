# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#========================================================================
# SCRIPT USED FOR WIRING ALL COMPONENTS
#========================================================================
import sys
sys.path.append('.')

import time
import signal
from src.data.trafficlights.TrafficLights import TraficDetector
from multiprocessing import Pipe, Process, Event 

# hardware imports
from src.hardware.camera.cameraprocess                      import CameraProcess
from src.hardware.camera.CameraSpooferProcess               import CameraSpooferProcess
from src.hardware.serialhandler.SerialHandlerProcess        import SerialHandlerProcess

# utility imports
from src.utils.camerastreamer.CameraStreamerProcess         import CameraStreamerProcess
from src.utils.remotecontrol.RemoteControlReceiverProcess   import RemoteControlReceiverProcess
from src.utils.linedetection.line                           import LineDetection

from src.utils.Stop.DistanceDetector                        import Distance 
# =============================== CONFIG =================================================
enableStream        =  True
enableCameraSpoof   =  False 
enableRc            =  False
enableData          =  False
# =============================== INITIALIZING PROCESSES =================================
allProcesses = list()
# rcSer, camSer = Pipe(duplex = false)      camera salje komande autu
# =============================== HARDWARE ===============================================
if enableStream:
    camStR, camStS = Pipe(duplex = False)           # camera  ->  line
    camLineStR, camLineSts = Pipe(duplex = False)       # line    ->  streamer

    if enableCameraSpoof:
        camSpoofer = CameraSpooferProcess([],[camStS],'vid')
        allProcesses.append(camSpoofer)

    else:
        camProc = CameraProcess([],[camStS])
        allProcesses.append(camProc)
    
    camLine = LineDetection([camStR],[])  
    allProcesses.append(camLine)
    #cv2.imshow(camLineStR.recv(), 'line')
    #camLine = CameralineFolow([camStR],[camSer])  salje komande na proces za serisuku komunikaciju
    #allProcess.append(camLine)
    #camSign = CameraDetevtSign([camStR],[camSer])
    #allProcess.append(camSigh)
    
    #streamProc = CameraStreamerProcess([camLineStR], [])
    #allProcesses.append(streamProc)
    #streamProc = CameraStreamerProcess([camStR], [])
    #allProcesses.append(streamProc)


# =============================== DATA ===================================================
#LocSys client process
if enableData:
    LocStR, LocStS = Pipe(duplex = False)
    # Semaphore colors list
        
    
       # if Semaphores.s1_state == 2:
        #    LocStS = {'action': '1', 'speed': 0.3}
        #else:
         #   locStS = {'action': '1', 'speed': 0.0}
        # Stop the listener
    
    #rcProc = RemoteControlReceiverProcess([],[LocStS])
    #allProcesses.append(rcProc)
    shProc = SerialHandlerProcess([LocStR], [])
    allProcesses.append(shProc)
    trafficlightProc = TraficDetector([],[LocStS])
    allProcesses.append(trafficlightProc)          # LocSys  ->  brain
   
# from data.localisationsystem.locsys import LocalisationSystemProcess
# LocSysProc = LocalisationSystemProcess([], [LocStS])
# allProcesses.append(LocSysProc)



# =============================== CONTROL =================================================
if enableRc:
    rcShR, rcShS   = Pipe(duplex = False)           # rc      ->  serial handler

    # serial handler process
    shProc = SerialHandlerProcess([rcShR], [])
    allProcesses.append(shProc)
    
    #shProc = SerialHandlerProcess([rcSer], [])     prima podatke od kamere i salje na nukleus
    #allProcesses.append(shProc)
    #print(rcShS)
    #rcShS = {'action': '1', 'speed': 0.18}
    #test = Distance([],[rcShS])
    #allProcesses.append(test)
    
    rcProc = RemoteControlReceiverProcess([],[rcShS])
    allProcesses.append(rcProc)

# ==================================================

# ===================================== START PROCESSES ==================================
print("Starting the processes!",allProcesses)
for proc in allProcesses:
    proc.daemon = True
    proc.start()
    #print(proc)


# ===================================== STAYING ALIVE ====================================
blocker = Event()  

try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")
    for proc in allProcesses:
        if hasattr(proc,'stop') and callable(getattr(proc,'stop')):
            print("Process with stop",proc)
            proc.stop()
            proc.join()
        else:
            print("Process witouth stop",proc)
            proc.terminate()
            proc.join()
