# BFMC - Brain Project

The project contains all the provided code for the RPi, more precisely:
- Firmware for communicating with the Nucleo and control the robot movements (Speed with constant current consumption, speed with constant speed, braking, moving and steering);
- Firmware for gathering data from the sensors (IMU and Camera);
- API's for communicating with the environmental servers at Bosch location;
- Simulated servers for the API's.

## The documentation is available in more details here:
[Documentation](https://boschfuturemobility.com/brain/)

## Samo remote control:
 - 1.enableRc =  True, ostali False
 - 2.u RemoteContorlTrasmiter na vasem laptopu, RemoteControlReciver na RPI ide IP adresa sa RPI
 - 3.python3 -m bin.remotecontroltransmitter na laptopu se poziva
  
## Remote Control + Stream:
 - 1.enableRc =  True, enableStream = True  ostali False
 - 2.Iz stream zakomentarisati  rcShR, rcShS   = Pipe(duplex = False),shProc = SerialHandlerProcess([rcShR], []),allProcesses.append(shProc)
 - 3.Obrisati rcShS i ostaviti samo [] umesto njega
 - 4.u RemoteContorlTrasmiter na vasem laptopu, RemoteControlReciver na RPI ide IP adresa sa RPI
 - 5.u CameraStreamer na RPI, CameraReciver na laptopu ide IP adresa sa laptopa
 - 6.python3 -m bin.remotecontroltransmitter na laptopu se poziva
 - 7.python3 -m bin.camerareceiver na laptopu se poziva
  
## Stream + line:
 - 1.enableStream = True  ostali False
 - 2. shProc = SerialHandlerProcess([rcShR], []), allProcesses.append(shProc)
 - 3. camLine = LineDetection([camStR],[rcShS]), allProcesses.append(camLine)
 - 4.u CameraStreamer na RPI, CameraReciver na laptopu ide IP adresa sa laptopa
  
## Stream + line + HW:
 - 1.enableStream = True  ostali False
 - 2. shProc = SerialHandlerProcess([rcShR], []), allProcesses.append(shProc)
 - 3. camLine = LineDetection([camStR],[camLineShS]), allProcesses.append(camLine)
 - 4. test = Distance([camLineShR],[rcShS]), allProcesses.append(test)
 - 5.u CameraStreamer na RPI, CameraReciver na laptopu ide IP adresa sa laptopa
