# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from threading import Thread
from src.hardware.serialhandler.messageconverter    import MessageConverter

class WriteThread(Thread):
    flag = 0;
    # ===================================== INIT =========================================
    def __init__(self, inP, serialCom, logFile):
        """The purpose of this thread is to redirectionate the received through input pipe to an other device by using serial communication. 
        
        Parameters
        ----------
        inP : multiprocessing.Pipe 
            Input pipe to receive the command from an other process.
        serialCom : serial.Serial
            The serial connection interface between the two device.
        logFile : FileHandler
            The log file handler to save the commands. 
        """
        
        super(WriteThread,self).__init__()
        self.flag = 0; 
        self.inP        =  inP
        self.serialCom  =  serialCom
        self.logFile    =  logFile
        self.messageConverter = MessageConverter()
        
        command = {'action': '6', 'kp': 0.115, 'ki': 0.99, 'kd': 0.00000222, 'tf': 0.04}
        command_msg = self.messageConverter.get_command(**command)
        self.serialCom.write(command_msg.encode('ascii'))
        self.logFile.write(command_msg)
        
        command = {'action': '4', 'activate': True}
        command_msg = self.messageConverter.get_command(**command)
        self.serialCom.write(command_msg.encode('ascii'))
        self.logFile.write(command_msg)
        
        command = {'action': '1', 'speed': 0.00}
        command_msg = self.messageConverter.get_command(**command)
        self.serialCom.write(command_msg.encode('ascii'))
        self.logFile.write(command_msg)
        
        command = {'action': '2', 'steerAngle': 0.0}
        command_msg = self.messageConverter.get_command(**command)
        self.serialCom.write(command_msg.encode('ascii'))
        self.logFile.write(command_msg)
        
    # ===================================== RUN ==========================================
    def run(self):
        """ Represents the thread activity to redirectionate the message.
        """
        try:
            while True:
                command = self.inP.recv()
                #print(command)
                # Unpacking the dictionary into action and values
                command_msg = self.messageConverter.get_command(**command)
                self.serialCom.write(command_msg.encode('ascii'))
                self.logFile.write(command_msg)
                """
                if flag == 1:
                    command = {'action': '1', 'speed': 0.00}
                    #print(command)
                    # Unpacking the dictionary into action and values
                    command_msg = self.messageConverter.get_command(**command)
                    #print("bilo stasas")
                    self.serialCom.write(command_msg.encode('ascii'))
                    self.logFile.write(command_msg)
                """
        except:
                print("Speed interupt")
