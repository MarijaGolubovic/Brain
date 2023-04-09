import time
import random
from threading import Thread
from multiprocessing import Pipe
from livetraffic import EnvironmentalHandler

if __name__ == '__main__':
    beacon = 23456

    id = 120
    serverpublickey = 'publickey_livetraffic_server_test.pem'
    clientprivatekey = 'privatekey_livetraffic_client_test.pem'

    #: For testing purposes, with the provided simulated livetraffic_system, the pair of keys above is used
    #       -   "publickey_livetraffic_server_test.pem"     --> Ensure by the client that the server is the actual server
    #       -   "privatekey_livetraffic_client_test.pem"    --> Ensure by the server that the client is the actual client

    #: At Bosch location during the competition 
    #       -   Use the "publickey_livetraffic_server.pem" instead of the "publickey_livetraffic_server_test.pem" 
    #       -   As for the "publickey_livetraffic_client.pem", you will have to generate a pair of keys, private and public, using the following terminal lines.  Before the competition, instruction of where to send your publickey_livetraffic_client.pem will be given.

    #: openssl genrsa -out privateckey_livetraffic_client.pem 2048 ----> Creates a private ssh key and stores it in the current dir with the given name
    #: openssl rsa -in privateckey_livetraffic_client.pem -pubout -out publickey_livetraffic_client.pem ----> Creates the corresponding public key out of the private one. 

    #:
    #: To test the functionality, 
    #       -   copy the generated public key under test/livetrafficSERVER/keys 
    #       -   rename the key using this format: "id_publickey.pem" ,where the id is the id of the key you are trying to connect with
    #       -   The given example connects with the id 120 and the same key is saved with "120_publickey.pem"
    
    gpsStR, gpsStS = Pipe(duplex = False)

    envhandler = EnvironmentalHandler(id, beacon, serverpublickey, gpsStR, clientprivatekey)
    envhandler.start()
    time.sleep(5)
    for x in range(1, 10):
        time.sleep(random.uniform(1,5))
        a = {"obstacle_id": int(random.uniform(0,25)), "x": random.uniform(0,15), "y": random.uniform(0,15)}
        gpsStS.send(a)
        
    envhandler.stop()
    envhandler.join()

