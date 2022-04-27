import socket
import sys
import time

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('192.168.68.109', 51747)
print('connecting to %s port %s' % server_address)
sock.connect(server_address)

try:
    pos = 0
    while True:
        # Send data
        message = '{"evt":"beat","change":false,"pos":%d,"bpm":109.09,"strength":0.6}' % pos
        print('sending "%s"' % message)
        sock.sendall(message.encode())
        time.sleep(0.1)
        message = """  {"evt":"subscribed","trigger":"deck 1 get_time elapsed absolute","value":9736}
{"evt":"subscribed","trigger":"deck 3 get_time elapsed absolute","value":9736}
{"evt":"subscribed","trigger":"deck 4 get_time elapsed absolute","value":9736}
{"evt":"subscribed","trigger":"deck 1 get_beatpos","value":13.776912}
{"evt":"subscribed","trigger":"deck 3 get_beatpos","value":13.776912}
{"evt":"subscribed","trigger":"deck 4 get_beatpos","value":13.776912}"""
        #print('sending "%s"' % message)
        #sock.sendall(message.encode())

        time.sleep(1)
        pos += 1

finally:
    print('closing socket')
    sock.close()