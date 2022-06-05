import time
import logging
import socket
import json
import datetime
from threading import Thread
from queue import Queue, Empty
import lib.clients.os2l_messages as os2l_messages


class Os2lSender:
    def __init__(self):
        self.analyser: "MusicAnalyser" = None
        self.dest_ipv4_address: str = None
        self.dest_port: int = None
        self.os2l_socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sending_thread = Thread(target=self._run_sending_thread)
        self.message_queue: Queue = Queue()
        self.is_running: bool = False
        self.logon_complete: bool = False

        # state once logged in
        self.send_update_frequency: datetime.timedelta = datetime.timedelta(milliseconds=25)
        self.last_update_sent: datetime.datetime = datetime.datetime.now()

    def set_analyser(self, analyser: "MusicAnalyser"):
        self.analyser: "MusicAnalyser" = analyser

    def start(self, ipv4_address: str, port: int):
        self.dest_ipv4_address = ipv4_address
        self.dest_port = port
        self.is_running = True
        self.sending_thread.start()

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.logon_complete = False
            self.sending_thread.join()

            # send stop message to soundswitch
            shutdown_message = os2l_messages.shutdown_message()
            logging.info(f'[os2l] sending shutdown message to soundswitch: {shutdown_message}')
            self._send_message(os2l_messages.shutdown_message())
            self.os2l_socket.close()

    def send_message(self, message: str):
        self.message_queue.put(message)

    def _run_sending_thread(self):
        logging.info(f'[os2l] started sender thread')
        logging.info(f'[os2l] connecting to {self.dest_ipv4_address}:{self.dest_port}')
        server_address = (self.dest_ipv4_address, self.dest_port)
        self.os2l_socket.connect(server_address)
        self.os2l_socket.setblocking(False)

        while self.is_running:
            self._poll_socket()
            self._poll_queue()
            if self.logon_complete:
                self._send_update_if_due()
            time.sleep(0.001)

    def _poll_socket(self):
        try:
            received_message = self.os2l_socket.recv(4096)
            self._on_message(received_message.decode('utf-8'))
        except socket.error:
            pass  # expected

    def _poll_queue(self):
        if not self.logon_complete:
            return
        try:
            message = self.message_queue.get(block=False)
            self._send_message(message)
        except Empty:
            pass  # expected

    def _send_update_if_due(self):
        now = datetime.datetime.now()
        is_playing = self.analyser.is_song_playing()
        if is_playing and now - self.last_update_sent > self.send_update_frequency:
            beat_position: float = self.analyser.get_beat_position()
            time_elapsed: datetime.timedelta = self.analyser.get_song_duration()
            time_elapsed_ms: int = int(time_elapsed.microseconds / 1000)
            self.send_message(os2l_messages.update_message(beat_position, time_elapsed_ms))
            self.last_update_sent = now

    def _send_message(self, message: str):
        self.os2l_socket.sendall(message.encode())

    def _on_message(self, message):
        logging.info(f'[os2l] received {message}')
        json_message = json.loads(message)

        if 'evt' not in json_message:
            logging.info(f'[os2l] unable to process received message, skipping')

        os2l_event: str = json_message['evt']
        if os2l_event == 'subscribe':
            self._on_subscribe_message(json_message)
        else:
            logging.info(f"[os2l] received message with unknown event '{os2l_event}', skipping")

    def _on_subscribe_message(self, json_message):
        """ this sort-of acts as the login handshake """
        assert 'frequency' in json_message, "'json_message' was not in subscribe message"
        self.send_update_frequency = datetime.timedelta(milliseconds=int(json_message['frequency']))
        logging.info(f'[os2l] setting update frequency to {self.send_update_frequency.microseconds / 1000} ms')
        self._send_message(os2l_messages.logon_message())
        self.logon_complete = True
