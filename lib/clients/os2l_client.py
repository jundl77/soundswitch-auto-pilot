import time
import logging
import socket
import datetime
from typing import List, Optional
from zeroconf import IPVersion, ServiceBrowser, ServiceStateChange, Zeroconf, ServiceInfo
from threading import Thread
from lib.clients.os2l_sender import Os2lSender
import lib.clients.os2l_messages as os2l_messages
from lib.analyser.music_analyser import MusicAnalyser

OS2L_SERVICE_NAME = '_os2l._tcp.local.'


class Os2lService:
    def __init__(self, ipv4_address: str, port: int, service_name: str, server: str):
        self.ipv4_address: str = ipv4_address
        self.port: int = port
        self.service_name: str = service_name
        self.server: str = server

    def __str__(self):
        return f"[service_connection: {self.ipv4_address}:{self.port}, service_name: {self.service_name}, server: {self.server}]"


global_services: List[Os2lService] = []
service_discovery_error: str = ''


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def on_service_state_change(zeroconf: Zeroconf,
                            service_type: str,
                            name: str,
                            state_change: ServiceStateChange) -> None:
    global global_services, service_discovery_error
    logging.info(f'[os2l-discovery] service state changed: [name: {name}, type {service_type}, state change: {state_change}]')

    local_ip = get_local_ip()

    if state_change is ServiceStateChange.Added:
        service_info: Optional[ServiceInfo] = zeroconf.get_service_info(service_type, name)
        if service_info:
            ipv4_addresses: List[str] = service_info.parsed_scoped_addresses(IPVersion.V4Only)
            kept_addresses = [address for address in ipv4_addresses if address == local_ip]
            if len(kept_addresses) == 0:
                # we time out after 5sec if we don't find anything
                return
            if len(kept_addresses) != 1:
                service_discovery_error = f'found more than one os2l service on local ip: {local_ip}, found: {kept_addresses}'
                return

            os2l_service = Os2lService(kept_addresses[0], service_info.port, service_info.name, service_info.server)
            logging.info(f'[os2l-discovery] found: {os2l_service}')
            global_services.append(os2l_service)


class Os2lClient:
    def __init__(self):
        self.os2l_sender: Os2lSender = Os2lSender()

    def set_analyser(self, analyser: MusicAnalyser):
        self.os2l_sender.set_analyser(analyser)

    def start(self):
        # we can't call this from within the main asyncio event-loop, so we spawn a thread and await its completion
        # instead
        thread = Thread(target=self._find_services)
        thread.start()
        thread.join()

        if service_discovery_error != '':
            raise RuntimeError(f"unable to find correct os2l service: {service_discovery_error}")

        assert len(global_services) == 1, f"more than one os2l service found: {global_services}"
        service = global_services[0]
        self.os2l_sender.start(service.ipv4_address, service.port)
        logging.info(f'[os2l] connected to service: {service})')

    def stop(self):
        if self.os2l_sender.is_running:
            logging.info(f'[os2l] stopping os2l client')
            self.os2l_sender.stop()

    def on_sound_start(self, time_elapsed: int, beat_pos: float, first_beat: float, bpm: float):
        self.os2l_sender.send_message(os2l_messages.logon_message())
        self.os2l_sender.send_message(os2l_messages.song_loaded_message(time_elapsed, beat_pos, first_beat, bpm))
        self.os2l_sender.send_message(os2l_messages.play_start_message())

    def on_sound_stop(self):
        self.os2l_sender.send_message(os2l_messages.play_stop_message())

    async def send_beat(self, change: bool, pos: int, bpm: float, strength: float):
        message = os2l_messages.beat_message(change, pos, bpm, strength)
        self.os2l_sender.send_message(message)
        logging.info(f'[os2l] sent beat message: {message}')

    def _find_services(self):
        global service_discovery_error
        zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        service_names = [OS2L_SERVICE_NAME]
        ServiceBrowser(zeroconf, service_names, handlers=[on_service_state_change])
        logging.info('[os2l-discovery] searching for soundswitch services..')
        search_start = datetime.datetime.now()
        while len(global_services) == 0:
            if service_discovery_error != '':
                break
            if datetime.datetime.now() - search_start > datetime.timedelta(seconds=5):
                service_discovery_error = 'unable to find soundswitch service after 5sec'
            time.sleep(0.1)
        zeroconf.close()
