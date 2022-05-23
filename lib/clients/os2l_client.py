import time
import logging
import asyncio
import socket
from typing import List, Optional, cast
from zeroconf import IPVersion, ServiceBrowser, ServiceStateChange, Zeroconf, ServiceInfo
from threading import Thread

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
    global global_services
    logging.info(f'[os2l-discovery] service state changed: [name: {name}, type {service_type}, state change: {state_change}]')

    local_ip = get_local_ip()

    if state_change is ServiceStateChange.Added:
        service_info: Optional[ServiceInfo] = zeroconf.get_service_info(service_type, name)
        if service_info:
            ipv4_addresses: List[str] = service_info.parsed_scoped_addresses(IPVersion.V4Only)
            kept_addresses = [address for address in ipv4_addresses if address == local_ip]
            assert len(kept_addresses) == 1, f"more than one os2l service found on local ip: {local_ip}"

            os2l_service = Os2lService(kept_addresses[0], service_info.port, service_info.name, service_info.server)
            logging.info(f'[os2l-discovery] found: {os2l_service}')
            global_services.append(os2l_service)


class Os2lClient:
    def __init__(self):
        self.os2l_socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        # we can't call this from within the main asyncio event-loop, so we spawn a thread and await its completion
        # instead
        thread = Thread(target=self._find_services)
        thread.start()
        thread.join()

        assert len(global_services) == 1, f"more than one os2l service found: {global_services}"
        service = global_services[0]
        server_address = (service.ipv4_address, service.port)
        logging.info(f'[os2l] connecting to {service.ipv4_address}:{service.port} ({service.service_name})')
        self.os2l_socket.connect(server_address)

    def stop(self):
        self.os2l_socket.close()

    async def send_beat(self, change: bool, pos: int, bpm: float, strength: float):
        change_str = 'true' if change else 'false'
        message = '{"evt":"beat","change":%s,"pos":%d,"bpm":%d,"strength":%d}' % (change_str, pos, bpm, strength)
        self.os2l_socket.sendall(message.encode())
        logging.info(f'[os2l] sent beat message: {message}')

    def _find_services(self):
        zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        service_names = [OS2L_SERVICE_NAME]
        ServiceBrowser(zeroconf, service_names, handlers=[on_service_state_change])
        logging.info('[os2l-discovery] searching for services..')
        while len(global_services) == 0:
            time.sleep(0.1)
        zeroconf.close()
