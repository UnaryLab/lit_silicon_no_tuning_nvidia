#!/usr/bin/env python3

from concurrent import futures
import logging
import os
import subprocess
from time import sleep

import grpc
import grpc.experimental


logger = logging.getLogger(__name__)
protos, services = grpc.protos_and_services("freq.proto")


def get_gpu_num(gpu: int) -> int:
    # Placeholder for platform-specific logical-to-physical GPU mapping.
    return gpu


def set_power(gpu_num: int, value: int) -> int:
    logger.info("SERVER: setting GPU%s power cap to %s W", gpu_num, value)
    result = subprocess.run(
        ["sudo", "nvidia-smi", "-i", str(gpu_num), "-pl", str(value)],
        check=False,
    )
    return result.returncode


class FreqServer(services.FreqServerServicer):
    def SetPower(self, request, context):
        for tries in range(5, 0, -1):
            ret = set_power(get_gpu_num(request.gpu), request.watts)
            if ret == 0:
                return protos.FreqAck(ack=0)
            logger.warning("Failed setting power cap, %s tries left", tries)
            sleep(2)
        logger.error("Failed setting power cap, no tries left")
        return protos.FreqAck(ack=ret)


def serve(socket_path: str = "/tmp/freq.sock") -> int:
    if socket_path.startswith("/") and os.path.exists(socket_path):
        os.unlink(socket_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    services.add_FreqServerServicer_to_server(FreqServer(), server)
    server.add_insecure_port(f"unix://{socket_path}")
    server.start()
    logger.info("Serving power server on %s", socket_path)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    logging.basicConfig(filename="freq_server.log", level=logging.INFO)
    serve()
