"""Trava de instância única: impede dois robôs rodando ao mesmo tempo sobre os mesmos dados.

Usa uma porta TCP em 127.0.0.1 derivada da pasta de dados. O sistema operacional libera a porta
sozinho quando o processo morre (Ctrl+C, fechar o terminal, travar), então não sobra "trava velha".
"""
from __future__ import annotations

import hashlib
import os
import socket


class AlreadyRunning(Exception):
    pass


def lock_port(data_dir: str) -> int:
    h = hashlib.sha256(os.path.abspath(data_dir).lower().encode()).hexdigest()
    return 40000 + int(h, 16) % 20000


def acquire_instance_lock(data_dir: str) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        s.bind(("127.0.0.1", lock_port(data_dir)))
    except OSError:
        s.close()
        raise AlreadyRunning("já existe um robô rodando com esta pasta de dados") from None
    return s
