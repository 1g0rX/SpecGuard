#!/usr/bin/env python3
"""SpecGuard — Launcher. Roda o servidor e abre o navegador automaticamente."""

import os
import sys
import webbrowser
import signal
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import app

server = None

def signal_handler(sig, frame):
    print("\nEncerrando SpecGuard...")
    os._exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    url = 'http://127.0.0.1:5000'
    print("=" * 50)
    print("  SpecGuard 🛡️ - Detecção de Violência via FFT")
    print("=" * 50)
    print(f"\n  Interface: {url}")
    print("  Pressione Ctrl+C para encerrar\n")

    webbrowser.open(url)
    app.run_server(port=5000, debug=False)
