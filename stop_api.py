#!/usr/bin/env python3
"""
Synthony API Server Stop Script (Python)

Alternative to stop_api.sh for cross-platform compatibility.
"""

import os
import sys
import signal
import time
from pathlib import Path


def stop_server():
    """Stop the API server."""
    print("Stopping Synthony API Server...")
    print()

    pid_file = Path("logs/api_server.pid")

    # Check if PID file exists
    if pid_file.exists():
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())

            print(f"  Found server process (PID: {pid})")

            # Try to kill the process
            try:
                os.kill(pid, signal.SIGTERM)
                print("  Sent SIGTERM signal")

                # Wait for process to stop
                time.sleep(2)

                # Check if still running
                try:
                    os.kill(pid, 0)  # Check if process exists
                    print("  Process still running, forcing stop...")
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass  # Process is dead

                print("✓ API server stopped")
                pid_file.unlink()

            except OSError as e:
                if e.errno == 3:  # No such process
                    print(f"⚠  Process not running (PID: {pid})")
                    pid_file.unlink()
                else:
                    print(f"✗ Error stopping process: {e}")
                    sys.exit(1)

        except ValueError:
            print("✗ Invalid PID in file")
            pid_file.unlink()
            sys.exit(1)

    else:
        # Try to find by port
        port = os.getenv("API_PORT", "8000")
        print(f"  Looking for process on port {port}...")

        try:
            import psutil

            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr.port == int(port):
                            pid = proc.pid
                            print(f"  Found process on port {port} (PID: {pid})")
                            proc.terminate()
                            time.sleep(1)
                            if proc.is_running():
                                proc.kill()
                            print("✓ API server stopped")
                            return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            print(f"⚠  No API server running on port {port}")

        except ImportError:
            print("⚠  No PID file found and psutil not available")
            print("  Install psutil: pip install psutil")
            print("  Or manually kill the server process")

    print()


if __name__ == "__main__":
    stop_server()
