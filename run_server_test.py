
import subprocess
import json
import sys
import os

def run_test():
    # Set PYTHONPATH to include the current directory and src
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{os.getcwd()}/src"

    # Start the server process
    process = subprocess.Popen(
        [sys.executable, "mcp_server/server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0 # Unbuffered
    )

    # Helper to send a message and get response
    def send_receive(msg):
        print(f"Sending: {json.dumps(msg)}")
        process.stdin.write(json.dumps(msg) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        print(f"Received: {response.strip()}")
        return response

    try:
        # 1. Initialize
        init_msg = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", # Updated protocol version
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            },
            "id": 1
        }
        send_receive(init_msg)

        # 2. Check initialized notification
        notify_msg = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        process.stdin.write(json.dumps(notify_msg) + "\n")
        process.stdin.flush()

        # 3. List Resources
        list_resources_msg = {
            "jsonrpc": "2.0",
            "method": "resources/list",
            "params": {},
            "id": 2
        }
        response = send_receive(list_resources_msg)
        
        # Check if list_resources returned successfully
        if "error" in response:
             print("ERROR: resources/list failed")
             print(process.stderr.read())

    except Exception as e:
        print(f"Exception: {e}")
        # Read stderr to see if there was a server crash
        print("Server stderr:")
        print(process.stderr.read())
    finally:
        process.terminate()

if __name__ == "__main__":
    run_test()
