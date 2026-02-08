#!/usr/bin/env python3
"""
Synthony API Server Startup Script (Python)

Alternative to start_api.sh for cross-platform compatibility.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def load_env_file(env_path=".env"):
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    if env_file.exists():
        print(f"✓ Loading environment variables from {env_path}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        return True
    return False


def check_installation():
    """Check if package is installed."""
    try:
        import synthony
        return True
    except ImportError:
        return False


def install_package():
    """Install the package with dependencies."""
    print("Installing package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[api,llm]"], check=True)
    print("✓ Package installed\n")


def display_config():
    """Display current configuration."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = os.getenv("API_PORT", "8000")
    workers = os.getenv("API_WORKERS", "1")

    print("\n" + "="*80)
    print("Synthony Data Analysis & Model Recommendation API")
    print("="*80)
    print()

    print("Configuration:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Workers: {workers}")
    print()

    # Display LLM configuration
    vllm_url = os.getenv("VLLM_URL")
    vllm_model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    vllm_key = os.getenv("VLLM_API_KEY")
    openai_url = os.getenv("OPENAI_URL", "https://api.openai.com/v1")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if vllm_url:
        vllm_key = os.getenv("VLLM_API_KEY")
        print("✓ VLLM configured:")
        print(f"  URL: {vllm_url}")
        print(f"  Model: {vllm_model}")
        print(f"  API Key: {'***configured***' if vllm_key else 'not set'}")
        print()
    elif openai_key:
        print("✓ OpenAI configured:")
        print(f"  URL: {openai_url}")
        print(f"  Model: {openai_model}")
        print(f"  API Key: {'***configured***' if openai_key else 'not set'}")
        print()
    else:
        print("⚠️  No LLM configured (rule-based mode only)")
        print("  To enable LLM mode, set VLLM_URL or OPENAI_API_KEY")
        print()

    # Check SystemPrompt
    system_prompt = os.getenv("SYNTHONY_SYSTEM_PROMPT")
    if system_prompt:
        if Path(system_prompt).exists():
            print(f"✓ Custom SystemPrompt: {system_prompt}")
        else:
            print(f"✗ Custom SystemPrompt not found: {system_prompt}")
    elif Path("docs/SystemPrompt_v3.md").exists():
        print("✓ Using default SystemPrompt: docs/SystemPrompt_v3.md")
    else:
        print("⚠️  SystemPrompt file not found")
    print()

    print("API will be available at:")
    print(f"  Main API:      http://localhost:{port}")
    print(f"  Swagger Docs:  http://localhost:{port}/docs")
    print(f"  ReDoc:         http://localhost:{port}/redoc")
    print()


def start_server(mode="dev"):
    """Start the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = os.getenv("API_PORT", "8000")
    workers = os.getenv("API_WORKERS", "4" if mode == "prod" else "1")

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "synthony.api.server:app",
        "--host", host,
        "--port", port,
    ]

    if mode == "prod":
        print("Starting API server in production mode...")
        cmd.extend(["--workers", workers])
        cmd.extend(["--log-level", "info"])
    elif mode == "background":
        print("Starting API server in background mode...")
        log_file = Path("logs") / f"api_server.log"
        print(f"  Log file: {log_file}")
        cmd.extend(["--reload"])

        # Start in background
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        # Save PID
        pid_file = Path("logs/api_server.pid")
        with open(pid_file, "w") as f:
            f.write(str(process.pid))

        print(f"\n✓ API server started successfully (PID: {process.pid})")
        print(f"\nTo view logs:")
        print(f"  tail -f {log_file}")
        print(f"\nTo stop server:")
        print(f"  python stop_api.py")
        print(f"  or: kill {process.pid}")
        return
    else:
        print("Starting API server in development mode...")
        print("Press CTRL+C to stop\n")
        cmd.extend(["--reload"])
        cmd.extend(["--log-level", "info"])

    # Start server (foreground)
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start Synthony API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_api.py                  # Development mode (default)
  python start_api.py --mode prod      # Production mode
  python start_api.py --mode background # Background mode
  python start_api.py --port 9000      # Custom port
  python start_api.py --env .env.prod  # Custom env file
        """
    )

    parser.add_argument(
        "--mode",
        choices=["dev", "prod", "background"],
        default="dev",
        help="Server mode (default: dev)"
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Server port (overrides .env)"
    )

    parser.add_argument(
        "--host",
        help="Server host (overrides .env)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers (production mode only)"
    )

    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file (default: .env)"
    )

    args = parser.parse_args()

    # Load environment variables
    if not load_env_file(args.env):
        if args.env != ".env":
            print(f"⚠️  Environment file not found: {args.env}")
            sys.exit(1)
        else:
            print("ℹ  No .env file found (optional)")
            print("   Create .env file to configure VLLM/OpenAI settings")
            print()

    # Override with command line arguments
    if args.port:
        os.environ["API_PORT"] = str(args.port)
    if args.host:
        os.environ["API_HOST"] = args.host
    if args.workers:
        os.environ["API_WORKERS"] = str(args.workers)

    # Check installation
    if not check_installation():
        print("✗ Error: synthony package not installed")
        install_package()

    # Display configuration
    display_config()

    # Start server
    start_server(args.mode)


if __name__ == "__main__":
    main()
