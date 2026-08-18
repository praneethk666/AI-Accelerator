"""
run_tunnel.py — Helper to launch and expose the Streamable MCP server to the internet or cloud.
Supports Cloudflared, Ngrok, Localtunnel, or Direct Host binding.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Host and Tunnel Runner for MCP Server")
    parser.add_argument(
        "--mode",
        choices=["local", "cloudflared", "ngrok", "localtunnel"],
        default="local",
        help="Hosting/tunnel mode (default: local)",
    )
    parser.add_argument("--port", type=int, default=8100, help="Server port (default: 8100)")
    args = parser.parse_args()

    print("=" * 70)
    print("  🚀 STREAMABLE MCP SERVER — HOSTING & TUNNEL RUNNER")
    print(f"  Mode: {args.mode.upper()} | Port: {args.port}")
    print("=" * 70)

    if args.mode == "local":
        print(f"\n[INFO] Starting MCP server locally at http://0.0.0.0:{args.port}")
        print(f"[INFO] MCP Endpoint:  http://localhost:{args.port}/mcp")
        print(f"[INFO] Health Probe:  http://localhost:{args.port}/health\n")
        cmd = [sys.executable, "-m", "src.server"]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    elif args.mode == "cloudflared":
        print(f"\n[INFO] Starting Cloudflared tunnel for port {args.port}...")
        print("[INFO] Run the server in a separate terminal: python -m src.server")
        cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{args.port}"]
        try:
            subprocess.run(cmd)
        except FileNotFoundError:
            print("[ERROR] 'cloudflared' CLI not found on system PATH.")
            print("[INFO] Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")

    elif args.mode == "ngrok":
        print(f"\n[INFO] Starting Ngrok tunnel on port {args.port}...")
        cmd = ["ngrok", "http", str(args.port)]
        try:
            subprocess.run(cmd)
        except FileNotFoundError:
            print("[ERROR] 'ngrok' CLI not found on system PATH.")
            print("[INFO] Download from: https://ngrok.com/download")

    elif args.mode == "localtunnel":
        print(f"\n[INFO] Starting Localtunnel on port {args.port} via npx...")
        cmd = ["npx", "localtunnel", "--port", str(args.port)]
        try:
            subprocess.run(cmd, shell=True)
        except Exception as e:
            print(f"[ERROR] Failed to launch localtunnel: {e}")


if __name__ == "__main__":
    main()
