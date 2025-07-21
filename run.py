#!/usr/bin/env python3
"""
GeoVibes Web Application Runner

This script creates a standalone web application from the GeoVibes interface
that can be accessed via a web browser instead of Jupyter notebook.

Usage:
    python run.py --config config.yaml
    python run.py --help

The script will start a web server and open the GeoVibes interface in your default browser.
"""

import argparse
import os
import sys
import webbrowser
import tempfile
import atexit
import subprocess
import json


def parse_arguments():
    """Parse command line arguments for GeoVibes configuration."""
    parser = argparse.ArgumentParser(
        description="Run GeoVibes as a standalone web application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML config file
  python run.py --config config.yaml
  
  # Use YAML config file with verbose output
  python run.py --config config.yaml --verbose
  
  # Specify custom port
  python run.py --config config.yaml --port 8888
        """,
    )

    # Configuration file (required)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML format) - REQUIRED",
    )

    # Web server options
    parser.add_argument(
        "--port",
        type=int,
        default=8866,
        help="Port to run the web server on (default: 8866)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the web server to (default: localhost)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not automatically open browser"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def create_notebook_content(config_path):
    """Create a temporary notebook that initializes GeoVibes with the given config path."""

    # Build the initialization code - just a single cell calling GeoVibes.create
    init_code = [
        "# Auto-generated GeoVibes initialization",
        "from geovibes.ui import GeoVibes",
        "",
        f"vibes = GeoVibes.create(config_path=r'{config_path}')",
    ]

    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": init_code,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    return notebook_content


def run_with_voila(config_path, args):
    """Run the application using Voila."""

    # Create temporary notebook
    notebook_content = create_notebook_content(config_path)

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_notebook = os.path.join(temp_dir, "geovibes_app.ipynb")

    # Cleanup function
    def cleanup():
        try:
            os.remove(temp_notebook)
            os.rmdir(temp_dir)
        except:
            pass

    atexit.register(cleanup)

    # Write notebook content
    with open(temp_notebook, "w") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"🚀 Starting GeoVibes web application on http://{args.host}:{args.port}")
    print(f"📊 Config: {config_path}")

    if not args.no_browser:
        # Give the server a moment to start before opening browser
        import threading
        import time

        def open_browser():
            time.sleep(3)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=open_browser, daemon=True).start()

    print("🔧 Starting Voila server...")

    # Build Voila command with error suppression
    voila_cmd = [
        sys.executable,
        "-m",
        "voila",
        temp_notebook,
        "--port",
        str(args.port),
        "--ip",
        args.host,
        "--no-browser",
    ]

    # Run Voila as subprocess with error suppression
    process = subprocess.Popen(
        voila_cmd,
        stderr=subprocess.PIPE if not args.verbose else None,
        universal_newlines=True,
    )

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down GeoVibes web application...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        cleanup()


def main():
    """Main entry point for the GeoVibes web application."""
    args = parse_arguments()

    # Validate that config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    if args.verbose:
        print(f"✅ Using configuration file: {args.config}")

    run_with_voila(args.config, args)


if __name__ == "__main__":
    main()
