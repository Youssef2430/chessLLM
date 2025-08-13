#!/usr/bin/env python3
"""
Human Engine Installation Helper for Chess LLM Benchmark

This script helps users install and configure human-like chess engines:
- Maia Chess Engine (most human-like)
- Leela Chess Zero (LCZero)
- Human-like Stockfish configuration

Usage:
    python install_human_engines.py [--engine ENGINE_TYPE] [--auto]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import urllib.request
import tempfile
import zipfile
import tarfile

def detect_os() -> str:
    """Detect the current operating system."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"

def run_command(cmd: List[str], check: bool = True) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"

def check_engine_installed(engine_name: str) -> Optional[str]:
    """Check if an engine is installed and return its path."""
    path = shutil.which(engine_name)
    if path:
        print(f"✅ {engine_name} found at: {path}")
        return path
    else:
        print(f"❌ {engine_name} not found in PATH")
        return None

def install_stockfish():
    """Install or verify Stockfish installation."""
    print("\n🤖 Installing/Verifying Stockfish...")

    # Check if already installed
    if check_engine_installed("stockfish"):
        print("✅ Stockfish is already installed!")
        return True

    os_type = detect_os()

    if os_type == "macos":
        print("Installing Stockfish via Homebrew...")
        if shutil.which("brew"):
            success, output = run_command(["brew", "install", "stockfish"])
            if success:
                print("✅ Stockfish installed successfully!")
                return True
            else:
                print(f"❌ Failed to install Stockfish: {output}")
        else:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")

    elif os_type == "linux":
        print("Installing Stockfish via package manager...")

        # Try apt (Ubuntu/Debian)
        if shutil.which("apt-get"):
            print("Using apt-get...")
            success, output = run_command(["sudo", "apt-get", "update"], check=False)
            success, output = run_command(["sudo", "apt-get", "install", "-y", "stockfish"])
            if success:
                print("✅ Stockfish installed successfully!")
                return True

        # Try yum (RedHat/CentOS)
        elif shutil.which("yum"):
            print("Using yum...")
            success, output = run_command(["sudo", "yum", "install", "-y", "stockfish"])
            if success:
                print("✅ Stockfish installed successfully!")
                return True

        # Try dnf (Fedora)
        elif shutil.which("dnf"):
            print("Using dnf...")
            success, output = run_command(["sudo", "dnf", "install", "-y", "stockfish"])
            if success:
                print("✅ Stockfish installed successfully!")
                return True

        print("❌ Could not install Stockfish automatically.")
        print("   Please install manually: sudo apt-get install stockfish")

    elif os_type == "windows":
        print("❌ Automatic Windows installation not supported.")
        print("   Please install Stockfish manually:")
        print("   1. Install Chocolatey: https://chocolatey.org/")
        print("   2. Run: choco install stockfish")
        print("   Or download from: https://stockfishchess.org/")

    return False

def install_lczero():
    """Install or verify Leela Chess Zero installation."""
    print("\n♟️  Installing/Verifying Leela Chess Zero...")

    # Check if already installed
    if check_engine_installed("lc0") or check_engine_installed("leela-chess-zero"):
        print("✅ Leela Chess Zero is already installed!")
        return True

    os_type = detect_os()

    if os_type == "macos":
        print("Installing LCZero via Homebrew...")
        if shutil.which("brew"):
            success, output = run_command(["brew", "install", "lc0"])
            if success:
                print("✅ LCZero installed successfully!")
                return True
            else:
                print(f"❌ Failed to install LCZero: {output}")
        else:
            print("❌ Homebrew not found.")

    elif os_type == "linux":
        print("Installing LCZero...")

        # Try snap first
        if shutil.which("snap"):
            print("Trying snap installation...")
            success, output = run_command(["sudo", "snap", "install", "lc0"], check=False)
            if success:
                print("✅ LCZero installed successfully via snap!")
                return True

        # Try apt
        if shutil.which("apt-get"):
            print("Trying apt installation...")
            success, output = run_command(["sudo", "apt-get", "install", "-y", "lc0"], check=False)
            if success:
                print("✅ LCZero installed successfully!")
                return True

        print("❌ Could not install LCZero automatically.")
        print("   Please install manually or download from: https://lczero.org/")

    elif os_type == "windows":
        print("❌ Automatic Windows installation not supported.")
        print("   Please download LCZero from: https://lczero.org/")

    return False

def install_maia():
    """Install or verify Maia Chess Engine."""
    print("\n🧠 Installing/Verifying Maia Chess Engine...")

    # Check if already installed
    if check_engine_installed("maia"):
        print("✅ Maia is already installed!")
        return True

    print("❌ Maia automatic installation not yet supported.")
    print("\n📋 To install Maia manually:")
    print("1. Visit: https://github.com/CSSLab/maia-chess")
    print("2. Follow the installation instructions for your platform")
    print("3. Make sure 'maia' is in your PATH")
    print("\nAlternatively, you can:")
    print("• Use LCZero with human-like settings (--human-engine-type lczero)")
    print("• Use human-like Stockfish (--human-engine-type human_stockfish)")

    return False

def create_engine_config():
    """Create a configuration file for human engines."""
    config_dir = Path.home() / ".chess_llm_bench"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "human_engines.conf"

    engines = {}

    # Check for installed engines
    for engine in ["stockfish", "lc0", "leela-chess-zero", "maia"]:
        path = shutil.which(engine)
        if path:
            engines[engine] = path

    # Write config
    with config_file.open("w") as f:
        f.write("# Human Engine Configuration\n")
        f.write("# Generated by install_human_engines.py\n\n")

        for engine, path in engines.items():
            f.write(f"{engine}_path = {path}\n")

        f.write("\n# Recommended human engine priority:\n")
        f.write("# 1. maia (most human-like)\n")
        f.write("# 2. lc0 (neural network)\n")
        f.write("# 3. stockfish (traditional with human settings)\n")

    print(f"✅ Configuration saved to: {config_file}")
    return config_file

def validate_engines():
    """Validate that installed engines work correctly."""
    print("\n🔍 Validating installed engines...")

    engines_to_test = [
        ("stockfish", "Stockfish"),
        ("lc0", "LCZero"),
        ("leela-chess-zero", "LCZero"),
        ("maia", "Maia")
    ]

    working_engines = []

    for engine_cmd, engine_name in engines_to_test:
        path = shutil.which(engine_cmd)
        if path:
            # Test UCI communication
            try:
                result = subprocess.run(
                    [path],
                    input="uci\nquit\n",
                    text=True,
                    capture_output=True,
                    timeout=5
                )

                if "uciok" in result.stdout.lower():
                    print(f"✅ {engine_name} is working correctly")
                    working_engines.append((engine_cmd, engine_name, path))
                else:
                    print(f"❌ {engine_name} found but not responding correctly")

            except Exception as e:
                print(f"❌ {engine_name} test failed: {e}")

    if working_engines:
        print(f"\n🎉 {len(working_engines)} human engines are ready!")
        print("\nYou can now use:")
        for engine_cmd, engine_name, path in working_engines:
            if engine_cmd == "maia":
                print(f"  --use-human-engine --human-engine-type maia")
            elif engine_cmd in ["lc0", "leela-chess-zero"]:
                print(f"  --use-human-engine --human-engine-type lczero")
            elif engine_cmd == "stockfish":
                print(f"  --use-human-engine --human-engine-type human_stockfish")
    else:
        print("\n❌ No working human engines found.")
        print("   You can still use regular Stockfish for benchmarks.")

    return working_engines

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="Install human-like chess engines for Chess LLM Benchmark"
    )
    parser.add_argument(
        "--engine",
        choices=["all", "maia", "lczero", "stockfish"],
        default="all",
        help="Engine to install (default: all)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Attempt automatic installation where possible"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing installations"
    )

    args = parser.parse_args()

    print("🏆 Chess LLM Benchmark - Human Engine Installer")
    print("=" * 50)

    os_type = detect_os()
    print(f"Detected OS: {os_type}")

    if args.validate_only:
        validate_engines()
        return

    # Installation phase
    if args.engine in ["all", "stockfish"]:
        install_stockfish()

    if args.engine in ["all", "lczero"]:
        install_lczero()

    if args.engine in ["all", "maia"]:
        install_maia()

    # Create configuration
    create_engine_config()

    # Validate installations
    working_engines = validate_engines()

    # Final instructions
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("\n1. Test your installation:")
    print("   python main.py --demo --use-human-engine")

    print("\n2. Run benchmarks with human engines:")
    print("   python main.py --preset premium --use-human-engine")

    if not working_engines:
        print("\n3. If no engines work, you can still use regular Stockfish:")
        print("   python main.py --preset premium")

    print("\n4. For more options, see:")
    print("   python main.py --help")

    print("\n🧠 Human engines provide more realistic opponents!")
    print("   Maia is trained on human games and plays like real players.")

if __name__ == "__main__":
    main()
