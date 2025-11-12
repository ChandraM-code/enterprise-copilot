#!/usr/bin/env python
"""
Management script for the Agentic Cache-Driven Application
Provides utilities for setup, testing, and maintenance
"""

import os
import sys
import subprocess
import argparse


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists(".env"):
        print("⚠️  .env file not found")
        print("   Creating .env from template...")
        if os.path.exists("env.template"):
            import shutil
            shutil.copy("env.template", ".env")
            print("✓ .env file created. Please edit it with your API keys.")
        else:
            print("❌ env.template not found")
        return False
    print("✓ .env file found")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def check_redis():
    """Check if Redis is accessible"""
    print("\n🔍 Checking Redis connection...")
    try:
        import redis
        from config import settings
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True
        )
        client.ping()
        print("✓ Redis is accessible")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Run: docker-compose up -d redis")
        return False


def check_qdrant():
    """Check if Qdrant is accessible"""
    print("\n🔍 Checking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        from config import settings
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        print("✓ Qdrant is accessible")
        return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   Run: docker-compose up -d qdrant")
        return False


def start_docker_services():
    """Start Docker services using docker-compose"""
    print("\n🐳 Starting Docker services...")
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("✓ Docker services started")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start Docker services")
        return False
    except FileNotFoundError:
        print("❌ docker-compose not found. Please install Docker.")
        return False


def stop_docker_services():
    """Stop Docker services"""
    print("\n🐳 Stopping Docker services...")
    try:
        subprocess.run(["docker-compose", "down"], check=True)
        print("✓ Docker services stopped")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to stop Docker services")
        return False


def run_app():
    """Run the FastAPI application"""
    print("\n🚀 Starting application...")
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped")


def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    try:
        subprocess.run([sys.executable, "test_api.py"])
    except KeyboardInterrupt:
        print("\n\n✓ Tests interrupted")


def run_examples():
    """Run example usage script"""
    print("\n📚 Running examples...")
    try:
        subprocess.run([sys.executable, "example_usage.py"])
    except KeyboardInterrupt:
        print("\n\n✓ Examples interrupted")


def setup_command(args):
    """Setup command: Install dependencies and check configuration"""
    print("="*60)
    print("  SETUP - Agentic Cache-Driven Application")
    print("="*60)
    
    check_python_version()
    check_env_file()
    
    if args.install_deps:
        install_dependencies()
    
    if args.start_services:
        start_docker_services()
        import time
        print("\n⏳ Waiting for services to start...")
        time.sleep(5)
    
    check_redis()
    check_qdrant()
    
    print("\n" + "="*60)
    print("  Setup complete!")
    print("  Next steps:")
    print("  1. Edit .env with your API keys")
    print("  2. Run: python manage.py run")
    print("="*60)


def run_command(args):
    """Run command: Start the application"""
    run_app()


def test_command(args):
    """Test command: Run tests"""
    run_tests()


def examples_command(args):
    """Examples command: Run examples"""
    run_examples()


def docker_command(args):
    """Docker command: Manage Docker services"""
    if args.action == "up":
        start_docker_services()
    elif args.action == "down":
        stop_docker_services()
    elif args.action == "restart":
        stop_docker_services()
        start_docker_services()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Management script for Agentic Cache-Driven Application"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the application")
    setup_parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    setup_parser.add_argument("--start-services", action="store_true", help="Start Docker services")
    setup_parser.set_defaults(func=setup_command)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the application")
    run_parser.set_defaults(func=run_command)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.set_defaults(func=test_command)
    
    # Examples command
    examples_parser = subparsers.add_parser("examples", help="Run examples")
    examples_parser.set_defaults(func=examples_command)
    
    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Manage Docker services")
    docker_parser.add_argument("action", choices=["up", "down", "restart"], help="Docker action")
    docker_parser.set_defaults(func=docker_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()

