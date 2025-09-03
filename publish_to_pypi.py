#!/usr/bin/env python3
"""
PyPI Publishing Script for QuickInsights
This script automates the process of building and publishing QuickInsights to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def clean_build_dirs():
    """Clean build directories."""
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                print(f"🧹 Removing {path}")
                shutil.rmtree(path)


def check_requirements():
    """Check if required tools are installed."""
    required_tools = ['python', 'pip', 'twine']
    for tool in required_tools:
        if not shutil.which(tool):
            print(f"❌ {tool} is not installed or not in PATH")
            return False
    return True


def build_package():
    """Build the package."""
    commands = [
        ("python -m pip install --upgrade build", "Installing/upgrading build tools"),
        ("python -m build", "Building package"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def check_package():
    """Check the built package."""
    commands = [
        ("python -m pip install --upgrade twine", "Installing/upgrading twine"),
        ("python -m twine check dist/*", "Checking package"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def upload_to_pypi(test=True):
    """Upload to PyPI."""
    if test:
        repository = "https://test.pypi.org/legacy/"
        description = "Uploading to Test PyPI"
    else:
        repository = "https://upload.pypi.org/legacy/"
        description = "Uploading to PyPI"
    
    command = f"python -m twine upload --repository-url {repository} dist/*"
    
    if not run_command(command, description):
        return False
    
    if test:
        print("\n🎉 Package uploaded to Test PyPI successfully!")
        print("🔗 Test PyPI URL: https://test.pypi.org/project/quickinsights/")
        print("\n📝 To install from Test PyPI:")
        print("pip install --index-url https://test.pypi.org/simple/ quickinsights")
    else:
        print("\n🎉 Package uploaded to PyPI successfully!")
        print("🔗 PyPI URL: https://pypi.org/project/quickinsights/")
        print("\n📝 To install from PyPI:")
        print("pip install quickinsights")
    
    return True


def main():
    """Main function."""
    print("🚀 QuickInsights PyPI Publishing Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("❌ Required tools not found. Please install them first.")
        sys.exit(1)
    
    # Clean build directories
    print("\n🧹 Cleaning build directories...")
    clean_build_dirs()
    
    # Build package
    if not build_package():
        print("❌ Package build failed.")
        sys.exit(1)
    
    # Check package
    if not check_package():
        print("❌ Package check failed.")
        sys.exit(1)
    
    # Ask user for upload destination
    print("\n📤 Upload Options:")
    print("1. Upload to Test PyPI (recommended for first upload)")
    print("2. Upload to PyPI (production)")
    print("3. Build only (no upload)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        upload_to_pypi(test=True)
    elif choice == "2":
        confirm = input("⚠️  Are you sure you want to upload to production PyPI? (yes/no): ").strip().lower()
        if confirm == "yes":
            upload_to_pypi(test=False)
        else:
            print("❌ Upload cancelled.")
    elif choice == "3":
        print("✅ Package built successfully. No upload performed.")
    else:
        print("❌ Invalid choice.")
        sys.exit(1)
    
    print("\n🎉 Script completed successfully!")


if __name__ == "__main__":
    main()
