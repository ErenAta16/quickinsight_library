#!/usr/bin/env python3
"""
QuickInsights Deployment Script
Bu script API key'leri .env dosyasından okur ve PyPI'ye yükler.
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ .env dosyası yüklendi")
    else:
        print("❌ .env dosyası bulunamadı!")
        print("Lütfen .env dosyası oluşturun ve API key'leri ekleyin.")
        sys.exit(1)

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import twine
        print("✅ twine kurulu")
    except ImportError:
        print("❌ twine kurulu değil!")
        print("Kurulum: pip install twine")
        sys.exit(1)

def build_package():
    """Build the package"""
    print("🔨 Paket build ediliyor...")
    try:
        subprocess.run([sys.executable, "-m", "build"], check=True)
        print("✅ Paket başarıyla build edildi")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build hatası: {e}")
        sys.exit(1)

def upload_to_test_pypi():
    """Upload to Test PyPI"""
    print("🚀 Test PyPI'ye yükleniyor...")
    
    test_token = os.getenv('TEST_PYPI_TOKEN')
    if not test_token:
        print("❌ TEST_PYPI_TOKEN bulunamadı!")
        return False
    
    try:
        cmd = [
            sys.executable, "-m", "twine", "upload",
            "--repository", "testpypi",
            "dist/*",
            "--username", "__token__",
            "--password", test_token
        ]
        subprocess.run(cmd, check=True)
        print("✅ Test PyPI'ye başarıyla yüklendi!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test PyPI yükleme hatası: {e}")
        return False

def upload_to_main_pypi():
    """Upload to Main PyPI"""
    print("🚀 Ana PyPI'ye yükleniyor...")
    
    main_token = os.getenv('MAIN_PYPI_TOKEN')
    if not main_token:
        print("❌ MAIN_PYPI_TOKEN bulunamadı!")
        return False
    
    try:
        cmd = [
            sys.executable, "-m", "twine", "upload",
            "dist/*",
            "--username", "__token__",
            "--password", main_token
        ]
        subprocess.run(cmd, check=True)
        print("✅ Ana PyPI'ye başarıyla yüklendi!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ana PyPI yükleme hatası: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 QuickInsights Deployment Script")
    print("=" * 50)
    
    # Load environment
    load_environment()
    
    # Check dependencies
    check_dependencies()
    
    # Build package
    build_package()
    
    # Ask user which repository to upload to
    print("\n📋 Hangi repository'ye yüklemek istiyorsunuz?")
    print("1. Test PyPI (önerilen)")
    print("2. Ana PyPI")
    print("3. Her ikisine")
    
    choice = input("\nSeçiminiz (1-3): ").strip()
    
    if choice == "1":
        upload_to_test_pypi()
    elif choice == "2":
        upload_to_main_pypi()
    elif choice == "3":
        upload_to_test_pypi()
        if input("\nTest PyPI'de sorun yoksa Ana PyPI'ye devam edelim mi? (y/n): ").lower() == 'y':
            upload_to_main_pypi()
    else:
        print("❌ Geçersiz seçim!")
        sys.exit(1)
    
    print("\n🎉 Deployment tamamlandı!")

if __name__ == "__main__":
    main()
