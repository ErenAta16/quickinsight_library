#!/usr/bin/env python3
"""
QuickInsights PyPI Deployment Script

Bu script, QuickInsights kütüphanesini PyPI'ya yüklemek için gerekli
adımları otomatikleştirir.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Tuple, Optional
import argparse


class PyPIDeployer:
    """PyPI deployment yöneticisi"""
    
    def __init__(self, test_pypi: bool = False):
        """
        PyPIDeployer başlatıcısı
        
        Parameters
        ----------
        test_pypi : bool
            Test PyPI'ya yüklemek isteyip istemediğiniz
        """
        self.test_pypi = test_pypi
        self.project_root = Path(__file__).parent.parent
        self.setup_py = self.project_root / "setup.py"
        self.dist_dir = self.project_root / "dist"
        
        # PyPI URL'leri
        if test_pypi:
            self.pypi_url = "https://test.pypi.org/legacy/"
            self.pypi_name = "Test PyPI"
        else:
            self.pypi_url = "https://upload.pypi.org/legacy/"
            self.pypi_name = "PyPI"
    
    def check_prerequisites(self) -> bool:
        """Gerekli araçların varlığını kontrol eder"""
        print("🔍 Gerekli araçlar kontrol ediliyor...")
        
        required_tools = ['python', 'pip', 'twine']
        missing_tools = []
        
        for tool in required_tools:
            try:
                subprocess.run([tool, '--version'], 
                             capture_output=True, check=True)
                print(f"   ✅ {tool}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
                print(f"   ❌ {tool}")
        
        if missing_tools:
            print(f"\n❌ Eksik araçlar: {', '.join(missing_tools)}")
            print("Lütfen eksik araçları yükleyin:")
            for tool in missing_tools:
                if tool == 'twine':
                    print(f"   pip install {tool}")
            return False
        
        print("✅ Tüm gerekli araçlar mevcut")
        return True
    
    def get_current_version(self) -> str:
        """setup.py'dan mevcut versiyonu okur"""
        try:
            with open(self.setup_py, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Version pattern'i bul
            version_match = re.search(r'version="([^"]+)"', content)
            if version_match:
                return version_match.group(1)
            else:
                raise ValueError("Version bulunamadı")
        except Exception as e:
            print(f"❌ Version okuma hatası: {e}")
            return "0.0.0"
    
    def update_version(self, new_version: str) -> bool:
        """setup.py'da versiyonu günceller"""
        try:
            with open(self.setup_py, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Version'ı güncelle
            old_pattern = r'version="([^"]+)"'
            new_content = re.sub(old_pattern, f'version="{new_version}"', content)
            
            with open(self.setup_py, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Version {new_version} olarak güncellendi")
            return True
        except Exception as e:
            print(f"❌ Version güncelleme hatası: {e}")
            return False
    
    def increment_version(self, current_version: str, increment_type: str) -> str:
        """Versiyonu artırır"""
        try:
            major, minor, patch = map(int, current_version.split('.'))
            
            if increment_type == 'major':
                major += 1
                minor = 0
                patch = 0
            elif increment_type == 'minor':
                minor += 1
                patch = 0
            elif increment_type == 'patch':
                patch += 1
            else:
                raise ValueError(f"Geçersiz increment type: {increment_type}")
            
            return f"{major}.{minor}.{patch}"
        except Exception as e:
            print(f"❌ Version artırma hatası: {e}")
            return current_version
    
    def clean_dist_directory(self) -> bool:
        """Dist dizinini temizler"""
        try:
            if self.dist_dir.exists():
                import shutil
                shutil.rmtree(self.dist_dir)
                print("✅ Dist dizini temizlendi")
            return True
        except Exception as e:
            print(f"❌ Dist dizini temizleme hatası: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Testleri çalıştırır"""
        print("\n🧪 Testler çalıştırılıyor...")
        
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/', 
                '--cov=quickinsights', '--cov-report=term-missing'
            ], cwd=self.project_root, check=True)
            
            print("✅ Tüm testler başarılı")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Test hatası: {e}")
            return False
    
    def check_code_quality(self) -> bool:
        """Kod kalitesi kontrolü yapar"""
        print("\n🔍 Kod kalitesi kontrol ediliyor...")
        
        # Black formatting check
        try:
            subprocess.run([
                'black', '--check', 'src/quickinsights/'
            ], cwd=self.project_root, check=True)
            print("   ✅ Black formatting")
        except subprocess.CalledProcessError:
            print("   ❌ Black formatting hatası")
            return False
        
        # Flake8 linting
        try:
            subprocess.run([
                'flake8', 'src/quickinsights/', 
                '--max-line-length=88', '--ignore=E203,W503'
            ], cwd=self.project_root, check=True)
            print("   ✅ Flake8 linting")
        except subprocess.CalledProcessError:
            print("   ⚠️  Flake8 linting uyarıları (devam ediliyor)")
        
        print("✅ Kod kalitesi kontrolü tamamlandı")
        return True
    
    def build_package(self) -> bool:
        """Paketi build eder"""
        print("\n🔨 Paket build ediliyor...")
        
        try:
            # Build komutu
            result = subprocess.run([
                'python', '-m', 'build'
            ], cwd=self.project_root, check=True)
            
            print("✅ Paket başarıyla build edildi")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Build hatası: {e}")
            return False
    
    def check_package(self) -> bool:
        """Build edilen paketi kontrol eder"""
        print("\n🔍 Build edilen paket kontrol ediliyor...")
        
        try:
            # Twine check
            result = subprocess.run([
                'twine', 'check', 'dist/*'
            ], cwd=self.project_root, check=True)
            
            print("✅ Paket kontrolü başarılı")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Paket kontrol hatası: {e}")
            return False
    
    def upload_to_pypi(self) -> bool:
        """Paketi PyPI'ya yükler"""
        print(f"\n📤 Paket {self.pypi_name}'ya yükleniyor...")
        
        try:
            # Upload komutu
            result = subprocess.run([
                'twine', 'upload', 'dist/*'
            ], cwd=self.project_root, check=True)
            
            print(f"✅ Paket başarıyla {self.pypi_name}'ya yüklendi!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Upload hatası: {e}")
            return False
    
    def deploy(self, increment_type: str = 'patch', force: bool = False) -> bool:
        """Ana deployment fonksiyonu"""
        print(f"🚀 QuickInsights {self.pypi_name} Deployment")
        print("=" * 50)
        
        # Gerekli araçları kontrol et
        if not self.check_prerequisites():
            return False
        
        # Mevcut versiyonu al
        current_version = self.get_current_version()
        print(f"📋 Mevcut versiyon: {current_version}")
        
        # Yeni versiyonu hesapla
        new_version = self.increment_version(current_version, increment_type)
        print(f"🆕 Yeni versiyon: {new_version}")
        
        if not force:
            # Kullanıcı onayı al
            response = input(f"\n{new_version} versiyonunu yüklemek istiyor musunuz? (y/N): ")
            if response.lower() != 'y':
                print("❌ Deployment iptal edildi")
                return False
        
        # Versiyonu güncelle
        if not self.update_version(new_version):
            return False
        
        # Testleri çalıştır
        if not self.run_tests():
            if not force:
                print("❌ Testler başarısız, deployment durduruldu")
                return False
            else:
                print("⚠️  Testler başarısız ama force=True, devam ediliyor")
        
        # Kod kalitesi kontrolü
        if not self.check_code_quality():
            if not force:
                print("❌ Kod kalitesi kontrolü başarısız, deployment durduruldu")
                return False
            else:
                print("⚠️  Kod kalitesi kontrolü başarısız ama force=True, devam ediliyor")
        
        # Dist dizinini temizle
        if not self.clean_dist_directory():
            return False
        
        # Paketi build et
        if not self.build_package():
            return False
        
        # Build edilen paketi kontrol et
        if not self.check_package():
            return False
        
        # PyPI'ya yükle
        if not self.upload_to_pypi():
            return False
        
        print(f"\n🎉 QuickInsights {new_version} başarıyla {self.pypi_name}'ya yüklendi!")
        print(f"📦 Kurulum: pip install quickinsights=={new_version}")
        
        return True


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='QuickInsights PyPI Deployment')
    parser.add_argument('--test', action='store_true', 
                       help='Test PyPI\'ya yükle')
    parser.add_argument('--increment', choices=['major', 'minor', 'patch'], 
                       default='patch', help='Version artırma tipi')
    parser.add_argument('--force', action='store_true', 
                       help='Hataları görmezden gel ve devam et')
    
    args = parser.parse_args()
    
    # Deployer oluştur
    deployer = PyPIDeployer(test_pypi=args.test)
    
    # Deployment'ı başlat
    success = deployer.deploy(
        increment_type=args.increment,
        force=args.force
    )
    
    if success:
        print("\n✅ Deployment başarıyla tamamlandı!")
        sys.exit(0)
    else:
        print("\n❌ Deployment başarısız!")
        sys.exit(1)


if __name__ == "__main__":
    main()
