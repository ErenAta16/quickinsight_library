"""
QuickInsights Test Dosyası

Bu dosya, kütüphanenin temel fonksiyonlarını test eder.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Ana dizini Python path'ine ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import quickinsights as qi


class TestQuickInsights(unittest.TestCase):
    """QuickInsights kütüphanesi test sınıfı"""
    
    def setUp(self):
        """Test öncesi hazırlık"""
        # Test veri seti oluştur
        np.random.seed(42)
        n_samples = 100
        
        self.test_data = pd.DataFrame({
            'yas': np.random.normal(30, 5, n_samples),
            'maas': np.random.normal(40000, 10000, n_samples),
            'sehir': np.random.choice(['İstanbul', 'Ankara'], n_samples),
            'egitim': np.random.choice(['Lise', 'Üniversite'], n_samples)
        })
        
        # Geçici dizin oluştur
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test sonrası temizlik"""
        # Geçici dosyaları temizle
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_data_info(self):
        """Veri bilgisi alma fonksiyonunu test et"""
        info = qi.get_data_info(self.test_data)
        
        self.assertEqual(info['rows'], 100)
        self.assertEqual(info['columns'], 4)
        self.assertIn('yas', info)
        self.assertIn('maas', info)
    
    def test_detect_outliers(self):
        """Aykırı değer tespit fonksiyonunu test et"""
        numeric_data = self.test_data[['yas', 'maas']]
        outliers = qi.detect_outliers(numeric_data)
        
        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertEqual(outliers.shape, numeric_data.shape)
    
    def test_validate_dataframe(self):
        """DataFrame doğrulama fonksiyonunu test et"""
        # Geçerli DataFrame
        self.assertTrue(qi.validate_dataframe(self.test_data))
        
        # Boş DataFrame
        with self.assertRaises(ValueError):
            qi.validate_dataframe(pd.DataFrame())
        
        # Geçersiz tip
        with self.assertRaises(TypeError):
            qi.validate_dataframe("geçersiz veri")
    
    def test_summary_stats(self):
        """İstatistiksel özet fonksiyonunu test et"""
        numeric_data = self.test_data[['yas', 'maas']]
        summary = qi.summary_stats(numeric_data)
        
        self.assertIn('yas', summary)
        self.assertIn('maas', summary)
        self.assertIn('mean', summary['yas'])
        self.assertIn('std', summary['yas'])
    
    def test_correlation_matrix(self):
        """Korelasyon matrisi fonksiyonunu test et"""
        numeric_data = self.test_data[['yas', 'maas']]
        
        # Hata vermeden çalışmalı
        try:
            qi.correlation_matrix(numeric_data, save_plot=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Korelasyon matrisi hatası: {e}")
    
    def test_distribution_plots(self):
        """Dağılım grafikleri fonksiyonunu test et"""
        numeric_data = self.test_data[['yas', 'maas']]
        
        # Hata vermeden çalışmalı
        try:
            qi.distribution_plots(numeric_data, save_plots=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Dağılım grafikleri hatası: {e}")
    
    def test_analyze_numeric(self):
        """Sayısal analiz fonksiyonunu test et"""
        numeric_data = self.test_data[['yas', 'maas']]
        
        # Hata vermeden çalışmalı
        try:
            result = qi.analyze_numeric(numeric_data, show_plots=False)
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"Sayısal analiz hatası: {e}")
    
    def test_analyze_categorical(self):
        """Kategorik analiz fonksiyonunu test et"""
        categorical_data = self.test_data[['sehir', 'egitim']]
        
        # Hata vermeden çalışmalı
        try:
            result = qi.analyze_categorical(categorical_data, show_plots=False)
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"Kategorik analiz hatası: {e}")
    
    def test_analyze_main(self):
        """Ana analiz fonksiyonunu test et"""
        # Hata vermeden çalışmalı
        try:
            result = qi.analyze(self.test_data, show_plots=False)
            self.assertIsInstance(result, dict)
            self.assertIn('data_info', result)
            self.assertIn('numeric_columns', result)
            self.assertIn('categorical_columns', result)
        except Exception as e:
            self.fail(f"Ana analiz hatası: {e}")
    
    def test_output_directory_creation(self):
        """Çıktı dizini oluşturma fonksiyonunu test et"""
        test_dir = os.path.join(self.temp_dir, 'test_output')
        
        # Dizin yoksa oluştur
        created_dir = qi.create_output_directory(test_dir)
        self.assertEqual(created_dir, test_dir)
        self.assertTrue(os.path.exists(test_dir))
        
        # Dizin zaten varsa
        created_dir = qi.create_output_directory(test_dir)
        self.assertEqual(created_dir, test_dir)


if __name__ == '__main__':
    # Testleri çalıştır
    unittest.main(verbosity=2)
