#!/usr/bin/env python3
"""
QuickInsights Performance Testing Script

Bu script, QuickInsights kütüphanesinin performansını test eder ve
benchmark sonuçlarını raporlar.
"""

import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# QuickInsights import
try:
    import quickinsights as qi
    print("✅ QuickInsights başarıyla import edildi")
except ImportError as e:
    print(f"❌ QuickInsights import hatası: {e}")
    exit(1)


class PerformanceTester:
    """QuickInsights performans test edici"""
    
    def __init__(self):
        """PerformanceTester başlatıcısı"""
        self.results = {}
        self.memory_usage = {}
        self.execution_times = {}
        
    def create_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """Test veri setleri oluşturur"""
        print("📊 Test veri setleri oluşturuluyor...")
        
        datasets = {}
        
        # Küçük veri seti (1K satır)
        datasets['small'] = pd.DataFrame({
            'numeric_1': np.random.normal(100, 20, 1000),
            'numeric_2': np.random.exponential(50, 1000),
            'numeric_3': np.random.uniform(0, 1000, 1000),
            'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'categorical_2': np.random.choice(['X', 'Y', 'Z'], 1000)
        })
        
        # Orta veri seti (10K satır)
        datasets['medium'] = pd.DataFrame({
            'numeric_1': np.random.normal(100, 20, 10000),
            'numeric_2': np.random.exponential(50, 10000),
            'numeric_3': np.random.uniform(0, 1000, 10000),
            'numeric_4': np.random.poisson(30, 10000),
            'categorical_1': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
            'categorical_2': np.random.choice(['X', 'Y', 'Z', 'W'], 10000),
            'categorical_3': np.random.choice(['Red', 'Blue', 'Green'], 10000)
        })
        
        # Büyük veri seti (100K satır)
        datasets['large'] = pd.DataFrame({
            'numeric_1': np.random.normal(100, 20, 100000),
            'numeric_2': np.random.exponential(50, 100000),
            'numeric_3': np.random.uniform(0, 1000, 100000),
            'numeric_4': np.random.poisson(30, 100000),
            'numeric_5': np.random.beta(2, 5, 100000),
            'categorical_1': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], 100000),
            'categorical_2': np.random.choice(['X', 'Y', 'Z', 'W', 'V'], 100000),
            'categorical_3': np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], 100000)
        })
        
        print(f"✅ {len(datasets)} test veri seti oluşturuldu")
        for name, df in datasets.items():
            print(f"   📊 {name}: {df.shape[0]:,} satır x {df.shape[1]} sütun")
        
        return datasets
    
    def measure_memory_usage(self, func, *args, **kwargs) -> Tuple[float, float]:
        """Fonksiyon çalıştırma sırasında bellek kullanımını ölçer"""
        process = psutil.Process()
        
        # Başlangıç bellek kullanımı
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Fonksiyonu çalıştır
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Son bellek kullanımı
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return execution_time, final_memory - initial_memory
    
    def test_basic_analysis(self, df: pd.DataFrame, dataset_name: str):
        """Temel analiz performansını test eder"""
        print(f"\n🔍 {dataset_name} veri seti için temel analiz test ediliyor...")
        
        # qi.analyze testi
        time_analyze, memory_analyze = self.measure_memory_usage(
            qi.analyze, df, show_plots=False, save_plots=False
        )
        
        # qi.get_data_info testi
        time_info, memory_info = self.measure_memory_usage(
            qi.get_data_info, df
        )
        
        # qi.detect_outliers testi
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            time_outliers, memory_outliers = self.measure_memory_usage(
                qi.detect_outliers, df[numeric_cols]
            )
        else:
            time_outliers, memory_outliers = 0, 0
        
        # qi.optimize_dtypes testi
        time_optimize, memory_optimize = self.measure_memory_usage(
            qi.optimize_dtypes, df
        )
        
        self.results[f'{dataset_name}_basic'] = {
            'analyze': {'time': time_analyze, 'memory': memory_analyze},
            'get_data_info': {'time': time_info, 'memory': memory_info},
            'detect_outliers': {'time': time_outliers, 'memory': memory_outliers},
            'optimize_dtypes': {'time': time_optimize, 'memory': memory_optimize}
        }
        
        print(f"✅ Temel analiz testi tamamlandı")
        print(f"   ⏱️  analyze: {time_analyze:.3f}s, {memory_analyze:.2f}MB")
        print(f"   ⏱️  get_data_info: {time_info:.3f}s, {memory_info:.2f}MB")
        print(f"   ⏱️  detect_outliers: {time_outliers:.3f}s, {memory_outliers:.2f}MB")
        print(f"   ⏱️  optimize_dtypes: {time_optimize:.3f}s, {memory_optimize:.2f}MB")
    
    def test_lazy_analyzer(self, df: pd.DataFrame, dataset_name: str):
        """LazyAnalyzer performansını test eder"""
        print(f"\n🚀 {dataset_name} veri seti için LazyAnalyzer test ediliyor...")
        
        # LazyAnalyzer oluştur
        lazy_analyzer = qi.LazyAnalyzer(df)
        
        # get_data_info testi
        time_info, memory_info = self.measure_memory_usage(
            lazy_analyzer.get_data_info
        )
        
        # get_numeric_analysis testi
        time_numeric, memory_numeric = self.measure_memory_usage(
            lazy_analyzer.get_numeric_analysis
        )
        
        # get_categorical_analysis testi
        time_categorical, memory_categorical = self.measure_memory_usage(
            lazy_analyzer.get_categorical_analysis
        )
        
        # get_all_analysis testi
        time_all, memory_all = self.measure_memory_usage(
            lazy_analyzer.get_all_analysis
        )
        
        self.results[f'{dataset_name}_lazy'] = {
            'get_data_info': {'time': time_info, 'memory': memory_info},
            'get_numeric_analysis': {'time': time_numeric, 'memory': memory_numeric},
            'get_categorical_analysis': {'time': time_categorical, 'memory': memory_categorical},
            'get_all_analysis': {'time': time_all, 'memory': memory_all}
        }
        
        print(f"✅ LazyAnalyzer testi tamamlandı")
        print(f"   ⏱️  get_data_info: {time_info:.3f}s, {memory_info:.2f}MB")
        print(f"   ⏱️  get_numeric_analysis: {time_numeric:.3f}s, {memory_numeric:.2f}MB")
        print(f"   ⏱️  get_categorical_analysis: {time_categorical:.3f}s, {memory_categorical:.2f}MB")
        print(f"   ⏱️  get_all_analysis: {time_all:.3f}s, {memory_all:.2f}MB")
    
    def test_parallel_analysis(self, df: pd.DataFrame, dataset_name: str):
        """Paralel analiz performansını test eder"""
        print(f"\n🔄 {dataset_name} veri seti için paralel analiz test ediliyor...")
        
        # Thread backend testi
        time_thread, memory_thread = self.measure_memory_usage(
            qi.parallel_analysis, df, backend='thread', n_jobs=4
        )
        
        # Process backend testi (sadece orta ve büyük veri setleri için)
        if dataset_name in ['medium', 'large']:
            time_process, memory_process = self.measure_memory_usage(
                qi.parallel_analysis, df, backend='process', n_jobs=4
            )
        else:
            time_process, memory_process = 0, 0
        
        self.results[f'{dataset_name}_parallel'] = {
            'thread': {'time': time_thread, 'memory': memory_thread},
            'process': {'time': time_process, 'memory': memory_process}
        }
        
        print(f"✅ Paralel analiz testi tamamlandı")
        print(f"   ⏱️  thread: {time_thread:.3f}s, {memory_thread:.2f}MB")
        if time_process > 0:
            print(f"   ⏱️  process: {time_process:.3f}s, {memory_process:.2f}MB")
    
    def test_chunked_analysis(self, df: pd.DataFrame, dataset_name: str):
        """Chunked analiz performansını test eder"""
        print(f"\n📦 {dataset_name} veri seti için chunked analiz test ediliyor...")
        
        # Farklı chunk boyutları test et
        chunk_sizes = [1000, 5000, 10000]
        
        chunk_results = {}
        for chunk_size in chunk_sizes:
            if chunk_size < len(df):
                time_chunk, memory_chunk = self.measure_memory_usage(
                    qi.chunked_analysis, df, chunk_size=chunk_size, n_jobs=4
                )
                chunk_results[chunk_size] = {'time': time_chunk, 'memory': memory_chunk}
            else:
                chunk_results[chunk_size] = {'time': 0, 'memory': 0}
        
        self.results[f'{dataset_name}_chunked'] = chunk_results
        
        print(f"✅ Chunked analiz testi tamamlandı")
        for chunk_size, result in chunk_results.items():
            if result['time'] > 0:
                print(f"   ⏱️  chunk_size={chunk_size}: {result['time']:.3f}s, {result['memory']:.2f}MB")
    
    def test_visualization_performance(self, df: pd.DataFrame, dataset_name: str):
        """Görselleştirme performansını test eder"""
        print(f"\n📈 {dataset_name} veri seti için görselleştirme test ediliyor...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # correlation_matrix testi
            time_corr, memory_corr = self.measure_memory_usage(
                qi.correlation_matrix, df[numeric_cols], save_plots=False
            )
            
            # distribution_plots testi
            time_dist, memory_dist = self.measure_memory_usage(
                qi.distribution_plots, df[numeric_cols], save_plots=False
            )
            
            self.results[f'{dataset_name}_visualization'] = {
                'correlation_matrix': {'time': time_corr, 'memory': memory_corr},
                'distribution_plots': {'time': time_dist, 'memory': memory_dist}
            }
            
            print(f"✅ Görselleştirme testi tamamlandı")
            print(f"   ⏱️  correlation_matrix: {time_corr:.3f}s, {memory_corr:.2f}MB")
            print(f"   ⏱️  distribution_plots: {time_dist:.3f}s, {memory_dist:.2f}MB")
        else:
            print(f"⚠️  Görselleştirme testi atlandı (yeterli sayısal sütun yok)")
    
    def run_all_tests(self):
        """Tüm performans testlerini çalıştırır"""
        print("🚀 QuickInsights Performans Test Suite Başlıyor...")
        print("=" * 60)
        
        # Test veri setleri oluştur
        datasets = self.create_test_datasets()
        
        # Her veri seti için testleri çalıştır
        for dataset_name, df in datasets.items():
            print(f"\n{'='*20} {dataset_name.upper()} VERİ SETİ {'='*20}")
            
            # Temel analiz testi
            self.test_basic_analysis(df, dataset_name)
            
            # LazyAnalyzer testi
            self.test_lazy_analyzer(df, dataset_name)
            
            # Paralel analiz testi
            self.test_parallel_analysis(df, dataset_name)
            
            # Chunked analiz testi
            self.test_chunked_analysis(df, dataset_name)
            
            # Görselleştirme testi
            self.test_visualization_performance(df, dataset_name)
        
        print(f"\n{'='*60}")
        print("🎉 Tüm performans testleri tamamlandı!")
        print("📊 Sonuçlar analiz ediliyor...")
    
    def generate_performance_report(self):
        """Performans raporu oluşturur"""
        print("\n📊 PERFORMANS RAPORU")
        print("=" * 40)
        
        # Genel istatistikler
        total_tests = len(self.results)
        print(f"📈 Toplam test sayısı: {total_tests}")
        
        # En hızlı ve en yavaş fonksiyonlar
        all_times = []
        for test_name, test_results in self.results.items():
            for func_name, func_results in test_results.items():
                if isinstance(func_results, dict) and 'time' in func_results:
                    all_times.append({
                        'test': test_name,
                        'function': func_name,
                        'time': func_results['time']
                    })
        
        if all_times:
            # En hızlı
            fastest = min(all_times, key=lambda x: x['time'])
            print(f"⚡ En hızlı: {fastest['function']} ({fastest['test']}) - {fastest['time']:.3f}s")
            
            # En yavaş
            slowest = max(all_times, key=lambda x: x['time'])
            print(f"🐌 En yavaş: {slowest['function']} ({slowest['test']}) - {slowest['time']:.3f}s")
            
            # Ortalama süre
            avg_time = sum(x['time'] for x in all_times) / len(all_times)
            print(f"📊 Ortalama süre: {avg_time:.3f}s")
        
        # Bellek kullanımı analizi
        all_memory = []
        for test_name, test_results in self.results.items():
            for func_name, func_results in test_results.items():
                if isinstance(func_results, dict) and 'memory' in func_results:
                    all_memory.append({
                        'test': test_name,
                        'function': func_name,
                        'memory': func_results['memory']
                    })
        
        if all_memory:
            # En az bellek
            min_memory = min(all_memory, key=lambda x: x['memory'])
            print(f"💾 En az bellek: {min_memory['function']} ({min_memory['test']}) - {min_memory['memory']:.2f}MB")
            
            # En çok bellek
            max_memory = max(all_memory, key=lambda x: x['memory'])
            print(f"🔥 En çok bellek: {max_memory['function']} ({max_memory['test']}) - {max_memory['memory']:.2f}MB")
            
            # Ortalama bellek
            avg_memory = sum(x['memory'] for x in all_memory) / len(all_memory)
            print(f"📊 Ortalama bellek: {avg_memory:.2f}MB")
        
        return self.results


def main():
    """Ana fonksiyon"""
    print("🚀 QuickInsights Performance Testing Suite")
    print("=" * 50)
    
    # Performance tester oluştur
    tester = PerformanceTester()
    
    try:
        # Tüm testleri çalıştır
        tester.run_all_tests()
        
        # Performans raporu oluştur
        results = tester.generate_performance_report()
        
        print(f"\n✅ Performans testi başarıyla tamamlandı!")
        print(f"📁 Sonuçlar: {len(results)} test kategorisi")
        
        # Sonuçları JSON olarak kaydet
        import json
        with open('performance_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Sonuçlar 'performance_results.json' dosyasına kaydedildi")
        
    except Exception as e:
        print(f"❌ Performans testi sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
