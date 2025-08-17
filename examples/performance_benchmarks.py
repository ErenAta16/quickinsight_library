"""
QuickInsights - Performans Benchmark Örnekleri

Bu dosya, QuickInsights kütüphanesinin performans özelliklerini ve
farklı yaklaşımların karşılaştırmasını göstermektedir.
"""

import pandas as pd
import numpy as np
import time
import quickinsights as qi
import matplotlib.pyplot as plt

def create_large_dataset(n_rows=100000, n_cols=20):
    """Büyük veri seti oluşturur"""
    print(f"📊 {n_rows:,} satır x {n_cols} sütun veri seti oluşturuluyor...")
    
    np.random.seed(42)
    
    # Sayısal sütunlar
    numeric_data = {}
    for i in range(n_cols // 2):
        col_name = f'num_{i}'
        if i % 3 == 0:
            numeric_data[col_name] = np.random.normal(100, 25, n_rows)
        elif i % 3 == 1:
            numeric_data[col_name] = np.random.exponential(50, n_rows)
        else:
            numeric_data[col_name] = np.random.uniform(0, 1000, n_rows)
    
    # Kategorik sütunlar
    categorical_data = {}
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i in range(n_cols // 2):
        col_name = f'cat_{i}'
        categorical_data[col_name] = np.random.choice(categories, n_rows)
    
    # Veri setini birleştir
    all_data = {**numeric_data, **categorical_data}
    df = pd.DataFrame(all_data)
    
    print(f"✅ Veri seti oluşturuldu: {df.shape}")
    print(f"💾 Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def benchmark_analysis_speed(df):
    """Analiz hızını benchmark eder"""
    print("\n🚀 Analiz Hızı Benchmark'ı")
    print("=" * 40)
    
    # Standart analiz
    start_time = time.time()
    results_standard = qi.analyze(df, show_plots=False, save_plots=False)
    standard_time = time.time() - start_time
    
    # Lazy analyzer
    start_time = time.time()
    lazy_analyzer = qi.LazyAnalyzer(df)
    lazy_analyzer.compute()
    lazy_time = time.time() - start_time
    
    # Paralel analiz
    start_time = time.time()
    parallel_results = qi.parallel_analysis(df, n_jobs=4)
    parallel_time = time.time() - start_time
    
    # Chunked analiz
    start_time = time.time()
    chunk_results = qi.chunked_analysis(df, chunk_size=10000)
    chunk_time = time.time() - start_time
    
    print(f"⏱️  Standart analiz: {standard_time:.2f} saniye")
    print(f"⚡ Lazy analyzer: {lazy_time:.2f} saniye")
    print(f"🔄 Paralel analiz: {parallel_time:.2f} saniye")
    print(f"📦 Chunked analiz: {chunk_time:.2f} saniye")
    
    # Hızlanma oranları
    print(f"\n📈 Hızlanma Oranları:")
    print(f"Lazy vs Standart: {standard_time/lazy_time:.1f}x")
    print(f"Paralel vs Standart: {standard_time/parallel_time:.1f}x")
    print(f"Chunked vs Standart: {standard_time/chunk_time:.1f}x")
    
    return {
        'standard': standard_time,
        'lazy': lazy_time,
        'parallel': parallel_time,
        'chunked': chunk_time
    }

def benchmark_memory_optimization(df):
    """Bellek optimizasyonunu benchmark eder"""
    print("\n💾 Bellek Optimizasyonu Benchmark'ı")
    print("=" * 40)
    
    # Orijinal bellek kullanımı
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Veri tipi optimizasyonu
    start_time = time.time()
    optimized_df = qi.optimize_dtypes(df)
    optimization_time = time.time() - start_time
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
    
    # Bellek tasarrufu
    memory_savings = ((original_memory - optimized_memory) / original_memory) * 100
    
    print(f"💾 Orijinal bellek: {original_memory:.2f} MB")
    print(f"🔧 Optimize edilmiş bellek: {optimized_memory:.2f} MB")
    print(f"💰 Bellek tasarrufu: {memory_savings:.1f}%")
    print(f"⏱️  Optimizasyon süresi: {optimization_time:.2f} saniye")
    
    return {
        'original_memory': original_memory,
        'optimized_memory': optimized_memory,
        'savings_percent': memory_savings,
        'optimization_time': optimization_time
    }

def benchmark_numba_performance(df):
    """Numba JIT compilation performansını benchmark eder"""
    print("\n⚡ Numba JIT Compilation Benchmark'ı")
    print("=" * 40)
    
    # Numba durumunu kontrol et
    numba_status = qi.get_numba_status()
    print(f"🔧 Numba durumu: {'✅ Aktif' if numba_status else '❌ Pasif'}")
    
    if numba_status:
        # Numba vs Pandas karşılaştırması
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        numeric_df = df[numeric_cols]
        
        # Pandas ile korelasyon
        start_time = time.time()
        pandas_corr = numeric_df.corr()
        pandas_time = time.time() - start_time
        
        # Numba ile korelasyon
        start_time = time.time()
        numba_corr = qi.fast_correlation_matrix(numeric_df)
        numba_time = time.time() - start_time
        
        print(f"🐼 Pandas korelasyon: {pandas_time:.4f} saniye")
        print(f"⚡ Numba korelasyon: {numba_time:.4f} saniye")
        print(f"📈 Hızlanma: {pandas_time/numba_time:.1f}x")
        
        return {
            'pandas_time': pandas_time,
            'numba_time': numba_time,
            'speedup': pandas_time/numba_time
        }
    else:
        print("⚠️  Numba kullanılamıyor, benchmark atlanıyor.")
        return None

def benchmark_dask_performance(df):
    """Dask paralel işleme performansını benchmark eder"""
    print("\n🔄 Dask Paralel İşleme Benchmark'ı")
    print("=" * 40)
    
    # Dask durumunu kontrol et
    dask_status = qi.get_dask_status()
    print(f"🔧 Dask durumu: {'✅ Aktif' if dask_status else '❌ Pasif'}")
    
    if dask_status:
        # Pandas vs Dask karşılaştırması
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        numeric_df = df[numeric_cols]
        
        # Pandas ile özet istatistikler
        start_time = time.time()
        pandas_stats = numeric_df.describe()
        pandas_time = time.time() - start_time
        
        # Dask ile özet istatistikler
        start_time = time.time()
        dask_stats = qi.dask_analyze_large_dataset(numeric_df)
        dask_time = time.time() - start_time
        
        print(f"🐼 Pandas özet: {pandas_time:.4f} saniye")
        print(f"🔄 Dask özet: {dask_time:.4f} saniye")
        print(f"📈 Hızlanma: {pandas_time/dask_time:.1f}x")
        
        return {
            'pandas_time': pandas_time,
            'dask_time': dask_time,
            'speedup': pandas_time/dask_time
        }
    else:
        print("⚠️  Dask kullanılamıyor, benchmark atlanıyor.")
        return None

def benchmark_gpu_performance(df):
    """GPU hızlandırma performansını benchmark eder"""
    print("\n🚀 GPU Hızlandırma Benchmark'ı")
    print("=" * 40)
    
    # GPU durumunu kontrol et
    gpu_status = qi.get_gpu_status()
    print(f"🔧 GPU durumu: {'✅ Aktif' if gpu_status else '❌ Pasif'}")
    
    if gpu_status:
        # CPU vs GPU karşılaştırması
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        numeric_df = df[numeric_cols]
        
        # CPU ile korelasyon
        start_time = time.time()
        cpu_corr = numeric_df.corr()
        cpu_time = time.time() - start_time
        
        # GPU ile korelasyon
        start_time = time.time()
        gpu_corr = qi.gpu_correlation_matrix(numeric_df)
        gpu_time = time.time() - start_time
        
        print(f"💻 CPU korelasyon: {cpu_time:.4f} saniye")
        print(f"🚀 GPU korelasyon: {gpu_time:.4f} saniye")
        print(f"📈 Hızlanma: {cpu_time/gpu_time:.1f}x")
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time/gpu_time
        }
    else:
        print("⚠️  GPU kullanılamıyor, benchmark atlanıyor.")
        return None

def plot_benchmark_results(results):
    """Benchmark sonuçlarını görselleştirir"""
    print("\n📊 Benchmark Sonuçları Görselleştiriliyor...")
    
    # Analiz hızı sonuçları
    if 'analysis_speed' in results:
        speed_data = results['analysis_speed']
        methods = list(speed_data.keys())
        times = list(speed_data.values())
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.bar(methods, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Analiz Hızı Karşılaştırması')
        plt.ylabel('Süre (saniye)')
        plt.xticks(rotation=45)
        
        # Hızlanma oranları
        speedup_data = [speed_data['standard'] / t for t in times]
        plt.subplot(1, 2, 2)
        plt.bar(methods, speedup_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Hızlanma Oranları')
        plt.ylabel('Hızlanma (x)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    """Ana fonksiyon"""
    print("🚀 QuickInsights - Performans Benchmark Suite")
    print("=" * 60)
    
    try:
        # Farklı boyutlarda veri setleri oluştur
        datasets = {
            'Küçük': create_large_dataset(10000, 10),
            'Orta': create_large_dataset(50000, 15),
            'Büyük': create_large_dataset(100000, 20)
        }
        
        all_results = {}
        
        for size_name, df in datasets.items():
            print(f"\n{'='*20} {size_name} Veri Seti {'='*20}")
            
            # Analiz hızı benchmark'ı
            speed_results = benchmark_analysis_speed(df)
            all_results[f'{size_name}_speed'] = speed_results
            
            # Bellek optimizasyonu benchmark'ı
            memory_results = benchmark_memory_optimization(df)
            all_results[f'{size_name}_memory'] = memory_results
            
            # Numba benchmark'ı
            numba_results = benchmark_numba_performance(df)
            if numba_results:
                all_results[f'{size_name}_numba'] = numba_results
            
            # Dask benchmark'ı
            dask_results = benchmark_dask_performance(df)
            if dask_results:
                all_results[f'{size_name}_dask'] = dask_results
            
            # GPU benchmark'ı
            gpu_results = benchmark_gpu_performance(df)
            if gpu_results:
                all_results[f'{size_name}_gpu'] = gpu_results
        
        # Sonuçları görselleştir
        plot_benchmark_results(all_results)
        
        print("\n✅ Tüm benchmark'lar tamamlandı!")
        print("\n📊 Sonuçlar:")
        for key, value in all_results.items():
            print(f"- {key}: {value}")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
