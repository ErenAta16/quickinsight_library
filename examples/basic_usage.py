"""
QuickInsights - Temel Kullanım Örneği

Bu dosya, QuickInsights kütüphanesinin temel özelliklerini göstermektedir.
"""

import pandas as pd
import numpy as np
import quickinsights as qi

def create_sample_data():
    """Örnek veri seti oluşturur"""
    np.random.seed(42)
    
    # Sayısal değişkenler
    n_samples = 1000
    data = {
        'yas': np.random.normal(35, 10, n_samples),
        'maas': np.random.normal(50000, 15000, n_samples),
        'deneyim': np.random.exponential(5, n_samples),
        'performans': np.random.uniform(0, 100, n_samples),
        'sehir': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa'], n_samples),
        'egitim': np.random.choice(['Lise', 'Üniversite', 'Yüksek Lisans', 'Doktora'], n_samples),
        'cinsiyet': np.random.choice(['Erkek', 'Kadın'], n_samples)
    }
    
    # Aykırı değerler ekle
    data['maas'][:10] = np.random.normal(150000, 20000, 10)  # Yüksek maaş
    data['yas'][:5] = np.random.normal(70, 5, 5)  # Yaşlı çalışanlar
    
    # Eksik değerler ekle
    data['performans'][:50] = np.nan
    
    return pd.DataFrame(data)

def basic_analysis_example():
    """Temel analiz örneği"""
    print("🚀 QuickInsights - Temel Analiz Örneği")
    print("=" * 50)
    
    # Veri setini oluştur
    df = create_sample_data()
    print(f"📊 Veri seti oluşturuldu: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # Temel analiz
    print("\n🔍 Kapsamlı analiz başlıyor...")
    results = qi.analyze(df, show_plots=True, save_plots=False)
    
    return results

def numeric_analysis_example():
    """Sayısal değişken analizi örneği"""
    print("\n🔢 Sayısal Değişken Analizi")
    print("-" * 30)
    
    df = create_sample_data()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Sayısal değişkenleri analiz et
    numeric_results = qi.analyze_numeric(df[numeric_cols])
    
    # Aykırı değerleri tespit et
    outliers = qi.detect_outliers(df[numeric_cols])
    print(f"⚠️  Tespit edilen aykırı değer sayısı: {outliers.sum().sum()}")
    
    return numeric_results

def categorical_analysis_example():
    """Kategorik değişken analizi örneği"""
    print("\n🏷️  Kategorik Değişken Analizi")
    print("-" * 30)
    
    df = create_sample_data()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Kategorik değişkenleri analiz et
    categorical_results = qi.analyze_categorical(df[categorical_cols])
    
    return categorical_results

def lazy_analyzer_example():
    """Lazy Analyzer örneği"""
    print("\n⚡ Lazy Analyzer Örneği")
    print("-" * 30)
    
    df = create_sample_data()
    
    # Lazy analyzer oluştur
    lazy_analyzer = qi.LazyAnalyzer(df)
    
    # Sadece gerekli analizleri yap
    print("📊 Veri seti bilgileri alınıyor...")
    data_info = lazy_analyzer.get_data_info()
    print(f"Satır sayısı: {data_info['rows']:,}")
    print(f"Bellek kullanımı: {data_info['memory_usage']:.2f} MB")
    
    print("\n🔢 Sayısal analiz yapılıyor...")
    numeric_analysis = lazy_analyzer.get_numeric_analysis()
    
    print("\n🏷️  Kategorik analiz yapılıyor...")
    categorical_analysis = lazy_analyzer.get_categorical_analysis()
    
    # Tüm analizleri yap
    print("\n🚀 Tüm analizler hesaplanıyor...")
    all_results = lazy_analyzer.compute()
    
    return all_results

def performance_optimization_example():
    """Performans optimizasyonu örneği"""
    print("\n⚡ Performans Optimizasyonu")
    print("-" * 30)
    
    df = create_sample_data()
    
    # Veri tipi optimizasyonu
    print("🔧 Veri tipi optimizasyonu yapılıyor...")
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    optimized_df = qi.optimize_dtypes(df)
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Orijinal bellek kullanımı: {original_memory:.2f} MB")
    print(f"Optimize edilmiş bellek kullanımı: {optimized_memory:.2f} MB")
    print(f"Bellek tasarrufu: {((original_memory - optimized_memory) / original_memory * 100):.1f}%")
    
    return optimized_df

def main():
    """Ana fonksiyon"""
    try:
        # Temel analiz
        basic_results = basic_analysis_example()
        
        # Sayısal analiz
        numeric_results = numeric_analysis_example()
        
        # Kategorik analiz
        categorical_results = categorical_analysis_example()
        
        # Lazy analyzer
        lazy_results = lazy_analyzer_example()
        
        # Performans optimizasyonu
        optimized_df = performance_optimization_example()
        
        print("\n✅ Tüm örnekler başarıyla tamamlandı!")
        print("\n📚 Daha fazla örnek için:")
        print("- examples/advanced_usage.py")
        print("- examples/performance_benchmarks.py")
        print("- examples/big_data_analysis.py")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("🔧 Hata detayları için lütfen log'ları kontrol edin.")

if __name__ == "__main__":
    main()
