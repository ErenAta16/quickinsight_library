#!/usr/bin/env python3
"""
QuickInsights v0.2.0 - Basic Showcase Demo
==========================================

Bu script, QuickInsights kütüphanesinin temel özelliklerini gösterir.
Kütüphaneyi hiç bilmeyen kullanıcılar için tasarlanmıştır.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("QuickInsights v0.2.0 Basic Showcase Demo Basliyor!")
print("=" * 60)
print()

# 1. VERİ HAZIRLAMA
print("1. VERİ HAZIRLAMA")
print("-" * 30)

# Satış verisi oluştur
np.random.seed(42)
n_records = 100

# Tarih aralığı
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_records)]

# Kategoriler
categories = ['Elektronik', 'Giyim', 'Ev & Bahce', 'Spor', 'Kitap']
regions = ['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Antalya']

# Demo veri
data = {
    'tarih': dates,
    'kategori': [random.choice(categories) for _ in range(n_records)],
    'bolge': [random.choice(regions) for _ in range(n_records)],
    'satis_miktari': np.random.normal(150, 50, n_records),
    'maliyet': np.random.normal(100, 30, n_records),
    'musteri_puani': np.random.uniform(1, 5, n_records),
    'indirim_orani': np.random.uniform(0, 0.3, n_records),
    'stok_miktari': np.random.poisson(50, n_records)
}

df = pd.DataFrame(data)
df['kar'] = df['satis_miktari'] - df['maliyet']
df['kar_marji'] = (df['kar'] / df['satis_miktari']) * 100

print(f"{len(df)} satir demo veri olusturuldu")
print(f"Tarih araligi: {df['tarih'].min().strftime('%Y-%m-%d')} - {df['tarih'].max().strftime('%Y-%m-%d')}")
print(f"Kategoriler: {', '.join(df['kategori'].unique())}")
print(f"Bolgeler: {', '.join(df['bolge'].unique())}")
print()

# 2. TEMEL VERİ ANALİZİ
print("2. TEMEL VERİ ANALİZİ")
print("-" * 30)

print("Veri Özeti:")
print(f"   Toplam Satis: {df['satis_miktari'].sum():,.2f}")
print(f"   Toplam Kar: {df['kar'].sum():,.2f}")
print(f"   Ortalama Musteri Puani: {df['musteri_puani'].mean():.2f}")
print(f"   En Iyi Kategori: {df.groupby('kategori')['kar'].sum().idxmax()}")
print(f"   En Iyi Bolge: {df.groupby('bolge')['kar'].mean().idxmax()}")
print()

print("Korelasyon Analizi:")
corr_matrix = df[['satis_miktari', 'kar', 'musteri_puani', 'indirim_orani']].corr()
print("   Korelasyon Matrisi:")
for i, col1 in enumerate(corr_matrix.columns):
    for j, col2 in enumerate(corr_matrix.columns):
        if i < j:
            corr_val = corr_matrix.iloc[i, j]
            print(f"     {col1} vs {col2}: {corr_val:.3f}")
print()

# 3. PANDAS ENTEGRASYONU
print("3. PANDAS ENTEGRASYONU")
print("-" * 30)

try:
    from quickinsights.pandas_integration import smart_group_analysis
    
    print("Akilli Grup Analizi...")
    group_result = smart_group_analysis(
        df,
        group_columns=['kategori', 'bolge'],
        value_columns=['satis_miktari', 'kar', 'musteri_puani'],
        output_dir='quickinsights_output',
        include_visualizations=False
    )
    print(f"✅ Grup analizi tamamlandi!")
    
except ImportError as e:
    print(f"⚠️  Pandas entegrasyonu yüklenemedi: {e}")
    print("💡 Çözüm: pip install quickinsights[pandas]")

print()

# 4. NUMPY ENTEGRASYONU
print("4. NUMPY ENTEGRASYONU")
print("-" * 30)

try:
    from quickinsights.numpy_integration import auto_math_analysis
    
    print("Otomatik Matematiksel Analiz...")
    numeric_data = df[['satis_miktari', 'kar', 'musteri_puani', 'indirim_orani']].values
    
    math_result = auto_math_analysis(
        numeric_data,
        analysis_types=['descriptive', 'correlation'],
        output_dir='quickinsights_output'
    )
    print(f"✅ Matematiksel analiz tamamlandi!")
    
except ImportError as e:
    print(f"⚠️  NumPy entegrasyonu yüklenemedi: {e}")
    print("💡 Çözüm: pip install quickinsights[numpy]")

print()

# 5. MAKİNE ÖĞRENMESİ
print("5. MAKİNE ÖĞRENMESİ")
print("-" * 30)

try:
    from quickinsights.ml_pipeline import auto_ml_pipeline
    
    print("Otomatik ML Pipeline...")
    
    df['yuksek_kar'] = (df['kar'] > df['kar'].median()).astype(int)
    
    feature_cols = ['satis_miktari', 'maliyet', 'musteri_puani', 'indirim_orani', 'stok_miktari']
    X = df[feature_cols].fillna(0)
    y = df['yuksek_kar']
    
    ml_result = auto_ml_pipeline(X, y, output_dir='quickinsights_output')
    print(f"✅ ML pipeline tamamlandi!")
    
except ImportError as e:
    print(f"⚠️  ML entegrasyonu yüklenemedi: {e}")
    print("💡 Çözüm: pip install quickinsights[ml]")

print()

# 6. PERFORMANCE ACCELERATION
print("6. PERFORMANCE ACCELERATION")
print("-" * 30)

try:
    from quickinsights.acceleration import gpu_available, memmap_array, chunked_apply
    
    print("GPU Availability Check...")
    gpu_status = gpu_available()
    print(f"✅ GPU durumu: {gpu_status}")
    
    print("Memory-Mapped Array...")
    try:
        memmap_array = memmap_array(
            'quickinsights_output/demo_data.mmap',
            dtype='float64',
            shape=df[['satis_miktari', 'kar']].values.shape
        )
        print(f"✅ Memory-mapped array olusturuldu!")
    except Exception as e:
        print(f"⚠️  Memory-mapped array hatası: {e}")
    
    print("Chunked Processing...")
    def chunk_function(chunk):
        return np.sum(chunk[:, 0])  # First column (satis_miktari)
    
    chunked_result = chunked_apply(
        chunk_function,
        df[['satis_miktari', 'kar']].values,
        chunk_rows=20
    )
    print(f"✅ Chunked processing tamamlandi!")
    
except ImportError as e:
    print(f"⚠️  Acceleration yüklenemedi: {e}")
    print("💡 Çözüm: pip install quickinsights[gpu]")

print()

# 7. ÖZET VE SONUÇ
print("7. ÖZET VE SONUÇ")
print("-" * 30)

print("QuickInsights v0.2.0 ile Neler Yapabilirsiniz:")
print()
print("VERİ ANALİZİ:")
print("   • Akıllı pivot tablolar")
print("   • Otomatik grup analizi")
print("   • Akıllı veri birleştirme")
print()
print("MATEMATİKSEL ANALİZ:")
print("   • Otomatik analiz seçimi")
print("   • Korelasyon analizi")
print("   • FFT ve eigenvalue analizi")
print()
print("MAKİNE ÖĞRENMESİ:")
print("   • Otomatik ML pipeline")
print("   • Akıllı özellik seçimi")
print("   • Model seçimi ve optimizasyon")
print()
print("BÜYÜK VERİ:")
print("   • Dask ile dağıtık işleme")
print("   • Chunked processing")
print("   • Memory optimization")
print()
print("NEURAL PATTERNS:")
print("   • Pattern mining")
print("   • Anomali tespiti")
print("   • Sequence signature extraction")
print()
print("QUANTUM INSIGHTS:")
print("   • Quantum-inspired sampling")
print("   • Amplitude PCA")
print("   • Quantum correlation mapping")
print()
print("HOLOGRAPHIC VIZ:")
print("   • 3D data embedding")
print("   • Volumetric density plots")
print("   • Interactive 3D visualizations")
print()
print("PERFORMANS:")
print("   • GPU acceleration")
print("   • Memory mapping")
print("   • Backend benchmarking")
print()

# KURULUM VE KULLANIM
print("KURULUM VE KULLANIM")
print("-" * 30)

print("📦 Kurulum:")
print("   pip install quickinsights[ml,quantum,gpu]")
print()
print("🐍 Python'da Kullanım:")
print("   from quickinsights.pandas_integration import smart_pivot_table")
print("   from quickinsights.neural_patterns import neural_pattern_mining")
print("   from quickinsights.quantum_insights import quantum_correlation_map")
print()
print("📚 Dokümantasyon:")
print("   https://github.com/ErenAta16/quickinsight_library/docs")
print()
print("🎯 PyPI Linki:")
print("   https://pypi.org/project/quickinsights/0.2.0/")
print()

print("🎉 Basic Showcase Demo Tamamlandı!")
print("=" * 60)
print("💡 QuickInsights ile veri analizi artık çok daha kolay!")
print("🚀 Geleceğin veri analizi kütüphanesi sizde!")
print("=" * 60)
