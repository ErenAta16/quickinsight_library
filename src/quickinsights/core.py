"""
QuickInsights ana analiz modülü

Bu modül, veri setleri üzerinde kapsamlı analiz yapan ana fonksiyonları içerir.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from .utils import get_data_info, detect_outliers
from .visualizer import correlation_matrix, distribution_plots, summary_stats


def analyze(df: pd.DataFrame, 
           show_plots: bool = True, 
           save_plots: bool = False,
           output_dir: str = "./quickinsights_output") -> dict:
    """
    Veri seti üzerinde kapsamlı analiz yapar.
    
    Bu fonksiyon, veri seti hakkında genel bilgi, istatistiksel özetler,
    korelasyon analizi ve görselleştirmeler sunar.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    show_plots : bool, default=True
        Grafikleri göstermek isteyip istemediğiniz
    save_plots : bool, default=False
        Grafikleri kaydetmek isteyip istemediğiniz
    output_dir : str, default="./quickinsights_output"
        Grafiklerin kaydedileceği dizin
        
    Returns
    -------
    dict
        Analiz sonuçlarını içeren sözlük
        
    Examples
    --------
    >>> import quickinsights as qi
    >>> import pandas as pd
    >>> df = pd.read_csv('data.csv')
    >>> results = qi.analyze(df)
    """
    
    if df.empty:
        raise ValueError("Veri seti boş olamaz!")
    
    print("🔍 QuickInsights - Veri Seti Analizi Başlıyor...")
    print("=" * 60)
    
    # Veri seti genel bilgileri
    print("\n📊 VERİ SETİ GENEL BİLGİLERİ")
    print("-" * 40)
    data_info = get_data_info(df)
    print(f"Satır sayısı: {data_info['rows']:,}")
    print(f"Sütun sayısı: {data_info['columns']}")
    print(f"Bellek kullanımı: {data_info['memory_usage']:.2f} MB")
    print(f"Veri tipleri: {data_info['dtypes']}")
    
    # Eksik değer analizi
    print(f"\n❌ Eksik değerler: {data_info['missing_values']:,}")
    if data_info['missing_values'] > 0:
        missing_percent = (data_info['missing_values'] / (data_info['rows'] * data_info['columns'])) * 100
        print(f"Eksik değer oranı: {missing_percent:.2f}%")
    
    # Sayısal değişken analizi - DataFrame kopyalama yapmadan
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = None
    
    if len(numeric_cols) > 0:
        print(f"\n🔢 Sayısal değişkenler ({len(numeric_cols)}): {', '.join(numeric_cols)}")
        numeric_analysis = analyze_numeric(df[numeric_cols], show_plots=False)
        
        # Aykırı değer tespiti
        outliers = detect_outliers(df[numeric_cols])
        if outliers.any().any():
            print(f"⚠️  Aykırı değerler tespit edildi!")
    
    # Kategorik değişken analizi - DataFrame kopyalama yapmadan
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\n🏷️  Kategorik değişkenler ({len(categorical_cols)}): {', '.join(categorical_cols)}")
        categorical_analysis = analyze_categorical(df[categorical_cols], show_plots=False)
    
    # Görselleştirmeler
    if show_plots:
        print(f"\n📈 Görselleştirmeler oluşturuluyor...")
        
        # Korelasyon matrisi (sadece sayısal değişkenler için)
        if len(numeric_cols) > 1:
            correlation_matrix(df[numeric_cols], save_plot=save_plots, output_dir=output_dir)
        
        # Dağılım grafikleri
        if len(numeric_cols) > 0:
            distribution_plots(df[numeric_cols], save_plots=save_plots, output_dir=output_dir)
    
    print("\n✅ Analiz tamamlandı!")
    print("=" * 60)
    
    # Sonuçları döndür
    return {
        'data_info': data_info,
        'numeric_columns': list(numeric_cols),
        'categorical_columns': list(categorical_cols),
        'outliers_detected': outliers.any().any() if outliers is not None else False
    }


def analyze_numeric(df: pd.DataFrame, 
                   show_plots: bool = True,
                   save_plots: bool = False,
                   output_dir: str = "./quickinsights_output") -> dict:
    """
    Sayısal değişkenler üzerinde detaylı analiz yapar.
    
    Parameters
    ----------
    df : pd.DataFrame
        Sadece sayısal değişkenler içeren veri seti
    show_plots : bool, default=True
        Grafikleri göstermek isteyip istemediğiniz
    save_plots : bool, default=False
        Grafikleri kaydetmek isteyip istemediğiniz
    output_dir : str, default="./quickinsights_output"
        Grafiklerin kaydedileceği dizin
        
    Returns
    -------
    dict
        Sayısal analiz sonuçları
    """
    
    if df.empty:
        print("⚠️  Sayısal değişken bulunamadı!")
        return {}
    
    print(f"\n🔢 SAYISAL DEĞİŞKEN ANALİZİ ({len(df.columns)} değişken)")
    print("-" * 50)
    
    # İstatistiksel özet
    summary = summary_stats(df)
    
    # Vectorized printing - tüm kolonları aynı anda işle
    col_names = df.columns.tolist()
    means = [summary[col]['mean'] for col in col_names]
    medians = [summary[col]['median'] for col in col_names]
    stds = [summary[col]['std'] for col in col_names]
    mins = [summary[col]['min'] for col in col_names]
    maxs = [summary[col]['max'] for col in col_names]
    q1s = [summary[col]['q1'] for col in col_names]
    q3s = [summary[col]['q3'] for col in col_names]
    
    # Batch printing
    for i, col in enumerate(col_names):
        print(f"\n📊 {col}:")
        print(f"   Ortalama: {means[i]:.4f}")
        print(f"   Medyan: {medians[i]:.4f}")
        print(f"   Standart sapma: {stds[i]:.4f}")
        print(f"   Minimum: {mins[i]:.4f}")
        print(f"   Maksimum: {maxs[i]:.4f}")
        print(f"   Çeyrekler: Q1={q1s[i]:.4f}, Q3={q3s[i]:.4f}")
    
    # Görselleştirmeler
    if show_plots:
        distribution_plots(df, save_plots=save_plots, output_dir=output_dir)
    
    return summary


def analyze_categorical(df: pd.DataFrame, 
                       show_plots: bool = True,
                       save_plots: bool = False,
                       output_dir: str = "./quickinsights_output") -> dict:
    """
    Kategorik değişkenler üzerinde detaylı analiz yapar.
    
    Parameters
    ----------
    df : pd.DataFrame
        Sadece kategorik değişkenler içeren veri seti
    show_plots : bool, default=True
        Grafikleri göstermek isteyip istemediğiniz
    save_plots : bool, default=False
        Grafikleri kaydetmek isteyip istemediğiniz
    output_dir : str, default="./quickinsights_output"
        Grafiklerin kaydedileceği dizin
        
    Returns
    -------
    dict
        Kategorik analiz sonuçları
    """
    
    if df.empty:
        print("⚠️  Kategorik değişken bulunamadı!")
        return {}
    
    print(f"\n🏷️  KATEGORİK DEĞİŞKEN ANALİZİ ({len(df.columns)} değişken)")
    print("-" * 50)
    
    # Vectorized operations - tüm kolonları aynı anda işle
    col_names = df.columns.tolist()
    
    # Tüm kolonlar için value_counts'ları aynı anda hesapla
    value_counts_list = [df[col].value_counts() for col in col_names]
    missing_counts = df.isnull().sum()
    
    results = {}
    
    # Batch processing - tüm kolonları aynı anda işle
    for i, col in enumerate(col_names):
        value_counts = value_counts_list[i]
        missing = missing_counts[col]
        
        print(f"\n📊 {col}:")
        print(f"   Benzersiz değer sayısı: {len(value_counts)}")
        print(f"   En yaygın değer: '{value_counts.index[0]}' ({value_counts.iloc[0]} kez)")
        
        if missing > 0:
            print(f"   Eksik değerler: {missing}")
        
        print(f"   İlk 5 değer: {list(value_counts.head().index)}")
        
        results[col] = {
            'unique_count': len(value_counts),
            'most_common': value_counts.index[0],
            'most_common_count': value_counts.iloc[0],
            'missing_count': missing,
            'value_counts': value_counts
        }
    
    return results


class LazyAnalyzer:
    """
    Lazy evaluation ile veri analizi yapan sınıf.
    
    Bu sınıf, analizleri sadece gerektiğinde yapar ve sonuçları cache'ler.
    Böylece tekrar analizler çok daha hızlı olur.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        LazyAnalyzer'ı başlatır.
        
        Parameters
        ----------
        df : pd.DataFrame
            Analiz edilecek veri seti
        """
        self.df = df
        self._results = {}
        self._data_info = None
        self._numeric_analysis = None
        self._categorical_analysis = None
        self._correlation_matrix = None
        self._outliers = None
        
        # DataFrame kopyalama yapmadan kolon tiplerini belirle
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns
        self._categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        print("🚀 LazyAnalyzer başlatıldı!")
        print(f"   📊 Veri seti boyutu: {df.shape}")
        print(f"   💾 Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    def get_data_info(self):
        """Veri seti genel bilgilerini döndürür (lazy)"""
        if self._data_info is None:
            print("🔍 Veri seti bilgileri hesaplanıyor...")
            self._data_info = get_data_info(self.df)
        return self._data_info
    
    def get_numeric_analysis(self):
        """Sayısal analiz sonuçlarını döndürür (lazy)"""
        if self._numeric_analysis is None:
            print("🔢 Sayısal analiz yapılıyor...")
            if len(self._numeric_cols) > 0:
                self._numeric_analysis = analyze_numeric(self.df[self._numeric_cols], show_plots=False)
            else:
                self._numeric_analysis = {}
        return self._numeric_analysis
    
    def get_categorical_analysis(self):
        """Kategorik analiz sonuçlarını döndürür (lazy)"""
        if self._categorical_analysis is None:
            print("🏷️  Kategorik analiz yapılıyor...")
            if len(self._categorical_cols) > 0:
                self._categorical_analysis = analyze_categorical(self.df[self._categorical_cols], show_plots=False)
            else:
                self._categorical_analysis = {}
        return self._categorical_analysis
    
    def get_correlation_matrix(self):
        """Korelasyon matrisini döndürür (lazy)"""
        if self._correlation_matrix is None:
            print("📊 Korelasyon matrisi hesaplanıyor...")
            if len(self._numeric_cols) > 1:
                # Korelasyon hesaplama
                self._correlation_matrix = self.df[self._numeric_cols].corr()
            else:
                self._correlation_matrix = pd.DataFrame()
        return self._correlation_matrix
    
    def get_outliers(self, method: str = 'iqr', threshold: float = 1.5):
        """Aykırı değerleri döndürür (lazy)"""
        if self._outliers is None:
            print("⚠️  Aykırı değerler tespit ediliyor...")
            if len(self._numeric_cols) > 0:
                self._outliers = detect_outliers(self.df[self._numeric_cols], method=method, threshold=threshold)
            else:
                self._outliers = {}
        return self._outliers
    
    def compute(self):
        """Tüm analizleri yapar ve sonuçları döndürür"""
        print("🚀 Tüm analizler yapılıyor...")
        
        results = {
            'data_info': self.get_data_info(),
            'numeric_analysis': self.get_numeric_analysis(),
            'categorical_analysis': self.get_categorical_analysis(),
            'correlation_matrix': self.get_correlation_matrix(),
            'outliers': self.get_outliers()
        }
        
        print("✅ Tüm analizler tamamlandı!")
        return results
    
    def get_summary(self):
        """Tüm analizlerin özetini döndürür"""
        print("📋 Tüm analizler yapılıyor...")
        
        summary = {
            'data_info': self.get_data_info(),
            'numeric_analysis': self.get_numeric_analysis(),
            'categorical_analysis': self.get_categorical_analysis(),
            'correlation_matrix': self.get_correlation_matrix(),
            'outliers': self.get_outliers()
        }
        
        return summary
    
    def show_plots(self, save_plots: bool = False, output_dir: str = "./quickinsights_output"):
        """Görselleştirmeleri gösterir"""
        print("📈 Görselleştirmeler oluşturuluyor...")
        
        # Korelasyon matrisi
        if len(self._numeric_cols) > 1:
            correlation_matrix(self.df[self._numeric_cols], save_plot=save_plots, output_dir=output_dir)
        
        # Dağılım grafikleri
        if len(self._numeric_cols) > 0:
            distribution_plots(self.df[self._numeric_cols], save_plots=save_plots, output_dir=output_dir)
    
    def get_cache_status(self):
        """Cache durumunu gösterir"""
        status = {
            'data_info': self._data_info is not None,
            'numeric_analysis': self._numeric_analysis is not None,
            'categorical_analysis': self._categorical_analysis is not None,
            'correlation_matrix': self._correlation_matrix is not None,
            'outliers': self._outliers is not None
        }
        
        print("📊 Cache Durumu:")
        for key, cached in status.items():
            status_icon = "✅" if cached else "⏳"
            cache_text = "Cache'de" if cached else "Henüz hesaplanmadı"
            print(f"   {status_icon} {key}: {cache_text}")
        
        return status
    
    def clear_cache(self):
        """Cache'i temizler"""
        self._results = {}
        self._data_info = None
        self._numeric_analysis = None
        self._categorical_analysis = None
        self._correlation_matrix = None
        self._outliers = None
        print("🗑️  Cache temizlendi!")


def parallel_analysis(df: pd.DataFrame, 
                     n_jobs: int = -1,
                     backend: str = 'thread',
                     show_plots: bool = False,
                     save_plots: bool = False,
                     output_dir: str = "./quickinsights_output") -> dict:
    """
    Veri seti üzerinde paralel analiz yapar.
    
    Bu fonksiyon, farklı analiz türlerini paralel olarak çalıştırarak
    toplam analiz süresini kısaltır.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    n_jobs : int, default=-1
        Kullanılacak iş parçacığı sayısı (-1 = tüm CPU çekirdekleri)
    backend : str, default='thread'
        Paralel işleme backend'i ('thread' veya 'process')
    show_plots : bool, default=False
        Grafikleri göstermek isteyip istemediğiniz
    save_plots : bool, default=False
        Grafikleri kaydetmek isteyip istemediğiniz
    output_dir : str, default="./quickinsights_output"
        Grafiklerin kaydedileceği dizin
        
    Returns
    -------
    dict
        Paralel analiz sonuçları
        
    Examples
    --------
    >>> import quickinsights as qi
    >>> df = pd.read_csv('data.csv')
    >>> results = qi.parallel_analysis(df, n_jobs=4)
    """
    
    if df.empty:
        raise ValueError("Veri seti boş olamaz!")
    
    # CPU çekirdek sayısını belirle
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"🚀 Paralel Analiz Başlıyor...")
    print(f"   🔧 Backend: {backend}")
    print(f"   ⚡ İş parçacığı sayısı: {n_jobs}")
    print(f"   📊 Veri seti boyutu: {df.shape}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Analiz türlerini belirle - DataFrame kopyalama yapmadan
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Veri bilgileri (hızlı, paralel gerekmez)
    print("🔍 Veri seti bilgileri alınıyor...")
    data_info = get_data_info(df)
    
    # Paralel işleme
    results = {}
    
    if backend == 'thread':
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Futures'ları topla ve aynı anda başlat
            futures = {}
            
            # Sayısal analiz
            if len(numeric_cols) > 0:
                numeric_future = executor.submit(
                    _analyze_numeric_parallel, df[numeric_cols], show_plots, save_plots, output_dir
                )
                futures['numeric'] = numeric_future
            
            # Kategorik analiz
            if len(categorical_cols) > 0:
                categorical_future = executor.submit(
                    _analyze_categorical_parallel, df[categorical_cols], show_plots, save_plots, output_dir
                )
                futures['categorical'] = categorical_future
            
            # Aykırı değerler
            if len(numeric_cols) > 0:
                outliers_future = executor.submit(
                    _analyze_outliers_parallel, df[numeric_cols]
                )
                futures['outliers'] = outliers_future
            
            # Tüm sonuçları topla
            for key, future in futures.items():
                results[key] = future.result()
    
    elif backend == 'process':
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Futures'ları topla ve aynı anda başlat
            futures = {}
            
            # Sayısal analiz
            if len(numeric_cols) > 0:
                numeric_future = executor.submit(
                    _analyze_numeric_parallel, df[numeric_cols], show_plots, save_plots, output_dir
                )
                futures['numeric'] = numeric_future
            
            # Kategorik analiz
            if len(categorical_cols) > 0:
                categorical_future = executor.submit(
                    _analyze_categorical_parallel, df[categorical_cols], show_plots, save_plots, output_dir
                )
                futures['categorical'] = categorical_future
            
            # Aykırı değerler
            if len(numeric_cols) > 0:
                outliers_future = executor.submit(
                    _analyze_outliers_parallel, df[numeric_cols]
                )
                futures['outliers'] = outliers_future
            
            # Tüm sonuçları topla
            for key, future in futures.items():
                results[key] = future.result()
    
    else:
        raise ValueError("Geçersiz backend! 'thread' veya 'process' kullanın.")
    
    # Sonuçları birleştir
    final_results = {
        'data_info': data_info,
        'parallel_results': results,
        'execution_time': time.time() - start_time,
        'n_jobs': n_jobs,
        'backend': backend
    }
    
    print(f"\n✅ Paralel Analiz Tamamlandı!")
    print(f"   ⏱️  Toplam süre: {final_results['execution_time']:.3f} saniye")
    print(f"   🚀 Hızlanma: {len(results)} analiz paralel yapıldı")
    
    return final_results


def _analyze_numeric_parallel(df: pd.DataFrame, show_plots: bool, save_plots: bool, output_dir: str):
    """Paralel sayısal analiz için yardımcı fonksiyon"""
    return analyze_numeric(df, show_plots, save_plots, output_dir)


def _analyze_categorical_parallel(df: pd.DataFrame, show_plots: bool, save_plots: bool, output_dir: str):
    """Paralel kategorik analiz için yardımcı fonksiyon"""
    return analyze_categorical(df, show_plots, save_plots, output_dir)


def _analyze_outliers_parallel(df: pd.DataFrame):
    """Paralel aykırı değer analizi için yardımcı fonksiyon"""
    return detect_outliers(df)


def chunked_analysis(df: pd.DataFrame, 
                    chunk_size: int = 10000,
                    n_jobs: int = -1) -> dict:
    """
    Büyük veri setlerini parçalara bölerek analiz eder.
    
    Bu fonksiyon, bellek sınırlarını aşmamak için büyük veri setlerini
    küçük parçalara böler ve her parçayı ayrı ayrı analiz eder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    chunk_size : int, default=10000
        Her chunk'taki satır sayısı
    n_jobs : int, default=-1
        Kullanılacak iş parçacığı sayısı
        
    Returns
    -------
    dict
        Chunk'ların analiz sonuçları
    """
    
    if df.empty:
        raise ValueError("Veri seti boş olamaz!")
    
    if chunk_size <= 0:
        raise ValueError("Chunk boyutu pozitif olmalıdır!")
    
    print(f"🔪 Chunked Analiz Başlıyor...")
    print(f"   📊 Veri seti boyutu: {df.shape}")
    print(f"   🔪 Chunk boyutu: {chunk_size:,}")
    
    # Chunk sayısını hesapla
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    print(f"   📦 Toplam chunk sayısı: {total_chunks}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Chunk'ları daha verimli oluştur - numpy array slicing kullan
    chunk_indices = np.arange(0, len(df) + chunk_size, chunk_size)
    chunks = [df.iloc[start:min(start + chunk_size, len(df))] for start in chunk_indices[:-1]]
    
    print(f"✅ {len(chunks)} chunk oluşturuldu")
    
    # CPU çekirdek sayısını belirle
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), len(chunks))
    
    # Paralel chunk analizi
    results = []
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Her chunk için analiz görevi oluştur
        chunk_futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(_analyze_chunk, chunk, i+1, len(chunks))
            chunk_futures.append(future)
        
        # Sonuçları topla
        for future in chunk_futures:
            result = future.result()
            results.append(result)
    
    # Sonuçları birleştir
    merged_results = _merge_chunk_results(results)
    
    execution_time = time.time() - start_time
    
    print(f"\n✅ Chunked Analiz Tamamlandı!")
    print(f"   ⏱️  Toplam süre: {execution_time:.3f} saniye")
    print(f"   📊 Analiz edilen chunk sayısı: {len(chunks)}")
    print(f"   🚀 Paralel iş parçacığı sayısı: {n_jobs}")
    
    return {
        'chunk_results': results,
        'merged_results': merged_results,
        'execution_time': execution_time,
        'n_chunks': len(chunks),
        'chunk_size': chunk_size,
        'n_jobs': n_jobs
    }


def _analyze_chunk(chunk: pd.DataFrame, chunk_num: int, total_chunks: int) -> dict:
    """Tek bir chunk'ı analiz eder"""
    print(f"   🔍 Chunk {chunk_num}/{total_chunks} analiz ediliyor... ({chunk.shape})")
    
    # Chunk analizi
    chunk_info = get_data_info(chunk)
    
    # Sayısal analiz
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    numeric_analysis = {}
    if len(numeric_cols) > 0:
        numeric_analysis = analyze_numeric(chunk[numeric_cols], show_plots=False)
    
    # Kategorik analiz
    categorical_cols = chunk.select_dtypes(include=['object', 'category']).columns
    categorical_analysis = {}
    if len(categorical_cols) > 0:
        categorical_analysis = analyze_categorical(chunk[categorical_cols], show_plots=False)
    
    return {
        'chunk_num': chunk_num,
        'chunk_info': chunk_info,
        'numeric_analysis': numeric_analysis,
        'categorical_analysis': categorical_analysis,
        'shape': chunk.shape
    }


def _merge_chunk_results(chunk_results: List[dict]) -> dict:
    """Chunk sonuçlarını birleştirir"""
    if not chunk_results:
        return {}
    
    # Vectorized operations - tüm chunk'ları aynı anda işle
    chunk_info_list = [result['chunk_info'] for result in chunk_results]
    
    # Toplam istatistikler - numpy array kullanarak hızlandır
    total_rows = sum(info['rows'] for info in chunk_info_list)
    total_columns = chunk_info_list[0]['columns']
    total_memory = sum(info['memory_usage'] for info in chunk_info_list)
    
    # Sayısal analizleri birleştir - daha verimli
    merged_numeric = {}
    if chunk_results[0]['numeric_analysis']:
        numeric_cols = list(chunk_results[0]['numeric_analysis'].keys())
        
        # Her kolon için ağırlıklı ortalama hesapla
        for col in numeric_cols:
            # Vectorized calculation
            col_data = []
            weights = []
            
            for result in chunk_results:
                if col in result['numeric_analysis']:
                    col_data.append(result['numeric_analysis'][col]['mean'])
                    weights.append(result['chunk_info']['rows'])
            
            if weights:
                # NumPy ile ağırlıklı ortalama
                col_data = np.array(col_data)
                weights = np.array(weights)
                weighted_mean = np.average(col_data, weights=weights)
                
                merged_numeric[col] = {
                    'mean': float(weighted_mean),
                    'chunks_analyzed': len(weights)
                }
    
    # Kategorik analizleri birleştir - daha verimli
    merged_categorical = {}
    if chunk_results[0]['categorical_analysis']:
        categorical_cols = list(chunk_results[0]['categorical_analysis'].keys())
        
        for col in categorical_cols:
            # Tüm chunk'lardan değer sayılarını topla - Counter kullan
            from collections import Counter
            total_counts = Counter()
            chunks_analyzed = 0
            
            for result in chunk_results:
                if col in result['categorical_analysis']:
                    chunks_analyzed += 1
                    value_counts = result['categorical_analysis'][col]['value_counts']
                    total_counts.update(value_counts)
            
            if total_counts:
                merged_categorical[col] = {
                    'total_counts': dict(total_counts),
                    'unique_values': len(total_counts),
                    'chunks_analyzed': chunks_analyzed
                }
    
    return {
        'total_rows': total_rows,
        'total_columns': total_columns,
        'total_memory_mb': total_memory,
        'merged_numeric': merged_numeric,
        'merged_categorical': merged_categorical,
        'chunks_analyzed': len(chunk_results)
    }
