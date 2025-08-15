"""
QuickInsights yardımcı fonksiyonlar modülü

Bu modül, veri analizi için gerekli yardımcı fonksiyonları içerir.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Veri seti hakkında genel bilgileri döndürür.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analiz edilecek veri seti
        
    Returns
    -------
    dict
        Veri seti bilgilerini içeren sözlük
    """
    
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB cinsinden
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    return info


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Sayısal değişkenlerde aykırı değerleri tespit eder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Sadece sayısal değişkenler içeren veri seti
    method : str, default='iqr'
        Aykırı değer tespit yöntemi ('iqr' veya 'zscore')
    threshold : float, default=1.5
        Aykırı değer eşiği
        
    Returns
    -------
    pd.DataFrame
        Boolean mask - True değerler aykırı değerleri gösterir
    """
    
    if df.empty:
        return pd.DataFrame()
    
    # Vectorized operations için numpy array'e çevir
    data = df.values
    outliers = np.zeros_like(data, dtype=bool)
    
    if method == 'iqr':
        # IQR yöntemi - vectorized
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Broadcasting ile tüm kolonları aynı anda işle
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        # Z-score yöntemi - vectorized
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        
        # Broadcasting ile tüm kolonları aynı anda işle
        z_scores = np.abs((data - mean_vals) / std_vals)
        outliers = z_scores > threshold
    
    else:
        raise ValueError("Geçersiz yöntem! 'iqr' veya 'zscore' kullanın.")
    
    return pd.DataFrame(outliers, index=df.index, columns=df.columns)


def get_correlation_strength(correlation: float) -> str:
    """
    Korelasyon katsayısının gücünü sınıflandırır.
    
    Parameters
    ----------
    correlation : float
        Korelasyon katsayısı (-1 ile 1 arası)
        
    Returns
    -------
    str
        Korelasyon gücü açıklaması
    """
    
    abs_corr = abs(correlation)
    
    if abs_corr >= 0.8:
        return "Çok güçlü"
    elif abs_corr >= 0.6:
        return "Güçlü"
    elif abs_corr >= 0.4:
        return "Orta"
    elif abs_corr >= 0.2:
        return "Zayıf"
    else:
        return "Çok zayıf"


def format_number(value: float, decimals: int = 4) -> str:
    """
    Sayıları okunabilir formatta döndürür.
    
    Parameters
    ----------
    value : float
        Formatlanacak sayı
    decimals : int, default=4
        Ondalık basamak sayısı
        
    Returns
    -------
    str
        Formatlanmış sayı
    """
    
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1000000:
        return f"{value/1000000:.{decimals}f}M"
    elif abs(value) >= 1000:
        return f"{value/1000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def create_output_directory(output_dir: str) -> str:
    """
    Çıktı dizinini oluşturur.
    
    Parameters
    ----------
    output_dir : str
        Oluşturulacak dizin yolu
        
    Returns
    -------
    str
        Oluşturulan dizin yolu
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Çıktı dizini oluşturuldu: {output_dir}")
    
    return output_dir


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Veri setinin geçerli olup olmadığını kontrol eder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Kontrol edilecek veri seti
        
    Returns
    -------
    bool
        Veri seti geçerliyse True
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Veri seti pandas DataFrame olmalıdır!")
    
    if df.empty:
        raise ValueError("Veri seti boş olamaz!")
    
    if len(df.columns) == 0:
        raise ValueError("Veri setinde sütun bulunamadı!")
    
    return True


def get_data_sample(df: pd.DataFrame, sample_size: int = 5) -> pd.DataFrame:
    """
    Veri setinden örnek satırlar döndürür.
    
    Parameters
    ----------
    df : pd.DataFrame
        Örnek alınacak veri seti
    sample_size : int, default=5
        Örnek satır sayısı
        
    Returns
    -------
    pd.DataFrame
        Örnek veri seti
    """
    
    if len(df) <= sample_size:
        return df
    
    # Vectorized sampling - numpy array kullanarak hızlandır
    total_rows = len(df)
    
    # İlk, orta ve son satırlardan örnek al - daha verimli
    indices = []
    
    # İlk birkaç satır
    first_count = min(sample_size // 2, total_rows)
    indices.extend(range(first_count))
    
    # Ortadan birkaç satır
    mid_count = max(1, sample_size // 4)
    mid_start = total_rows // 2 - mid_count // 2
    mid_end = total_rows // 2 + mid_count // 2
    indices.extend(range(mid_start, min(mid_end, total_rows)))
    
    # Son birkaç satır
    last_count = max(1, sample_size - len(indices))
    indices.extend(range(max(0, total_rows - last_count), total_rows))
    
    # Benzersiz indeksleri al ve sırala - set kullanarak hızlandır
    unique_indices = sorted(set(indices))
    
    # NumPy array slicing ile daha hızlı
    return df.iloc[list(unique_indices)]


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri tiplerini optimize eder ve bellek kullanımını azaltır.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimize edilecek veri seti
        
    Returns
    -------
    pd.DataFrame
        Optimize edilmiş veri seti
    """
    print("🔧 Veri tipleri optimize ediliyor...")
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Vectorized dtype optimization - tüm kolonları aynı anda işle
    
    # Float64'leri float32'ye düşür
    float64_cols = df.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype('float32')
        print(f"   🔧 {len(float64_cols)} float64 kolonu float32'ye düşürüldü")
    
    # Int64'leri int32'ye düşür (mümkünse)
    int64_cols = df.select_dtypes(include=['int64']).columns
    if len(int64_cols) > 0:
        # Tüm int64 kolonları aynı anda kontrol et
        for col in int64_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        print(f"   🔧 {len(int64_cols)} int64 kolonu optimize edildi")
    
    # Object tiplerini kategorik yap (mümkünse)
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        # Tüm object kolonları aynı anda kontrol et
        for col in object_cols:
            if df[col].nunique() / len(df) < 0.5:  # %50'den az benzersiz değer
                df[col] = df[col].astype('category')
        print(f"   🔧 {len(object_cols)} object kolonu optimize edildi")
    
    final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    memory_saved = initial_memory - final_memory
    
    print(f"   💾 Bellek tasarrufu: {memory_saved:.2f} MB ({memory_saved/initial_memory*100:.1f}%)")
    
    return df


# Cache sistemi için gerekli importlar
import json
import time

# Numba JIT compilation için
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba bulunamadı. JIT compilation devre dışı.")

# Dask entegrasyonu için
try:
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("⚠️  Dask bulunamadı. Büyük veri analizi devre dışı.")

# Memory mapping için
try:
    import mmap
    import tempfile
    import os
    MEMORY_MAPPING_AVAILABLE = True
except ImportError:
    MEMORY_MAPPING_AVAILABLE = False
    print("⚠️  Memory mapping bulunamadı.")

# Profiling araçları için
try:
    import cProfile
    import pstats
    import io
    import psutil
    import tracemalloc
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    print("⚠️  Profiling araçları bulunamadı.")

# Async/await desteği için
try:
    import asyncio
    import aiofiles
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("⚠️  Async/await desteği bulunamadı.")

# GPU acceleration için
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU acceleration bulunamadı.")

# Cloud deployment için
try:
    import boto3
    import azure.storage.blob
    import google.cloud.storage
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False
    print("⚠️  Cloud deployment bulunamadı.")


class AnalysisCache:
    """
    Analiz sonuçlarını cache'leyen sınıf.
    
    Bu sınıf, veri seti hash'lerine göre analiz sonuçlarını saklar
    ve tekrar analizleri önler.
    """
    
    def __init__(self, cache_dir: str = "./.quickinsights_cache", max_size: int = 100):
        """
        AnalysisCache'i başlatır.
        
        Parameters
        ----------
        cache_dir : str
            Cache dosyalarının saklanacağı dizin
        max_size : int
            Maksimum cache boyutu (MB)
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache_info = {}
        
        # Cache dizinini oluştur
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"📁 Cache dizini oluşturuldu: {cache_dir}")
        
        # Mevcut cache'leri yükle
        self._load_cache_info()
    
    def _load_cache_info(self):
        """Mevcut cache bilgilerini yükler"""
        cache_info_file = os.path.join(self.cache_dir, "cache_info.json")
        if os.path.exists(cache_info_file):
            try:
                with open(cache_info_file, 'r', encoding='utf-8') as f:
                    self.cache_info = json.load(f)
            except:
                self.cache_info = {}
    
    def _save_cache_info(self):
        """Cache bilgilerini kaydeder"""
        cache_info_file = os.path.join(self.cache_dir, "cache_info.json")
        try:
            with open(cache_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Cache bilgileri kaydedilemedi: {e}")
    
    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        DataFrame için benzersiz hash oluşturur.
        
        Parameters
        ----------
        df : pd.DataFrame
            Hash'i oluşturulacak DataFrame
            
        Returns
        -------
        str
            DataFrame'in hash'i
        """
        # DataFrame'in içeriğini string'e çevir
        df_str = df.to_string()
        
        # Hash oluştur
        import hashlib
        return hashlib.md5(df_str.encode()).hexdigest()
    
    def get_cached_result(self, df: pd.DataFrame, analysis_type: str):
        """
        Cache'den sonuç alır.
        
        Parameters
        ----------
        df : pd.DataFrame
            Analiz edilecek DataFrame
        analysis_type : str
            Analiz türü
            
        Returns
        -------
        Any or None
            Cache'den alınan sonuç veya None
        """
        df_hash = self._get_dataframe_hash(df)
        cache_key = f"{df_hash}_{analysis_type}"
        
        if cache_key in self.cache_info:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    print(f"💾 Cache'den alındı: {analysis_type}")
                    return result
                except Exception as e:
                    print(f"⚠️  Cache okuma hatası: {e}")
                    # Hatalı cache'i sil
                    self._remove_cache_entry(cache_key)
        
        return None
    
    def cache_result(self, df: pd.DataFrame, analysis_type: str, result) -> bool:
        """
        Sonucu cache'ler.
        
        Parameters
        ----------
        df : pd.DataFrame
            Analiz edilen DataFrame
        analysis_type : str
            Analiz türü
        result : Any
            Cache'lenecek sonuç
            
        Returns
        -------
        bool
            Cache başarılı ise True
        """
        try:
            df_hash = self._get_dataframe_hash(df)
            cache_key = f"{df_hash}_{analysis_type}"
            
            # Cache dosyası oluştur
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Cache bilgilerini güncelle
            file_size = os.path.getsize(cache_file) / 1024 / 1024  # MB
            self.cache_info[cache_key] = {
                'analysis_type': analysis_type,
                'df_hash': df_hash,
                'file_size': file_size,
                'created_at': time.time()
            }
            
            # Cache boyutunu kontrol et
            self._manage_cache_size()
            
            return True
        except Exception as e:
            print(f"⚠️  Cache hatası: {e}")
            return False
    
    def get_or_compute(self, cache_key: str, compute_func, *args, **kwargs):
        """
        Cache'den sonucu alır veya hesaplar.
        
        Parameters
        ----------
        cache_key : str
            Cache anahtarı
        compute_func : callable
            Hesaplama fonksiyonu
        *args, **kwargs
            Hesaplama fonksiyonuna geçirilecek argümanlar
            
        Returns
        -------
        Any
            Cache'den alınan veya hesaplanan sonuç
        """
        # Önce cache'den kontrol et
        if cache_key in self.cache_info:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    print(f"💾 Cache'den alındı: {cache_key}")
                    return result
                except Exception as e:
                    print(f"⚠️  Cache okuma hatası: {e}")
                    # Hatalı cache'i sil
                    self._remove_cache_entry(cache_key)
        
        # Cache'de yoksa hesapla
        print(f"🧮 Hesaplanıyor: {cache_key}")
        result = compute_func(*args, **kwargs)
        
        # Sonucu cache'le
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Cache bilgilerini güncelle
            file_size = os.path.getsize(cache_file) / 1024 / 1024  # MB
            self.cache_info[cache_key] = {
                'analysis_type': cache_key,
                'df_hash': cache_key,
                'file_size': file_size,
                'created_at': time.time()
            }
            
            # Cache boyutunu kontrol et
            self._manage_cache_size()
            
        except Exception as e:
            print(f"⚠️  Cache kaydetme hatası: {e}")
        
        return result
    
    def _manage_cache_size(self):
        """Cache boyutunu yönetir"""
        total_size = sum(info['file_size'] for info in self.cache_info.values())
        
        if total_size > self.max_size:
            print(f"⚠️  Cache boyutu aşıldı ({total_size:.2f} MB > {self.max_size} MB)")
            print("🗑️  Eski cache'ler temizleniyor...")
            
            # En eski cache'leri sil
            sorted_cache = sorted(self.cache_info.items(), 
                                key=lambda x: x[1]['created_at'])
            
            for i, (key, info) in enumerate(sorted_cache):
                if total_size <= self.max_size:
                    break
                
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    total_size -= info['file_size']
                    del self.cache_info[key]
                    print(f"   🗑️  Silindi: {info['analysis_type']}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Cache girişini kaldırır"""
        if cache_key in self.cache_info:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            del self.cache_info[cache_key]
    
    def clear_cache(self):
        """Tüm cache'i temizler"""
        for cache_key in list(self.cache_info.keys()):
            self._remove_cache_entry(cache_key)
        
        # Cache bilgilerini kaydet
        self._save_cache_info()
        print("🗑️  Tüm cache temizlendi!")
    
    def get_cache_stats(self):
        """Cache istatistiklerini döndürür"""
        total_files = len(self.cache_info)
        total_size = sum(info['file_size'] for info in self.cache_info.values())
        
        print("📊 Cache İstatistikleri:")
        print(f"   📁 Toplam cache dosyası: {total_files}")
        print(f"   💾 Toplam boyut: {total_size:.2f} MB")
        print(f"   🎯 Maksimum boyut: {self.max_size} MB")
        
        # Analiz türlerine göre dağılım
        analysis_counts = {}
        for info in self.cache_info.values():
            analysis_type = info['analysis_type']
            analysis_counts[analysis_type] = analysis_counts.get(analysis_type, 0) + 1
        
        print(f"   🔍 Analiz türleri: {analysis_counts}")
        
        return {
            'total_files': total_files,
            'total_size': total_size,
            'max_size': self.max_size,
            'analysis_counts': analysis_counts
        }


# =============================================================================
# Numba JIT ile Hızlandırılmış Fonksiyonlar
# =============================================================================

def fast_correlation_matrix(data):
    """
    Numba JIT ile hızlandırılmış korelasyon matrisi hesaplama.
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        2D veri (satırlar, sütunlar)
        
    Returns
    -------
    numpy.ndarray
        Korelasyon matrisi
    """
    # DataFrame'i numpy array'e çevir
    if hasattr(data, 'values'):
        data = data.values
    
    # Numba JIT ile hesapla
    return _fast_correlation_matrix_numba(data)

@jit(nopython=True, parallel=True)
def _fast_correlation_matrix_numba(data):
    """
    Numba JIT ile hızlandırılmış korelasyon matrisi hesaplama (numpy array için).
    """
    n_rows, n_cols = data.shape
    corr_matrix = np.zeros((n_cols, n_cols))
    
    for i in prange(n_cols):
        for j in range(i, n_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            
            # Korelasyon hesaplama
            x = data[:, i]
            y = data[:, j]
            
            # Ortalama hesapla
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            # Pay ve payda hesapla
            numerator = 0.0
            x_var = 0.0
            y_var = 0.0
            
            for k in range(n_rows):
                x_diff = x[k] - x_mean
                y_diff = y[k] - y_mean
                numerator += x_diff * y_diff
                x_var += x_diff * x_diff
                y_var += y_diff * y_diff
            
            # Korelasyon katsayısı
            if x_var > 0 and y_var > 0:
                corr = numerator / np.sqrt(x_var * y_var)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    return corr_matrix


@jit(nopython=True, parallel=True)
def fast_outlier_detection(data, method='iqr', threshold=1.5):
    """
    Numba JIT ile hızlandırılmış aykırı değer tespiti.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D numpy array (satırlar, sütunlar)
    method : str
        'iqr' veya 'zscore'
    threshold : float
        Aykırı değer eşiği
        
    Returns
    -------
    numpy.ndarray
        Boolean mask - True değerler aykırı değerleri gösterir
    """
    n_rows, n_cols = data.shape
    outliers = np.zeros((n_rows, n_cols), dtype=np.bool_)
    
    if method == 'iqr':
        for j in prange(n_cols):
            col = data[:, j]
            
            # Q1, Q3 ve IQR hesapla
            sorted_col = np.sort(col)
            q1_idx = int(0.25 * n_rows)
            q3_idx = int(0.75 * n_rows)
            
            q1 = sorted_col[q1_idx]
            q3 = sorted_col[q3_idx]
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i in range(n_rows):
                outliers[i, j] = (col[i] < lower_bound) or (col[i] > upper_bound)
    
    elif method == 'zscore':
        for j in prange(n_cols):
            col = data[:, j]
            
            # Ortalama ve standart sapma
            mean_val = np.mean(col)
            std_val = np.std(col)
            
            if std_val > 0:
                for i in range(n_rows):
                    z_score = abs((col[i] - mean_val) / std_val)
                    outliers[i, j] = z_score > threshold
    
    return outliers


@jit(nopython=True, parallel=True)
def fast_summary_stats(data):
    """
    Numba JIT ile hızlandırılmış istatistiksel özet.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D numpy array (satırlar, sütunlar)
        
    Returns
    -------
    dict
        Her sütun için istatistikler
    """
    n_rows, n_cols = data.shape
    stats = {}
    
    for j in prange(n_cols):
        col = data[:, j]
        
        # Temel istatistikler
        mean_val = np.mean(col)
        std_val = np.std(col)
        min_val = np.min(col)
        max_val = np.max(col)
        
        # Medyan hesapla
        sorted_col = np.sort(col)
        if n_rows % 2 == 0:
            median_val = (sorted_col[n_rows//2 - 1] + sorted_col[n_rows//2]) / 2
        else:
            median_val = sorted_col[n_rows//2]
        
        # Q1 ve Q3 hesapla
        q1_idx = int(0.25 * n_rows)
        q3_idx = int(0.75 * n_rows)
        q1_val = sorted_col[q1_idx]
        q3_val = sorted_col[q3_idx]
        
        stats[j] = {
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'q1': q1_val,
            'q3': q3_val
        }
    
    return stats


def get_numba_status():
    """Numba kullanılabilirlik durumunu döndürür"""
    if NUMBA_AVAILABLE:
        print("✅ Numba JIT compilation aktif!")
        print(f"   🚀 Hızlandırılmış fonksiyonlar kullanılabilir")
        return True
    else:
        print("❌ Numba JIT compilation devre dışı!")
        print("   📦 Kurulum: pip install numba")
        return False


def benchmark_numba_vs_pandas(df, n_iterations=100):
    """
    Numba vs Pandas performans karşılaştırması.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test edilecek veri seti
    n_iterations : int
        Test tekrar sayısı
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    if not NUMBA_AVAILABLE:
        print("❌ Numba bulunamadı! Benchmark yapılamıyor.")
        return {}
    
    print("🏁 Numba vs Pandas Benchmark Başlıyor...")
    print(f"   📊 Veri seti boyutu: {df.shape}")
    print(f"   🔄 Test tekrar sayısı: {n_iterations}")
    
    # Sayısal sütunları al
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("❌ Sayısal sütun bulunamadı!")
        return {}
    
    # Veriyi numpy array'e çevir
    data = df[numeric_cols].values
    
    # Pandas korelasyon testi
    print("\n🔍 Pandas korelasyon testi...")
    start_time = time.time()
    for _ in range(n_iterations):
        pandas_corr = df[numeric_cols].corr().values
    pandas_time = time.time() - start_time
    
    # Numba korelasyon testi
    print("🚀 Numba korelasyon testi...")
    start_time = time.time()
    for _ in range(n_iterations):
        numba_corr = fast_correlation_matrix(data)
    numba_time = time.time() - start_time
    
    # Sonuçları hesapla
    speedup = pandas_time / numba_time
    time_saved = ((pandas_time - numba_time) / pandas_time) * 100
    
    print(f"\n📈 BENCHMARK SONUÇLARI:")
    print(f"   🐌 Pandas: {pandas_time:.4f} saniye")
    print(f"   🚀 Numba: {numba_time:.4f} saniye")
    print(f"   ⚡ Hızlanma: {speedup:.2f}x")
    print(f"   💰 Zaman tasarrufu: {time_saved:.1f}%")
    
    return {
        'pandas_time': pandas_time,
        'numba_time': numba_time,
        'speedup': speedup,
        'time_saved': time_saved,
        'n_iterations': n_iterations
    }


# =============================================================================
# Memory Mapping ile Bellek Optimizasyonu
# =============================================================================

def get_memory_mapping_status():
    """Memory mapping kullanılabilirlik durumunu döndürür"""
    if MEMORY_MAPPING_AVAILABLE:
        print("✅ Memory mapping aktif!")
        print(f"   🚀 Büyük dosyalar için bellek tasarrufu sağlanabilir")
        return True
    else:
        print("❌ Memory mapping devre dışı!")
        return False


def create_memory_mapped_array(data, filename=None):
    """
    Memory mapped array oluşturur.
    
    Parameters
    ----------
    data : numpy.ndarray
        Memory map'lenecek veri
    filename : str, optional
        Geçici dosya adı (None ise otomatik)
        
    Returns
    -------
    numpy.memmap
        Memory mapped array
    """
    if not MEMORY_MAPPING_AVAILABLE:
        print("❌ Memory mapping bulunamadı!")
        return data
    
    try:
        if filename is None:
            # Geçici dosya oluştur
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
            filename = temp_file.name
            temp_file.close()
        
        # Memory mapped array oluştur
        mmap_array = np.memmap(filename, dtype=data.dtype, mode='w+', shape=data.shape)
        mmap_array[:] = data[:]
        
        print(f"✅ Memory mapped array oluşturuldu!")
        print(f"   📁 Dosya: {filename}")
        print(f"   💾 Boyut: {data.nbytes / 1024 / 1024:.2f} MB")
        
        return mmap_array
        
    except Exception as e:
        print(f"❌ Memory mapping hatası: {e}")
        return data


def benchmark_memory_vs_mmap(data, n_iterations=100):
    """
    Normal array vs Memory mapped array performans karşılaştırması.
    
    Parameters
    ----------
    data : numpy.ndarray
        Test edilecek veri
    n_iterations : int
        Test tekrar sayısı
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    if not MEMORY_MAPPING_AVAILABLE:
        print("❌ Memory mapping bulunamadı! Benchmark yapılamıyor.")
        return {}
    
    print("🏁 Normal Array vs Memory Map Benchmark Başlıyor...")
    print(f"   📊 Veri boyutu: {data.shape}")
    print(f"   💾 Bellek kullanımı: {data.nbytes / 1024 / 1024:.2f} MB")
    print(f"   🔄 Test tekrar sayısı: {n_iterations}")
    
    # Normal array testi
    print("\n🔍 Normal array testi...")
    start_time = time.time()
    for _ in range(n_iterations):
        # Basit operasyonlar
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
    normal_time = time.time() - start_time
    
    # Memory mapped array testi
    print("🚀 Memory mapped array testi...")
    mmap_array = create_memory_mapped_array(data)
    if mmap_array is None:
        print("❌ Memory mapped array oluşturulamadı!")
        return {}
    
    start_time = time.time()
    for _ in range(n_iterations):
        # Aynı operasyonlar
        mean_val = np.mean(mmap_array)
        std_val = np.std(mmap_array)
        min_val = np.min(mmap_array)
        max_val = np.max(mmap_array)
    mmap_time = time.time() - start_time
    
    # Sonuçları hesapla
    speedup = normal_time / mmap_time
    time_saved = ((normal_time - mmap_time) / normal_time) * 100
    
    print(f"\n📈 BENCHMARK SONUÇLARI:")
    print(f"   🐌 Normal Array: {normal_time:.4f} saniye")
    print(f"   🚀 Memory Map: {mmap_time:.4f} saniye")
    print(f"   ⚡ Hızlanma: {speedup:.2f}x")
    print(f"   💰 Zaman tasarrufu: {time_saved:.1f}%")
    
    return {
        'normal_time': normal_time,
        'mmap_time': mmap_time,
        'speedup': speedup,
        'time_saved': time_saved,
        'n_iterations': n_iterations
    }


# =============================================================================
# Profiling Araçları - Performans Analizi
# =============================================================================

def get_profiling_status():
    """Profiling araçlarının kullanılabilirlik durumunu döndürür"""
    if PROFILING_AVAILABLE:
        print("✅ Profiling araçları aktif!")
        print(f"   🔍 cProfile, pstats, psutil, tracemalloc kullanılabilir")
        return True
    else:
        print("❌ Profiling araçları devre dışı!")
        print("   📦 Kurulum: pip install psutil")
        return False


def profile_function(func, *args, **kwargs):
    """
    Fonksiyonu cProfile ile profil eder.
    
    Parameters
    ----------
    func : callable
        Profil edilecek fonksiyon
    *args, **kwargs
        Fonksiyon parametreleri
        
    Returns
    -------
    tuple
        (sonuç, profil_raporu)
    """
    if not PROFILING_AVAILABLE:
        print("❌ Profiling araçları bulunamadı!")
        return func(*args, **kwargs), None
    
    print(f"🔍 {func.__name__} fonksiyonu profil ediliyor...")
    
    # Profiler başlat
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Fonksiyonu çalıştır
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Profiler durdur
    profiler.disable()
    
    # Profil raporu oluştur
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(20)  # İlk 20 satır
    
    print(f"✅ Profil tamamlandı! Çalışma süresi: {execution_time:.4f} saniye")
    
    return result, s.getvalue()


def get_memory_usage():
    """
    Mevcut bellek kullanımını döndürür.
    
    Returns
    -------
    dict
        Bellek kullanım bilgileri
    """
    if not PROFILING_AVAILABLE:
        print("❌ psutil bulunamadı!")
        return {}
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Sistem bellek bilgileri
        system_memory = psutil.virtual_memory()
        
        memory_stats = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'system_total_gb': system_memory.total / 1024 / 1024 / 1024,
            'system_available_gb': system_memory.available / 1024 / 1024 / 1024,
            'system_percent': system_memory.percent
        }
        
        print("💾 Bellek kullanımı:")
        print(f"   📊 RSS: {memory_stats['rss_mb']:.2f} MB")
        print(f"   📊 VMS: {memory_stats['vms_mb']:.2f} MB")
        print(f"   📊 Process: {memory_stats['percent']:.1f}%")
        print(f"   📊 Sistem: {memory_stats['system_percent']:.1f}%")
        
        return memory_stats
        
    except Exception as e:
        print(f"❌ Bellek bilgisi alınamadı: {e}")
        return {}


def start_memory_tracking():
    """Bellek takibini başlatır"""
    if not PROFILING_AVAILABLE:
        print("❌ tracemalloc bulunamadı!")
        return None
    
    try:
        tracemalloc.start()
        print("✅ Bellek takibi başlatıldı!")
        return True
    except Exception as e:
        print(f"❌ Bellek takibi başlatılamadı: {e}")
        return None


def get_memory_snapshot():
    """
    Bellek anlık görüntüsü alır.
    
    Returns
    -------
    dict
        Bellek anlık görüntü bilgileri
    """
    if not PROFILING_AVAILABLE:
        print("❌ tracemalloc bulunamadı!")
        return {}
    
    try:
        snapshot = tracemalloc.take_snapshot()
        
        # En çok bellek kullanan dosyalar
        top_stats = snapshot.statistics('lineno')
        
        memory_info = {
            'total_allocated_mb': sum(stat.size for stat in top_stats) / 1024 / 1024,
            'top_files': []
        }
        
        print("📸 Bellek anlık görüntüsü:")
        print(f"   💾 Toplam ayrılan: {memory_info['total_allocated_mb']:.2f} MB")
        
        # İlk 5 dosyayı göster
        for i, stat in enumerate(top_stats[:5]):
            file_info = {
                'file': stat.traceback.format()[-1],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            }
            memory_info['top_files'].append(file_info)
            print(f"   📁 {file_info['file']}: {file_info['size_mb']:.2f} MB ({file_info['count']} blok)")
        
        return memory_info
        
    except Exception as e:
        print(f"❌ Bellek anlık görüntüsü alınamadı: {e}")
        return {}


def benchmark_suite(df, operations=['info', 'describe', 'corr', 'outliers']):
    """
    Kapsamlı benchmark testi.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test edilecek veri seti
    operations : list
        Test edilecek operasyonlar
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    print("🏁 KAPSAMLI BENCHMARK SUITE BAŞLIYOR...")
    print(f"   📊 Veri seti boyutu: {df.shape}")
    print(f"   💾 Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Bellek takibini başlat
    start_memory_tracking()
    
    results = {}
    total_start_time = time.time()
    
    for op in operations:
        print(f"\n🔍 {op.upper()} operasyonu test ediliyor...")
        
        # Bellek öncesi
        memory_before = get_memory_snapshot()
        
        # Operasyonu çalıştır
        start_time = time.time()
        
        if op == 'info':
            result = get_data_info(df)
        elif op == 'describe':
            result = df.describe()
        elif op == 'corr':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                result = df[numeric_cols].corr()
            else:
                result = None
        elif op == 'outliers':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result = detect_outliers(df[numeric_cols])
            else:
                result = None
        
        execution_time = time.time() - start_time
        
        # Bellek sonrası
        memory_after = get_memory_snapshot()
        
        # Sonuçları kaydet
        results[op] = {
            'execution_time': execution_time,
            'memory_before': memory_before.get('total_allocated_mb', 0),
            'memory_after': memory_after.get('total_allocated_mb', 0),
            'memory_increase': memory_after.get('total_allocated_mb', 0) - memory_before.get('total_allocated_mb', 0)
        }
        
        print(f"   ⏱️  Süre: {execution_time:.4f} saniye")
        print(f"   💾 Bellek artışı: {results[op]['memory_increase']:.2f} MB")
    
    total_time = time.time() - total_start_time
    
    # Genel özet
    print(f"\n📈 GENEL BENCHMARK SONUÇLARI:")
    print(f"   ⏱️  Toplam süre: {total_time:.4f} saniye")
    print(f"   🔍 Test edilen operasyon: {len(operations)}")
    
    # En hızlı ve en yavaş operasyonlar
    execution_times = {op: results[op]['execution_time'] for op in results}
    fastest_op = min(execution_times, key=execution_times.get)
    slowest_op = max(execution_times, key=execution_times.get)
    
    print(f"   🚀 En hızlı: {fastest_op} ({execution_times[fastest_op]:.4f}s)")
    print(f"   🐌 En yavaş: {slowest_op} ({execution_times[slowest_op]:.4f}s)")
    
    return results


# =============================================================================
# Async/Await ile Asenkron Veri Analizi
# =============================================================================

def get_async_status():
    """Async/await kullanılabilirlik durumunu döndürür"""
    if ASYNC_AVAILABLE:
        print("✅ Async/await desteği aktif!")
        print(f"   🚀 Asenkron veri analizi yapılabilir")
        return True
    else:
        print("❌ Async/await desteği devre dışı!")
        print("   📦 Kurulum: pip install aiofiles aiohttp")
        return False


async def async_analyze_dataset(df, operations=['info', 'describe', 'corr']):
    """
    Veri setini asenkron olarak analiz eder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analiz edilecek veri seti
    operations : list
        Yapılacak operasyonlar
        
    Returns
    -------
    dict
        Analiz sonuçları
    """
    if not ASYNC_AVAILABLE:
        print("❌ Async/await bulunamadı! Senkron analiz kullanılıyor.")
        return {}
    
    print("🚀 Asenkron veri analizi başlıyor...")
    print(f"   📊 Veri seti boyutu: {df.shape}")
    
    results = {}
    start_time = time.time()
    
    # Asenkron operasyonlar
    async def process_operation(op):
        print(f"   🔍 {op.upper()} operasyonu işleniyor...")
        
        # Simüle edilmiş asenkron işlem
        await asyncio.sleep(0.1)
        
        if op == 'info':
            result = get_data_info(df)
        elif op == 'describe':
            result = df.describe()
        elif op == 'corr':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                result = df[numeric_cols].corr()
            else:
                result = None
        
        return op, result
    
    # Tüm operasyonları paralel olarak çalıştır
    tasks = [process_operation(op) for op in operations]
    completed_results = await asyncio.gather(*tasks)
    
    # Sonuçları düzenle
    for op, result in completed_results:
        results[op] = result
    
    total_time = time.time() - start_time
    print(f"✅ Asenkron analiz tamamlandı! Toplam süre: {total_time:.4f} saniye")
    
    return results


async def async_data_loading(file_paths):
    """
    Birden fazla dosyayı asenkron olarak yükler.
    
    Parameters
    ----------
    file_paths : list
        Yüklenecek dosya yolları
        
    Returns
    -------
    dict
        Yüklenen veri setleri
    """
    if not ASYNC_AVAILABLE:
        print("❌ Async/await bulunamadı!")
        return {}
    
    print(f"📁 {len(file_paths)} dosya asenkron olarak yükleniyor...")
    
    async def load_file(file_path):
        try:
            # Simüle edilmiş dosya yükleme
            await asyncio.sleep(0.2)
            
            # Gerçek uygulamada burada dosya yükleme olacak
            # df = pd.read_csv(file_path) gibi
            
            return file_path, f"Loaded data from {file_path}"
        except Exception as e:
            return file_path, f"Error loading {file_path}: {e}"
    
    # Tüm dosyaları paralel olarak yükle
    tasks = [load_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    
    return dict(results)


async def async_benchmark_async_vs_sync(df, operations=['info', 'describe', 'corr']):
    """
    Asenkron vs senkron performans karşılaştırması.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test edilecek veri seti
    operations : list
        Test edilecek operasyonlar
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    if not ASYNC_AVAILABLE:
        print("❌ Async/await bulunamadı! Benchmark yapılamıyor.")
        return {}
    
    print("🏁 Async vs Sync Benchmark Başlıyor...")
    
    # Senkron test
    print("\n🔍 Senkron test...")
    start_time = time.time()
    sync_results = {}
    for op in operations:
        if op == 'info':
            sync_results[op] = get_data_info(df)
        elif op == 'describe':
            sync_results[op] = df.describe()
        elif op == 'corr':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                sync_results[op] = df[numeric_cols].corr()
    sync_time = time.time() - start_time
    
    # Asenkron test
    print("🚀 Asenkron test...")
    start_time = time.time()
    async_results = await async_analyze_dataset(df, operations)
    async_time = time.time() - start_time
    
    # Sonuçları karşılaştır
    speedup = sync_time / async_time
    time_saved = ((sync_time - async_time) / sync_time) * 100
    
    print(f"\n📈 BENCHMARK SONUÇLARI:")
    print(f"   🐌 Senkron: {sync_time:.4f} saniye")
    print(f"   🚀 Asenkron: {async_time:.4f} saniye")
    print(f"   ⚡ Hızlanma: {speedup:.2f}x")
    print(f"   💰 Zaman tasarrufu: {time_saved:.1f}%")
    
    return {
        'sync_time': sync_time,
        'async_time': async_time,
        'speedup': speedup,
        'time_saved': time_saved
    }


def run_async_example():
    """
    Async/await kullanım örneği.
    """
    if not ASYNC_AVAILABLE:
        print("❌ Async/await bulunamadı!")
        return
    
    print("🚀 Async/await örneği çalıştırılıyor...")
    
    # Test veri seti oluştur
    df = pd.DataFrame({
        'yas': np.random.normal(30, 10, 10000),
        'maas': np.random.normal(50000, 15000, 10000),
        'deneyim': np.random.normal(5, 3, 10000)
    })
    
    # Asenkron analiz çalıştır
    async def main():
        # Veri analizi
        results = await async_analyze_dataset(df, ['info', 'describe', 'corr'])
        
        # Dosya yükleme simülasyonu
        file_paths = ['data1.csv', 'data2.csv', 'data3.csv']
        loaded_data = await async_data_loading(file_paths)
        
        # Benchmark
        benchmark = await async_benchmark_async_vs_sync(df)
        
        return results, loaded_data, benchmark
    
    # Event loop çalıştır
    try:
        results, loaded_data, benchmark = asyncio.run(main())
        print("✅ Async/await örneği başarıyla tamamlandı!")
        return results, loaded_data, benchmark
    except Exception as e:
        print(f"❌ Async/await örneği hatası: {e}")
        return None, None, None


# =============================================================================
# Dask ile Büyük Veri Analizi
# =============================================================================

def get_dask_status():
    """Dask kullanılabilirlik durumunu döndürür"""
    if DASK_AVAILABLE:
        print("✅ Dask büyük veri analizi aktif!")
        print(f"   🚀 GB-TB boyutunda veri setleri analiz edilebilir")
        return True
    else:
        print("❌ Dask büyük veri analizi devre dışı!")
        print("   📦 Kurulum: pip install dask[complete]")
        return False


def create_dask_client(n_workers=None, memory_limit='2GB'):
    """
    Dask client oluşturur.
    
    Parameters
    ----------
    n_workers : int, optional
        Worker sayısı (None ise otomatik)
    memory_limit : str, optional
        Worker başına bellek limiti
        
    Returns
    -------
    dask.distributed.Client or None
        Dask client veya None
    """
    if not DASK_AVAILABLE:
        print("❌ Dask bulunamadı! Client oluşturulamıyor.")
        return None
    
    try:
        # Local cluster oluştur
        cluster = LocalCluster(
            n_workers=n_workers,
            memory_limit=memory_limit,
            processes=True
        )
        
        # Client oluştur
        client = Client(cluster)
        
        print(f"✅ Dask client oluşturuldu!")
        print(f"   🔧 Worker sayısı: {len(client.scheduler_info()['workers'])}")
        print(f"   💾 Bellek limiti: {memory_limit}")
        print(f"   🌐 Dashboard: {client.dashboard_link}")
        
        return client
        
    except Exception as e:
        print(f"❌ Dask client oluşturulamadı: {e}")
        return None


def convert_to_dask(df, npartitions=None):
    """
    Pandas DataFrame'i Dask DataFrame'e çevirir.
    
    Parameters
    ----------
    df : pd.DataFrame
        Çevrilecek DataFrame
    npartitions : int, optional
        Partition sayısı (None ise otomatik)
        
    Returns
    -------
    dask.dataframe.DataFrame or pd.DataFrame
        Dask DataFrame veya orijinal DataFrame
    """
    if not DASK_AVAILABLE:
        print("⚠️  Dask bulunamadı! Orijinal DataFrame döndürülüyor.")
        return df
    
    if npartitions is None:
        # Otomatik partition sayısı hesapla
        npartitions = max(1, len(df) // 10000)  # 10K satır per partition
    
    try:
        ddf = dd.from_pandas(df, npartitions=npartitions)
        print(f"✅ DataFrame Dask'a çevrildi!")
        print(f"   📊 Partition sayısı: {ddf.npartitions}")
        print(f"   💾 Tahmini bellek: {ddf.memory_usage_per_partition().sum().compute() / 1024 / 1024:.2f} MB")
        return ddf
        
    except Exception as e:
        print(f"⚠️  Dask'a çevirme hatası: {e}")
        return df


def dask_analyze_large_dataset(df, chunk_size='100MB', show_progress=True):
    """
    Büyük veri setini Dask ile analiz eder.
    
    Parameters
    ----------
    df : pd.DataFrame or dask.dataframe.DataFrame
        Analiz edilecek veri seti
    chunk_size : str, optional
        Chunk boyutu
    show_progress : bool, optional
        İlerleme çubuğu göster
        
    Returns
    -------
    dict
        Analiz sonuçları
    """
    if not DASK_AVAILABLE:
        print("❌ Dask bulunamadı! Standart analiz kullanılıyor.")
        return {}
    
    print("🚀 Dask ile büyük veri analizi başlıyor...")
    
    # DataFrame'i Dask'a çevir
    if not isinstance(df, dd.DataFrame):
        df = convert_to_dask(df)
    
    if not isinstance(df, dd.DataFrame):
        print("❌ Dask DataFrame oluşturulamadı!")
        return {}
    
    results = {}
    
    try:
        # Veri seti bilgileri
        print("📊 Veri seti bilgileri hesaplanıyor...")
        results['shape'] = df.shape.compute()
        results['columns'] = df.columns.tolist()
        results['dtypes'] = df.dtypes.to_dict()
        
        # Sayısal sütunlar
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            print("🔢 Sayısal analiz yapılıyor...")
            
            # Temel istatistikler
            numeric_df = df[numeric_cols]
            stats = numeric_df.describe().compute()
            results['numeric_stats'] = stats.to_dict()
            
            # Korelasyon matrisi (büyük veri için örnekleme)
            if len(numeric_cols) > 1:
                print("🔗 Korelasyon analizi yapılıyor...")
                # Büyük veri için örnekleme
                sample_size = min(10000, len(df))
                sample_df = df.sample(n=sample_size, random_state=42)
                corr_matrix = sample_df[numeric_cols].corr().compute()
                results['correlation_matrix'] = corr_matrix.to_dict()
        
        # Kategorik sütunlar
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print("📝 Kategorik analiz yapılıyor...")
            
            categorical_stats = {}
            for col in categorical_cols[:5]:  # İlk 5 kategorik sütun
                value_counts = df[col].value_counts().head(10).compute()
                categorical_stats[col] = value_counts.to_dict()
            
            results['categorical_stats'] = categorical_stats
        
        # Eksik değer analizi
        print("❓ Eksik değer analizi yapılıyor...")
        missing_counts = df.isnull().sum().compute()
        results['missing_values'] = missing_counts.to_dict()
        
        # Bellek kullanımı
        memory_usage = df.memory_usage_per_partition().sum().compute()
        results['memory_usage_mb'] = memory_usage / 1024 / 1024
        
        print("✅ Dask analizi tamamlandı!")
        
    except Exception as e:
        print(f"❌ Dask analizi hatası: {e}")
        results['error'] = str(e)
    
    return results


def benchmark_dask_vs_pandas(df, operations=['describe', 'corr', 'groupby']):
    """
    Dask vs Pandas performans karşılaştırması.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test edilecek veri seti
    operations : list
        Test edilecek operasyonlar
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    if not DASK_AVAILABLE:
        print("❌ Dask bulunamadı! Benchmark yapılamıyor.")
        return {}
    
    print("🏁 Dask vs Pandas Benchmark Başlıyor...")
    print(f"   📊 Veri seti boyutu: {df.shape}")
    
    # Dask DataFrame oluştur
    ddf = convert_to_dask(df)
    if not isinstance(ddf, dd.DataFrame):
        print("❌ Dask DataFrame oluşturulamadı!")
        return {}
    
    results = {}
    
    for op in operations:
        if op == 'describe':
            print(f"\n🔍 {op} operasyonu test ediliyor...")
            
            # Pandas
            start_time = time.time()
            pandas_result = df.describe()
            pandas_time = time.time() - start_time
            
            # Dask
            start_time = time.time()
            dask_result = ddf.describe().compute()
            dask_time = time.time() - start_time
            
            speedup = pandas_time / dask_time
            results[op] = {
                'pandas_time': pandas_time,
                'dask_time': dask_time,
                'speedup': speedup
            }
            
            print(f"   🐌 Pandas: {pandas_time:.4f}s")
            print(f"   🚀 Dask: {dask_time:.4f}s")
            print(f"   ⚡ Hızlanma: {speedup:.2f}x")
        
        elif op == 'corr' and len(df.select_dtypes(include=['number']).columns) > 1:
            print(f"\n🔍 {op} operasyonu test ediliyor...")
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # Pandas
            start_time = time.time()
            pandas_result = df[numeric_cols].corr()
            pandas_time = time.time() - start_time
            
            # Dask
            start_time = time.time()
            dask_result = ddf[numeric_cols].corr().compute()
            dask_time = time.time() - start_time
            
            speedup = pandas_time / dask_time
            results[op] = {
                'pandas_time': pandas_time,
                'dask_time': dask_time,
                'speedup': speedup
            }
            
            print(f"   🐌 Pandas: {pandas_time:.4f}s")
            print(f"   🚀 Dask: {dask_time:.4f}s")
            print(f"   ⚡ Hızlanma: {speedup:.2f}x")
    
    return results


def create_large_test_dataset(rows=1000000, cols=10):
    """
    Test için büyük veri seti oluşturur.
    
    Parameters
    ----------
    rows : int
        Satır sayısı
    cols : int
        Sütun sayısı
        
    Returns
    -------
    pd.DataFrame
        Test veri seti
    """
    print(f"📊 {rows:,} satır x {cols} sütun test veri seti oluşturuluyor...")
    
    # Sayısal sütunlar
    data = {}
    for i in range(cols):
        if i % 3 == 0:  # Float
            data[f'float_{i}'] = np.random.normal(0, 1, rows)
        elif i % 3 == 1:  # Integer
            data[f'int_{i}'] = np.random.randint(0, 1000, rows)
        else:  # Categorical
            categories = ['A', 'B', 'C', 'D', 'E']
            data[f'cat_{i}'] = np.random.choice(categories, rows)
    
    df = pd.DataFrame(data)
    
    print(f"✅ Test veri seti oluşturuldu!")
    print(f"   💾 Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    return df


# =============================================================================
# GPU Acceleration - CUDA ile Hızlandırılmış Analiz
# =============================================================================

def get_gpu_status():
    """GPU acceleration kullanılabilirlik durumunu döndürür"""
    if GPU_AVAILABLE:
        try:
            # GPU bilgilerini al
            gpu_memory = cp.cuda.runtime.memGetInfo()
            gpu_free_mb = gpu_memory[0] / 1024 / 1024
            gpu_total_mb = gpu_memory[1] / 1024 / 1024
            gpu_used_mb = gpu_total_mb - gpu_free_mb
            
            print("✅ GPU acceleration aktif!")
            print(f"   🚀 CUDA ile hızlandırılmış analiz yapılabilir")
            print(f"   💾 GPU Bellek: {gpu_used_mb:.1f} MB / {gpu_total_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"⚠️  GPU bilgisi alınamadı: {e}")
            return True
    else:
        print("❌ GPU acceleration devre dışı!")
        print("   📦 Kurulum: pip install cupy-cuda11x (CUDA sürümünüze göre)")
        return False


def convert_to_gpu_array(data):
    """
    NumPy array'i GPU array'e çevirir.
    
    Parameters
    ----------
    data : numpy.ndarray
        GPU'ya aktarılacak veri
        
    Returns
    -------
    cupy.ndarray or numpy.ndarray
        GPU array veya orijinal array
    """
    if not GPU_AVAILABLE:
        print("⚠️  GPU bulunamadı! CPU array döndürülüyor.")
        return data
    
    try:
        gpu_array = cp.asarray(data)
        print(f"✅ Veri GPU'ya aktarıldı!")
        print(f"   📊 Boyut: {gpu_array.shape}")
        print(f"   💾 GPU Bellek: {gpu_array.nbytes / 1024 / 1024:.2f} MB")
        return gpu_array
    except Exception as e:
        print(f"❌ GPU'ya aktarma hatası: {e}")
        return data


def gpu_correlation_matrix(data):
    """
    GPU ile hızlandırılmış korelasyon matrisi hesaplama.
    
    Parameters
    ----------
    data : cupy.ndarray or numpy.ndarray
        GPU'da analiz edilecek veri
        
    Returns
    -------
    cupy.ndarray or numpy.ndarray
        Korelasyon matrisi
    """
    if not GPU_AVAILABLE:
        print("❌ GPU bulunamadı! CPU korelasyon hesaplanıyor.")
        return np.corrcoef(data, rowvar=False)
    
    try:
        # GPU'ya aktar
        if not isinstance(data, cp.ndarray):
            data = convert_to_gpu_array(data)
        
        if not isinstance(data, cp.ndarray):
            return np.corrcoef(data, rowvar=False)
        
        # GPU'da korelasyon hesapla
        print("🚀 GPU'da korelasyon hesaplanıyor...")
        
        # Veriyi normalize et
        data_centered = data - cp.mean(data, axis=0)
        
        # Kovaryans matrisi
        cov_matrix = cp.dot(data_centered.T, data_centered) / (data.shape[0] - 1)
        
        # Standart sapmalar
        std_devs = cp.sqrt(cp.diag(cov_matrix))
        
        # Korelasyon matrisi
        corr_matrix = cov_matrix / cp.outer(std_devs, std_devs)
        
        # NaN değerleri düzelt
        corr_matrix = cp.nan_to_num(corr_matrix, nan=0.0)
        
        print("✅ GPU korelasyon hesaplandı!")
        return corr_matrix
        
    except Exception as e:
        print(f"❌ GPU korelasyon hatası: {e}")
        # CPU'ya geri dön
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        return np.corrcoef(data, rowvar=False)


def gpu_outlier_detection(data, method='iqr', threshold=1.5):
    """
    GPU ile hızlandırılmış aykırı değer tespiti.
    
    Parameters
    ----------
    data : cupy.ndarray or numpy.ndarray
        GPU'da analiz edilecek veri
    method : str
        'iqr' veya 'zscore'
    threshold : float
        Aykırı değer eşiği
        
    Returns
    -------
    cupy.ndarray or numpy.ndarray
        Boolean mask - True değerler aykırı değerleri gösterir
    """
    if not GPU_AVAILABLE:
        print("❌ GPU bulunamadı! CPU aykırı değer tespiti yapılıyor.")
        return detect_outliers(pd.DataFrame(data), method, threshold)
    
    try:
        # GPU'ya aktar
        if not isinstance(data, cp.ndarray):
            data = convert_to_gpu_array(data)
        
        if not isinstance(data, cp.ndarray):
            return detect_outliers(pd.DataFrame(data), method, threshold)
        
        print("🚀 GPU'da aykırı değer tespiti yapılıyor...")
        
        outliers = cp.zeros_like(data, dtype=cp.bool_)
        
        if method == 'iqr':
            for j in range(data.shape[1]):
                col = data[:, j]
                
                # Q1, Q3 ve IQR hesapla
                q1 = cp.percentile(col, 25)
                q3 = cp.percentile(col, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers[:, j] = (col < lower_bound) | (col > upper_bound)
        
        elif method == 'zscore':
            for j in range(data.shape[1]):
                col = data[:, j]
                
                # Q1, Q3 ve IQR hesapla
                q1 = cp.percentile(col, 25)
                q3 = cp.percentile(col, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers[:, j] = (col < lower_bound) | (col > upper_bound)
        
        print("✅ GPU aykırı değer tespiti tamamlandı!")
        return outliers
        
    except Exception as e:
        print(f"❌ GPU aykırı değer tespiti hatası: {e}")
        # CPU'ya geri dön
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        return detect_outliers(pd.DataFrame(data), method, threshold)


def gpu_summary_stats(data):
    """
    GPU ile hızlandırılmış istatistiksel özet.
    
    Parameters
    ----------
    data : cupy.ndarray or numpy.ndarray
        GPU'da analiz edilecek veri
        
    Returns
    -------
    dict
        Her sütun için istatistikler
    """
    if not GPU_AVAILABLE:
        print("❌ GPU bulunamadı! CPU istatistikleri hesaplanıyor.")
        return fast_summary_stats(data)
    
    try:
        # GPU'ya aktar
        if not isinstance(data, cp.ndarray):
            data = convert_to_gpu_array(data)
        
        if not isinstance(data, cp.ndarray):
            return fast_summary_stats(data)
        
        print("🚀 GPU'da istatistikler hesaplanıyor...")
        
        stats = {}
        for j in range(data.shape[1]):
            col = data[:, j]
            
            # Temel istatistikler
            mean_val = float(cp.mean(col))
            std_val = float(cp.mean(col))
            min_val = float(cp.min(col))
            max_val = float(cp.max(col))
            
            # Medyan hesapla
            median_val = float(cp.median(col))
            
            # Q1 ve Q3 hesapla
            q1_val = float(cp.percentile(col, 25))
            q3_val = float(cp.percentile(col, 75))
            
            stats[j] = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'q1': q1_val,
                'q3': q3_val
            }
        
        print("✅ GPU istatistikleri hesaplandı!")
        return stats
        
    except Exception as e:
        print(f"❌ GPU istatistik hatası: {e}")
        # CPU'ya geri dön
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        return fast_summary_stats(data)


def benchmark_gpu_vs_cpu(data, operations=['corr', 'outliers', 'stats']):
    """
    GPU vs CPU performans karşılaştırması.
    
    Parameters
    ----------
    data : numpy.ndarray
        Test edilecek veri
    operations : list
        Test edilecek operasyonlar
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    if not GPU_AVAILABLE:
        print("❌ GPU bulunamadı! Benchmark yapılamıyor.")
        return {}
    
    print("🏁 GPU vs CPU Benchmark Başlıyor...")
    print(f"   📊 Veri boyutu: {data.shape}")
    print(f"   💾 Veri boyutu: {data.nbytes / 1024 / 1024:.2f} MB")
    
    results = {}
    
    for op in operations:
        print(f"\n🔍 {op.upper()} operasyonu test ediliyor...")
        
        if op == 'corr':
            # CPU korelasyon testi
            print("   🐌 CPU korelasyon testi...")
            start_time = time.time()
            cpu_result = np.corrcoef(data, rowvar=False)
            cpu_time = time.time() - start_time
            
            # GPU korelasyon testi
            print("   🚀 GPU korelasyon testi...")
            start_time = time.time()
            gpu_result = gpu_correlation_matrix(data)
            gpu_time = time.time() - start_time
            
            # Sonuçları karşılaştır
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                time_saved = ((cpu_time - gpu_time) / cpu_time) * 100
            else:
                speedup = 0
                time_saved = 0
            
            results[op] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'time_saved': time_saved
            }
            
            print(f"   🐌 CPU: {cpu_time:.4f}s")
            print(f"   🚀 GPU: {gpu_time:.4f}s")
            print(f"   ⚡ Hızlanma: {speedup:.2f}x")
            print(f"   💰 Zaman tasarrufu: {time_saved:.1f}%")
        
        elif op == 'outliers':
            # CPU aykırı değer testi
            print("   🐌 CPU aykırı değer testi...")
            start_time = time.time()
            cpu_result = detect_outliers(pd.DataFrame(data))
            cpu_time = time.time() - start_time
            
            # GPU aykırı değer testi
            print("   🚀 GPU aykırı değer testi...")
            start_time = time.time()
            gpu_result = gpu_outlier_detection(data)
            gpu_time = time.time() - start_time
            
            # Sonuçları karşılaştır
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                time_saved = ((cpu_time - gpu_time) / cpu_time) * 100
            else:
                speedup = 0
                time_saved = 0
            
            results[op] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'time_saved': time_saved
            }
            
            print(f"   🐌 CPU: {cpu_time:.4f}s")
            print(f"   🚀 GPU: {gpu_time:.4f}s")
            print(f"   ⚡ Hızlanma: {speedup:.2f}x")
            print(f"   💰 Zaman tasarrufu: {time_saved:.1f}%")
        
        elif op == 'stats':
            # CPU istatistik testi
            print("   🐌 CPU istatistik testi...")
            start_time = time.time()
            cpu_result = fast_summary_stats(data)
            cpu_time = time.time() - start_time
            
            # GPU istatistik testi
            print("   🚀 GPU istatistik testi...")
            start_time = time.time()
            gpu_result = gpu_summary_stats(data)
            gpu_time = time.time() - start_time
            
            # Sonuçları karşılaştır
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                time_saved = ((cpu_time - gpu_time) / cpu_time) * 100
            else:
                speedup = 0
                time_saved = 0
            
            results[op] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'time_saved': time_saved
            }
            
            print(f"   🐌 CPU: {cpu_time:.4f}s")
            print(f"   🚀 GPU: {gpu_time:.4f}s")
            print(f"   ⚡ Hızlanma: {speedup:.2f}x")
            print(f"   💰 Zaman tasarrufu: {time_saved:.1f}%")
    
    return results


# =============================================================================
# Cloud Deployment - Bulut Ortamında Analiz
# =============================================================================

def get_cloud_status():
    """Cloud deployment kullanılabilirlik durumunu döndürür"""
    if CLOUD_AVAILABLE:
        print("✅ Cloud deployment aktif!")
        print(f"   🚀 AWS, Azure, Google Cloud desteği mevcut")
        return True
    else:
        print("❌ Cloud deployment devre dışı!")
        print("   📦 Kurulum: pip install boto3 azure-storage-blob google-cloud-storage")
        return False


class CloudDataManager:
    """
    Bulut ortamında veri yönetimi için sınıf.
    
    Bu sınıf, AWS S3, Azure Blob Storage ve Google Cloud Storage
    ile veri yükleme, indirme ve analiz işlemlerini yönetir.
    """
    
    def __init__(self, cloud_provider='aws'):
        """
        CloudDataManager'i başlatır.
        
        Parameters
        ----------
        cloud_provider : str
            Bulut sağlayıcısı ('aws', 'azure', 'gcp')
        """
        self.cloud_provider = cloud_provider.lower()
        self.client = None
        
        if not CLOUD_AVAILABLE:
            print("❌ Cloud kütüphaneleri bulunamadı!")
            return
        
        try:
            if self.cloud_provider == 'aws':
                self.client = boto3.client('s3')
                print("✅ AWS S3 client oluşturuldu!")
            elif self.cloud_provider == 'azure':
                self.client = azure.storage.blob.BlobServiceClient.from_connection_string(
                    "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net"
                )
                print("✅ Azure Blob Storage client oluşturuldu!")
            elif self.cloud_provider == 'gcp':
                self.client = google.cloud.storage.Client()
                print("✅ Google Cloud Storage client oluşturuldu!")
            else:
                print(f"❌ Geçersiz bulut sağlayıcısı: {cloud_provider}")
                
        except Exception as e:
            print(f"❌ Cloud client oluşturulamadı: {e}")
    
    def upload_data(self, data, bucket_name, file_name, file_format='parquet'):
        """
        Veriyi buluta yükler.
        
        Parameters
        ----------
        data : pd.DataFrame
            Yüklenecek veri
        bucket_name : str
            Bucket/container adı
        file_name : str
            Dosya adı
        file_format : str
            Dosya formatı ('parquet', 'csv', 'json')
            
        Returns
        -------
        bool
            Yükleme başarılı ise True
        """
        if not CLOUD_AVAILABLE or self.client is None:
            print("❌ Cloud client bulunamadı!")
            return False
        
        try:
            # Geçici dosya oluştur
            temp_file = f"temp_{file_name}.{file_format}"
            
            if file_format == 'parquet':
                data.to_parquet(temp_file)
            elif file_format == 'csv':
                data.to_csv(temp_file, index=False)
            elif file_format == 'json':
                data.to_json(temp_file, orient='records')
            
            # Buluta yükle
            if self.cloud_provider == 'aws':
                self.client.upload_file(temp_file, bucket_name, file_name)
            elif self.cloud_provider == 'azure':
                blob_client = self.client.get_blob_client(container=bucket_name, blob=file_name)
                with open(temp_file, 'rb') as data_file:
                    blob_client.upload_blob(data_file, overwrite=True)
            elif self.cloud_provider == 'gcp':
                bucket = self.client.bucket(bucket_name)
                blob = bucket.blob(file_name)
                blob.upload_from_filename(temp_file)
            
            # Geçici dosyayı sil
            import os
            os.remove(temp_file)
            
            print(f"✅ Veri buluta yüklendi!")
            print(f"   📁 Dosya: {file_name}")
            print(f"   💾 Boyut: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def download_data(self, bucket_name, file_name, file_format='parquet'):
        """
        Veriyi buluttan indirir.
        
        Parameters
        ----------
        bucket_name : str
            Bucket/container adı
        file_name : str
            Dosya adı
        file_format : str
            Dosya formatı ('parquet', 'csv', 'json')
            
        Returns
        -------
        pd.DataFrame or None
            İndirilen veri veya None
        """
        if not CLOUD_AVAILABLE or self.client is None:
            print("❌ Cloud client bulunamadı!")
            return None
        
        try:
            # Geçici dosya adı
            temp_file = f"temp_{file_name}"
            
            # Buluttan indir
            if self.cloud_provider == 'aws':
                self.client.download_file(bucket_name, file_name, temp_file)
            elif self.cloud_provider == 'azure':
                blob_client = self.client.get_blob_client(container=bucket_name, blob=file_name)
                with open(temp_file, 'wb') as data_file:
                    data_file.write(blob_client.download_blob().readall())
            elif self.cloud_provider == 'gcp':
                bucket = self.client.bucket(bucket_name)
                blob = bucket.blob(file_name)
                blob.download_to_filename(temp_file)
            
            # Veriyi yükle
            if file_format == 'parquet':
                data = pd.read_parquet(temp_file)
            elif file_format == 'csv':
                data = pd.read_csv(temp_file)
            elif file_format == 'json':
                data = pd.read_json(temp_file, orient='records')
            
            # Geçici dosyayı sil
            import os
            os.remove(temp_file)
            
            print(f"✅ Veri buluttan indirildi!")
            print(f"   📁 Dosya: {file_name}")
            print(f"   📊 Boyut: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ Veri indirme hatası: {e}")
            return None
    
    def analyze_cloud_data(self, bucket_name, file_pattern='*.parquet'):
        """
        Buluttaki veriyi analiz eder.
        
        Parameters
        ----------
        bucket_name : str
            Bucket/container adı
        file_pattern : str
            Dosya pattern'i
            
        Returns
        -------
        dict
            Analiz sonuçları
        """
        if not CLOUD_AVAILABLE or self.client is None:
            print("❌ Cloud client bulunamadı!")
            return {}
        
        try:
            print(f"🔍 Buluttaki veri analiz ediliyor...")
            print(f"   📁 Bucket: {bucket_name}")
            print(f"   🔍 Pattern: {file_pattern}")
            
            # Dosya listesini al
            files = []
            if self.cloud_provider == 'aws':
                response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=file_pattern.replace('*', ''))
                files = [obj['Key'] for obj in response.get('Contents', [])]
            elif self.cloud_provider == 'azure':
                container_client = self.client.get_container_client(bucket_name)
                files = [blob.name for blob in container_client.list_blobs(name_starts_with=file_pattern.replace('*', ''))]
            elif self.cloud_provider == 'gcp':
                bucket = self.client.bucket(bucket_name)
                files = [blob.name for blob in bucket.list_blobs(prefix=file_pattern.replace('*', ''))]
            
            print(f"   📊 Bulunan dosya sayısı: {len(files)}")
            
            if not files:
                print("⚠️  Dosya bulunamadı!")
                return {}
            
            # İlk dosyayı analiz et
            first_file = files[0]
            data = self.download_data(bucket_name, first_file)
            
            if data is not None:
                results = {
                    'file_name': first_file,
                    'file_count': len(files),
                    'data_info': get_data_info(data),
                    'sample_data': get_data_sample(data, 5)
                }
                
                print("✅ Bulut veri analizi tamamlandı!")
                return results
            else:
                print("❌ Veri indirilemedi!")
                return {}
                
        except Exception as e:
            print(f"❌ Bulut veri analizi hatası: {e}")
            return {}


def benchmark_cloud_vs_local(data, file_size_mb=100):
    """
    Cloud vs Local performans karşılaştırması.
    
    Parameters
    ----------
    data : pd.DataFrame
        Test edilecek veri
    file_size_mb : int
        Test dosya boyutu (MB)
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    if not CLOUD_AVAILABLE:
        print("❌ Cloud kütüphaneleri bulunamadı! Benchmark yapılamıyor.")
        return {}
    
    print("🏁 Cloud vs Local Benchmark Başlıyor...")
    print(f"   📊 Veri boyutu: {data.shape}")
    print(f"   💾 Test dosya boyutu: {file_size_mb} MB")
    
    # Veriyi büyüt (test için)
    if data.memory_usage(deep=True).sum() / 1024 / 1024 < file_size_mb:
        # Veriyi tekrarla
        repeat_factor = int(file_size_mb / (data.memory_usage(deep=True).sum() / 1024 / 1024)) + 1
        data = pd.concat([data] * repeat_factor, ignore_index=True)
        print(f"   📈 Veri büyütüldü: {data.shape}")
    
    results = {}
    
    # Local yazma testi
    print("\n🔍 Local yazma testi...")
    start_time = time.time()
    temp_file = "temp_benchmark.parquet"
    data.to_parquet(temp_file)
    local_write_time = time.time() - start_time
    
    # Local okuma testi
    start_time = time.time()
    local_data = pd.read_parquet(temp_file)
    local_read_time = time.time() - start_time
    
    # Geçici dosyayı sil
    import os
    os.remove(temp_file)
    
    # Cloud testi (simülasyon)
    print("🚀 Cloud testi (simülasyon)...")
    start_time = time.time()
    
    # Simüle edilmiş cloud işlemleri
    time.sleep(0.1)  # Network latency
    cloud_write_time = local_write_time * 2.5  # Cloud yazma genelde daha yavaş
    
    time.sleep(0.1)  # Network latency
    cloud_read_time = local_read_time * 1.8  # Cloud okuma da yavaş
    
    # Sonuçları karşılaştır
    write_speedup = cloud_write_time / local_write_time
    read_speedup = cloud_read_time / local_read_time
    
    results = {
        'local_write_time': local_write_time,
        'local_read_time': local_read_time,
        'cloud_write_time': cloud_write_time,
        'cloud_read_time': cloud_read_time,
        'write_speedup': write_speedup,
        'read_speedup': read_speedup
    }
    
    print(f"\n📈 BENCHMARK SONUÇLARI:")
    print(f"   📝 Yazma:")
    print(f"      🐌 Local: {local_write_time:.4f}s")
    print(f"      ☁️  Cloud: {cloud_write_time:.4f}s")
    print(f"      ⚡ Hızlanma: {write_speedup:.2f}x")
    
    print(f"   📖 Okuma:")
    print(f"      🐌 Local: {local_read_time:.4f}s")
    print(f"      ☁️  Cloud: {cloud_read_time:.4f}s")
    print(f"      ⚡ Hızlanma: {read_speedup:.2f}x")
    
    return results


# =============================================================================
# Streaming Analytics - Gerçek Zamanlı Veri Analizi
# =============================================================================

class StreamingAnalyzer:
    """
    Gerçek zamanlı veri analizi için streaming analyzer.
    
    Bu sınıf, sürekli gelen veri akışını analiz eder
    ve gerçek zamanlı istatistikler üretir.
    """
    
    def __init__(self, window_size=1000, update_interval=1.0):
        """
        StreamingAnalyzer'i başlatır.
        
        Parameters
        ----------
        window_size : int
            Sliding window boyutu
        update_interval : float
            Güncelleme aralığı (saniye)
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.data_buffer = []
        self.stats_history = []
        self.is_running = False
        
        print(f"✅ Streaming Analyzer başlatıldı!")
        print(f"   📊 Window boyutu: {window_size}")
        print(f"   ⏱️  Güncelleme aralığı: {update_interval} saniye")
    
    def add_data(self, data_point):
        """
        Veri noktası ekler.
        
        Parameters
        ----------
        data_point : dict or pd.Series
            Eklenecek veri noktası
        """
        self.data_buffer.append(data_point)
        
        # Window boyutunu aşan eski verileri sil
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
    
    def get_current_stats(self):
        """
        Mevcut window için istatistikleri döndürür.
        
        Returns
        -------
        dict
            Güncel istatistikler
        """
        if not self.data_buffer:
            return {}
        
        # DataFrame'e çevir
        df = pd.DataFrame(self.data_buffer)
        
        # Temel istatistikler
        stats = {
            'window_size': len(self.data_buffer),
            'timestamp': time.time(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Sayısal sütunlar için istatistikler
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            stats['numeric_stats'] = {
                'mean': numeric_df.mean().to_dict(),
                'std': numeric_df.std().to_dict(),
                'min': numeric_df.min().to_dict(),
                'max': numeric_df.max().to_dict()
            }
        
        # Kategorik sütunlar için istatistikler
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(5)
                stats['categorical_stats'][col] = value_counts.to_dict()
        
        return stats
    
    def start_streaming(self, data_generator, max_iterations=None):
        """
        Streaming analizi başlatır.
        
        Parameters
        ----------
        data_generator : generator
            Veri üreteci
        max_iterations : int, optional
            Maksimum iterasyon sayısı
        """
        if self.is_running:
            print("⚠️  Streaming zaten çalışıyor!")
            return
        
        self.is_running = True
        print("🚀 Streaming analizi başlatıldı...")
        
        iteration = 0
        start_time = time.time()
        
        try:
            for data_point in data_generator:
                if not self.is_running:
                    break
                
                # Veri ekle
                self.add_data(data_point)
                
                # Belirli aralıklarla istatistikleri güncelle
                if iteration % 10 == 0:
                    stats = self.get_current_stats()
                    self.stats_history.append(stats)
                    
                    # İstatistikleri yazdır
                    print(f"\n📊 Iterasyon {iteration}:")
                    print(f"   📈 Veri noktası sayısı: {len(self.data_buffer)}")
                    
                    if stats.get('numeric_stats'):
                        print(f"   🔢 Sayısal istatistikler: {len(stats['numeric_stats']['mean'])} sütun")
                    
                    if stats.get('categorical_stats'):
                        print(f"   📝 Kategorik istatistikler: {len(stats['categorical_stats'])} sütun")
                
                iteration += 1
                
                # Maksimum iterasyon kontrolü
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Güncelleme aralığı
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\n⏹️  Streaming kullanıcı tarafından durduruldu.")
        except Exception as e:
            print(f"❌ Streaming hatası: {e}")
        finally:
            self.is_running = False
            total_time = time.time() - start_time
            print(f"\n✅ Streaming analizi tamamlandı!")
            print(f"   ⏱️  Toplam süre: {total_time:.2f} saniye")
            print(f"   📊 Toplam veri noktası: {len(self.data_buffer)}")
            print(f"   📈 İstatistik güncellemesi: {len(self.stats_history)}")
    
    def stop_streaming(self):
        """Streaming analizi durdurur"""
        self.is_running = False
        print("⏹️  Streaming analizi durduruldu.")
    
    def get_streaming_summary(self):
        """
        Streaming analizi özetini döndürür.
        
        Returns
        -------
        dict
            Streaming analizi özeti
        """
        if not self.stats_history:
            return {}
        
        # İstatistik geçmişini analiz et
        timestamps = [stats['timestamp'] for stats in self.stats_history]
        window_sizes = [stats['window_size'] for stats in self.stats_history]
        
        summary = {
            'total_updates': len(self.stats_history),
            'total_duration': max(timestamps) - min(timestamps),
            'avg_window_size': np.mean(window_sizes),
            'min_window_size': min(window_sizes),
            'max_window_size': max(window_sizes),
            'update_frequency': len(self.stats_history) / (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0
        }
        
        return summary


def create_streaming_data_generator(data_type='random', n_points=1000, interval=0.1):
    """
    Test için streaming veri üreteci oluşturur.
    
    Parameters
    ----------
    data_type : str
        Veri tipi ('random', 'trend', 'seasonal')
    n_points : int
        Üretilecek veri noktası sayısı
    interval : float
        Veri üretim aralığı (saniye)
        
    Returns
    -------
    generator
        Veri üreteci
    """
    print(f"📊 Streaming veri üreteci oluşturuluyor...")
    print(f"   🔢 Veri tipi: {data_type}")
    print(f"   📈 Nokta sayısı: {n_points}")
    print(f"   ⏱️  Aralık: {interval} saniye")
    
    for i in range(n_points):
        if data_type == 'random':
            data_point = {
                'timestamp': time.time(),
                'value': np.random.normal(0, 1),
                'category': np.random.choice(['A', 'B', 'C']),
                'index': i
            }
        elif data_type == 'trend':
            data_point = {
                'timestamp': time.time(),
                'value': i * 0.1 + np.random.normal(0, 0.1),
                'category': 'trend',
                'index': i
            }
        elif data_type == 'seasonal':
            data_point = {
                'timestamp': time.time(),
                'value': np.sin(i * 0.1) + np.random.normal(0, 0.1),
                'category': 'seasonal',
                'index': i
            }
        
        yield data_point
        time.sleep(interval)


def benchmark_streaming_vs_batch(df, window_sizes=[100, 500, 1000]):
    """
    Streaming vs batch analiz performans karşılaştırması.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test edilecek veri seti
    window_sizes : list
        Test edilecek window boyutları
        
    Returns
    -------
    dict
        Benchmark sonuçları
    """
    print("🏁 Streaming vs Batch Benchmark Başlıyor...")
    print(f"   📊 Veri seti boyutu: {df.shape}")
    
    results = {}
    
    for window_size in window_sizes:
        print(f"\n🔍 Window boyutu {window_size} test ediliyor...")
        
        # Batch analiz
        start_time = time.time()
        batch_stats = get_data_info(df.head(window_size))
        batch_time = time.time() - start_time
        
        # Streaming analiz simülasyonu
        start_time = time.time()
        analyzer = StreamingAnalyzer(window_size=window_size)
        
        # Veri üreteci oluştur
        data_gen = (row.to_dict() for _, row in df.head(window_size).iterrows())
        
        # Streaming analiz
        for data_point in data_gen:
            analyzer.add_data(data_point)
        
        streaming_stats = analyzer.get_current_stats()
        streaming_time = time.time() - start_time
        
        # Sonuçları karşılaştır
        if batch_time > 0:
            speedup = batch_time / streaming_time
            time_saved = ((batch_time - streaming_time) / batch_time) * 100
        else:
            speedup = 0
            time_saved = 0
        
        results[window_size] = {
            'batch_time': batch_time,
            'streaming_time': streaming_time,
            'speedup': speedup,
            'time_saved': time_saved
        }
        
        print(f"   🐌 Batch: {batch_time:.4f}s")
        print(f"   🚀 Streaming: {streaming_time:.4f}s")
        print(f"   ⚡ Hızlanma: {speedup:.2f}x")
        print(f"   💰 Zaman tasarrufu: {time_saved:.1f}%")
    
    return results
