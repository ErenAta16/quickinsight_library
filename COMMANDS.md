# 🚀 QuickInsights Komut Listesi

Bu dokümanda QuickInsights kütüphanesinin tüm komutları ve kullanım örnekleri bulunmaktadır.

## 📊 Temel Analiz Komutları

### **Veri Seti Genel Bilgileri**
```python
import quickinsights as qi
import pandas as pd

# Veri seti hakkında genel bilgi
df = pd.read_csv('data.csv')
info = qi.get_data_info(df)

# Veri seti doğrulama
is_valid = qi.validate_dataframe(df)
```

### **Sayısal Değişken Analizi**
```python
# Sayısal değişkenler için detaylı analiz
numeric_analysis = qi.analyze_numeric(df)

# İstatistiksel özet
summary = qi.summary_stats(df)

# Aykırı değer tespiti
outliers = qi.detect_outliers(df)
```

### **Kategorik Değişken Analizi**
```python
# Kategorik değişkenler için analiz
categorical_analysis = qi.analyze_categorical(df)
```

### **Kapsamlı Analiz**
```python
# Tek komutla tüm analizleri yap
results = qi.analyze(df, save_plots=True, output_dir="./output")
```

## 🎨 Görselleştirme Komutları

### **Korelasyon Analizi**
```python
# Korelasyon matrisi
qi.correlation_matrix(df, method='pearson', save_plots=True)

# İnteraktif korelasyon matrisi
qi.create_interactive_plots(df, save_html=True)
```

### **Dağılım Grafikleri**
```python
# Dağılım grafikleri
qi.distribution_plots(df, save_plots=True)

# Kutu grafikleri
qi.box_plots(df, save_plot=True)
```

## ⚡ Performans Optimizasyon Komutları

### **Lazy Evaluation**
```python
# Lazy evaluation ile fonksiyon sarmalama
@qi.lazy_evaluate
def expensive_function(x):
    # Pahalı hesaplama
    return x ** 2

# Fonksiyon sadece çağrıldığında çalışır
lazy_result = expensive_function(5)
result = lazy_result()  # Şimdi çalışır
```

### **Caching (Önbellekleme)**
```python
# Sonuçları cache'leme
@qi.cache_result(ttl=3600)  # 1 saat
def slow_function(x):
    # Yavaş hesaplama
    return x ** 3

# İlk çağrı yavaş, sonraki çağrılar hızlı
result1 = slow_function(5)
result2 = slow_function(5)  # Cache'den gelir
```

### **Parallel Processing**
```python
# Paralel işleme
def process_item(x):
    return x ** 2

# 10 öğeyi paralel işle
results = qi.parallel_process(process_item, range(10))
```

### **Chunked Processing**
```python
# Büyük veri setlerini parçalara bölerek işleme
def process_chunk(chunk):
    return chunk.sum()

# 1000'lik parçalar halinde işle
chunk_results = qi.chunked_process(process_chunk, df, chunk_size=1000)
```

### **Memory Optimization**
```python
# Bellek optimizasyonu
optimized_df = qi.memory_optimize(df)
```

### **Performance Profiling**
```python
# Performans profili
def test_function():
    return sum(range(1000000))

profile_result = qi.performance_profile(test_function)
```

### **Benchmarking**
```python
# Fonksiyon benchmark'ı
benchmark_result = qi.benchmark_function(test_function, test_data, iterations=100)
```

## 📊 Big Data Komutları

### **Status Kontrolleri**
```python
# Dask durumu
dask_status = qi.get_dask_status()

# GPU durumu
gpu_status = qi.get_gpu_status()

# Memory mapping durumu
memory_mapping_status = qi.get_memory_mapping_status()

# Distributed computing durumu
distributed_status = qi.get_distributed_status()
```

### **Memory Management**
```python
# Bellek kullanım tahmini
memory_info = qi.estimate_memory_usage(df)

# Sistem bellek bilgisi
system_memory = qi.get_system_memory_info()

# Bellek kısıtlamaları kontrolü
constraints = qi.check_memory_constraints(estimated_memory_mb)
```

### **Large File Processing**
```python
# Büyük dosyaları parçalara bölerek işleme
for chunk in qi.process_large_file('large_file.csv', chunk_size=10000):
    processed_chunk = process_chunk(chunk)
    # İşlenmiş chunk'ı kaydet
```

### **Streaming Data**
```python
# Veri akışı işleme
for batch in qi.stream_data(df, batch_size=1000):
    processed_batch = process_batch(batch)
    # İşlenmiş batch'i kaydet
```

## ☁️ Cloud Integration Komutları

### **Status Kontrolleri**
```python
# AWS durumu
aws_status = qi.get_aws_status()

# Azure durumu
azure_status = qi.get_azure_status()

# Google Cloud durumu
gcp_status = qi.get_gcp_status()
```

### **Cloud Operations**
```python
# Dosya yükleme
upload_result = qi.upload_to_cloud('local_file.csv', 'aws', 'file_key.csv', bucket_name='my-bucket')

# Dosya indirme
download_result = qi.download_from_cloud('aws', 'my-bucket', 'file_key.csv')

# Dosya listesi
files = qi.list_cloud_files('aws')

# Cloud veri işleme
def process_cloud_data(data):
    return data.upper()

result = qi.process_cloud_data('aws', 'my-bucket', process_cloud_data)
```

## ✅ Data Validation Komutları

### **Column Type Validation**
```python
# Sütun tipi doğrulama
expected_types = {
    'age': 'numeric',
    'name': 'object',
    'salary': 'numeric'
}
validation_result = qi.validate_column_types(df, expected_types)
```

### **Data Quality Check**
```python
# Veri kalitesi kontrolü
quality_report = qi.check_data_quality(df)
```

### **Data Cleaning**
```python
# Veri temizleme
cleaned_df = qi.clean_data(df)
```

### **Schema Validation**
```python
# Şema doğrulama
schema = {
    'age': {'type': 'numeric', 'required': True},
    'name': {'type': 'object', 'required': True}
}
schema_validation = qi.validate_schema(df, schema)
```

### **Anomaly Detection**
```python
# Anomali tespiti
anomalies = qi.detect_anomalies(df)
```

### **Format Validation**
```python
# Email format doğrulama
email_valid = qi.validate_email_format('test@example.com')

# Telefon format doğrulama
phone_valid = qi.validate_phone_format('+90-555-123-4567')

# Tarih format doğrulama
date_valid = qi.validate_date_format('2024-01-15')
```

## 🔧 Utility Komutları

### **Status ve Bilgi**
```python
# Tüm utility durumları
utility_status = qi.get_utility_status()

# Durumları yazdır
qi.print_utility_status()

# Mevcut özellikler
features = qi.get_available_features()

# Bağımlılık kontrolü
dependencies = qi.check_dependencies()

# Sistem bilgisi
system_info = qi.get_system_info()

# Utility raporu
report = qi.create_utility_report()
```

### **Lazy Loading**
```python
# Performance utilities
performance_utils = qi.get_performance_utils()

# Big data utilities
big_data_utils = qi.get_big_data_utils()

# GPU utilities
gpu_utils = qi.get_gpu_utils()

# Cloud utilities
cloud_utils = qi.get_cloud_utils()

# Validation utilities
validation_utils = qi.get_validation_utils()

# Tüm utilities
all_utils = qi.get_all_utils()
```

## 🎯 Kullanım Senaryoları

### **Hızlı Veri Keşfi**
```python
import quickinsights as qi
import pandas as pd

# Veri setini yükle
df = pd.read_csv('data.csv')

# Tek komutla kapsamlı analiz
qi.analyze(df, save_plots=True, show_plots=False)
```

### **Performans Optimizasyonu**
```python
# Büyük veri seti için optimizasyon
@qi.lazy_evaluate
@qi.cache_result(ttl=3600)
def expensive_analysis(df):
    return df.groupby('category').agg(['mean', 'std', 'count'])

# Analiz sadece gerektiğinde yapılır ve sonuçlar cache'lenir
lazy_analysis = expensive_analysis(large_df)
```

### **Cloud Data Processing**
```python
# Cloud'dan veri indir ve işle
def process_data(data):
    # Veri işleme mantığı
    return processed_data

result = qi.process_cloud_data('aws', 'data-bucket', process_data)
```

### **Real-time Data Validation**
```python
# Gerçek zamanlı veri doğrulama
def validate_stream_data(data):
    if qi.validate_dataframe(data):
        quality_report = qi.check_data_quality(data)
        if quality_report['overall_score'] > 0.8:
            return True
    return False
```

## 📚 Komut Referansı

### **Fonksiyon Kategorileri**
- **Core Analysis**: `analyze`, `get_data_info`, `analyze_numeric`, `analyze_categorical`
- **Visualization**: `correlation_matrix`, `distribution_plots`, `box_plots`, `create_interactive_plots`
- **Performance**: `lazy_evaluate`, `cache_result`, `parallel_process`, `chunked_process`
- **Big Data**: `process_large_file`, `stream_data`, `estimate_memory_usage`
- **Cloud**: `upload_to_cloud`, `download_from_cloud`, `list_cloud_files`
- **Validation**: `validate_dataframe`, `check_data_quality`, `clean_data`
- **Utilities**: `get_utility_status`, `check_dependencies`, `get_system_info`

### **Parametreler**
- **save_plots**: Grafikleri kaydet (True/False)
- **show_plots**: Grafikleri göster (True/False)
- **output_dir**: Çıktı dizini
- **chunk_size**: Parça boyutu
- **max_workers**: Maksimum worker sayısı
- **ttl**: Cache süresi (saniye)
- **iterations**: Benchmark iterasyon sayısı

Bu komut listesi QuickInsights kütüphanesinin tüm özelliklerini kapsamaktadır. Her komut için detaylı dokümantasyon ve örnekler kullanıcıların hızlıca başlamasına yardımcı olur.
