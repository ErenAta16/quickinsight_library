# QuickInsights API Reference

## 📚 Genel Bakış

QuickInsights, büyük veri setleri için yaratıcı ve yenilikçi analiz araçları sağlayan Python kütüphanesidir.

## 🔧 Kurulum

```bash
pip install quickinsights
```

## 📊 Ana Fonksiyonlar

### `analyze(df, show_plots=True, save_plots=False, output_dir="./quickinsights_output")`

Veri seti üzerinde kapsamlı analiz yapar.

**Parametreler:**
- `df` (pd.DataFrame): Analiz edilecek veri seti
- `show_plots` (bool): Grafikleri göstermek isteyip istemediğiniz
- `save_plots` (bool): Grafikleri kaydetmek isteyip istemediğiniz
- `output_dir` (str): Grafiklerin kaydedileceği dizin

**Dönen Değer:**
- `dict`: Analiz sonuçları

**Örnek:**
```python
import quickinsights as qi
import pandas as pd

df = pd.read_csv('data.csv')
results = qi.analyze(df, save_plots=True)
```

### `analyze_numeric(df, show_plots=True, save_plots=False, output_dir="./quickinsights_output")`

Sayısal değişkenler üzerinde detaylı analiz yapar.

**Parametreler:**
- `df` (pd.DataFrame): Sadece sayısal değişkenler içeren veri seti
- `show_plots` (bool): Grafikleri göstermek isteyip istemediğiniz
- `save_plots` (bool): Grafikleri kaydetmek isteyip istemediğiniz
- `output_dir` (str): Grafiklerin kaydedileceği dizin

**Dönen Değer:**
- `dict`: Sayısal analiz sonuçları

### `analyze_categorical(df, show_plots=True, save_plots=False, output_dir="./quickinsights_output")`

Kategorik değişkenler üzerinde detaylı analiz yapar.

**Parametreler:**
- `df` (pd.DataFrame): Sadece kategorik değişkenler içeren veri seti
- `show_plots` (bool): Grafikleri göstermek isteyip istemediğiniz
- `save_plots` (bool): Grafikleri kaydetmek isteyip istemediğiniz
- `output_dir` (str): Grafiklerin kaydedileceği dizin

**Dönen Değer:**
- `dict`: Kategorik analiz sonuçları

## 🎨 Görselleştirme Fonksiyonları

### `correlation_matrix(df, method='pearson', save_plots=False, output_dir="./quickinsights_output")`

Sayısal değişkenler arası korelasyon matrisini görselleştirir.

**Parametreler:**
- `df` (pd.DataFrame): Sadece sayısal değişkenler içeren veri seti
- `method` (str): Korelasyon hesaplama yöntemi ('pearson', 'spearman')
- `save_plots` (bool): Grafiği kaydetmek isteyip istemediğiniz
- `output_dir` (str): Grafiğin kaydedileceği dizin

### `distribution_plots(df, save_plots=False, output_dir="./quickinsights_output")`

Sayısal değişkenlerin dağılım grafiklerini oluşturur.

**Parametreler:**
- `df` (pd.DataFrame): Sadece sayısal değişkenler içeren veri seti
- `save_plots` (bool): Grafikleri kaydetmek isteyip istemediğiniz
- `output_dir` (str): Grafiklerin kaydedileceği dizin

### `summary_stats(df)`

Veri setinin istatistiksel özetini hesaplar.

**Parametreler:**
- `df` (pd.DataFrame): Analiz edilecek veri seti

**Dönen Değer:**
- `dict`: İstatistiksel özet

## 🛠️ Yardımcı Fonksiyonlar

### `get_data_info(df)`

Veri seti hakkında genel bilgi verir.

**Parametreler:**
- `df` (pd.DataFrame): Analiz edilecek veri seti

**Dönen Değer:**
- `dict`: Veri seti bilgileri

### `detect_outliers(df)`

Veri setindeki aykırı değerleri tespit eder.

**Parametreler:**
- `df` (pd.DataFrame): Analiz edilecek veri seti

**Dönen Değer:**
- `pd.DataFrame`: Aykırı değer matrisi (boolean)

### `optimize_dtypes(df)`

Veri tiplerini optimize ederek bellek kullanımını azaltır.

**Parametreler:**
- `df` (pd.DataFrame): Optimize edilecek veri seti

**Dönen Değer:**
- `pd.DataFrame`: Optimize edilmiş veri seti

### `validate_dataframe(df)`

DataFrame'in geçerli olup olmadığını kontrol eder.

**Parametreler:**
- `df`: Kontrol edilecek veri

**Dönen Değer:**
- `bool`: DataFrame geçerliyse True, değilse False

**Hatalar:**
- `TypeError`: Veri bir DataFrame değilse
- `ValueError`: DataFrame boşsa

## 🚀 Performans Sınıfları

### `LazyAnalyzer(df)`

Lazy evaluation ile veri analizi yapar.

**Parametreler:**
- `df` (pd.DataFrame): Analiz edilecek veri seti

**Metodlar:**
- `get_data_info()`: Veri seti bilgilerini döndürür
- `get_numeric_analysis()`: Sayısal analiz yapar
- `get_categorical_analysis()`: Kategorik analiz yapar
- `get_all_analysis()`: Tüm analizleri yapar

**Örnek:**
```python
lazy_analyzer = qi.LazyAnalyzer(df)
data_info = lazy_analyzer.get_data_info()
numeric_analysis = lazy_analyzer.get_numeric_analysis()
```

## 🔄 Paralel İşleme

### `parallel_analysis(df, backend='thread', n_jobs=-1)`

Paralel olarak veri analizi yapar.

**Parametreler:**
- `df` (pd.DataFrame): Analiz edilecek veri seti
- `backend` (str): Paralel işleme backend'i ('thread', 'process')
- `n_jobs` (int): Kullanılacak iş parçacığı sayısı (-1 = tüm CPU'lar)

**Dönen Değer:**
- `dict`: Paralel analiz sonuçları

### `chunked_analysis(df, chunk_size=10000, n_jobs=-1)`

Büyük veri setlerini chunk'lara bölerek analiz eder.

**Parametreler:**
- `df` (pd.DataFrame): Analiz edilecek veri seti
- `chunk_size` (int): Her chunk'taki satır sayısı
- `n_jobs` (int): Kullanılacak iş parçacığı sayısı

**Dönen Değer:**
- `dict`: Chunked analiz sonuçları

## 🎨 Yaratıcı Görselleştirme

### `CreativeVizEngine(df)`

Yaratıcı görselleştirmeler oluşturur.

**Metodlar:**
- `create_radar_chart(numeric_cols, title)`: Radar chart
- `create_3d_scatter(x_col, y_col, z_col)`: 3D scatter plot
- `create_interactive_network(source_col, target_col)`: Network graph
- `create_animated_timeline(time_col, value_col)`: Animasyonlu timeline
- `create_sunburst_chart(path_cols, value_col)`: Sunburst chart

## 🧠 AI Destekli Analiz

### `AIInsightEngine(df)`

AI destekli veri insights sağlar.

**Metodlar:**
- `discover_patterns(max_patterns=10)`: Otomatik pattern discovery
- `detect_anomalies(method='auto')`: Anomali tespiti
- `predict_trends(target_col, horizon=5)`: Trend tahmini
- `generate_insights_report()`: Kapsamlı insights raporu

## ⚡ Gerçek Zamanlı Analiz

### `RealTimePipeline(name)`

Gerçek zamanlı veri işleme pipeline'ı.

**Metodlar:**
- `add_transformation(transformation)`: Dönüşüm ekler
- `add_outlier_detector(method, threshold)`: Outlier detector ekler
- `add_anomaly_detector(window_size, threshold)`: Anomaly detector ekler
- `start()`: Pipeline'ı başlatır
- `stop()`: Pipeline'ı durdurur
- `process_data(data)`: Veriyi işler

## 📊 Performans Benchmark'ları

### `get_performance_utils()`

Performans utilities için lazy import.

**Dönen Değer:**
- `dict`: Performans utility fonksiyonları

### `get_big_data_utils()`

Big data utilities için lazy import.

**Dönen Değer:**
- `dict`: Big data utility fonksiyonları

### `get_gpu_utils()`

GPU utilities için lazy import.

**Dönen Değer:**
- `dict`: GPU utility fonksiyonları

### `get_cloud_utils()`

Cloud utilities için lazy import.

**Dönen Değer:**
- `dict`: Cloud utility fonksiyonları

## 🔧 Konfigürasyon

### Çıktı Dizini
Varsayılan çıktı dizini: `./quickinsights_output`

### Grafik Formatları
- PNG: Yüksek kalite, düşük boyut
- HTML: İnteraktif Plotly grafikleri

### Bellek Optimizasyonu
- Otomatik veri tipi optimizasyonu
- Lazy evaluation
- Chunked processing

## 📝 Örnekler

Daha fazla örnek için `examples/` dizinini inceleyin:
- `basic_usage.py`: Temel kullanım örnekleri
- `performance_benchmarks.py`: Performans benchmark'ları
- `big_data_analysis.py`: Büyük veri analizi örnekleri

## 🐛 Hata Yönetimi

Kütüphane, hataları graceful bir şekilde yönetir:
- Eksik bağımlılıklar için uyarılar
- Geçersiz veri tipleri için TypeError
- Boş veri setleri için ValueError
- Dosya yazma hataları için IOError

## 📈 Performans İpuçları

1. **Lazy Evaluation**: Sadece gerektiğinde analiz yapın
2. **Chunked Processing**: Büyük veri setleri için chunk'ları kullanın
3. **Parallel Processing**: Çoklu CPU çekirdeklerini kullanın
4. **Memory Optimization**: Veri tiplerini optimize edin
5. **GPU Acceleration**: Uygun olduğunda GPU kullanın
