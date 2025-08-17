# QuickInsights Proje Yapısı 📁

Bu dosya, QuickInsights Python kütüphanesinin proje yapısını ve dosya organizasyonunu açıklar.

## 🏗️ Proje Dizin Yapısı

```
QuickInsights/
├── 📁 src/
│   └── 📁 quickinsights/
│       ├── 📄 __init__.py          # Ana kütüphane giriş noktası (sadeleştirilmiş)
│       ├── 📄 core.py              # Ana analiz fonksiyonları
│       ├── 📄 visualizer.py        # Görselleştirme modülü
│       └── 📄 utils.py             # Yardımcı fonksiyonlar ve performans araçları
├── 📁 examples/                     # ✅ YENİ: Kullanım örnekleri
│   ├── 📄 basic_usage.py          # Temel kullanım örneği
│   ├── 📄 performance_benchmarks.py # Performans benchmark'ları
│   └── 📄 quickinsights_tutorial.ipynb # Jupyter notebook tutorial
├── 📁 tests/                       # ✅ GELİŞTİRİLDİ: Kapsamlı test suite
│   └── 📄 test_quickinsights.py   # Test dosyaları (test coverage artırıldı)
├── 📄 README.md                    # Proje açıklaması
├── 📄 setup.py                     # Kurulum konfigürasyonu
├── 📄 requirements.txt             # ✅ GÜNCELLENDİ: Tüm bağımlılıklar
├── 📄 PROJECT_STRUCTURE.md         # Bu dosya
└── 📄 OPTIMIZATION_ROADMAP.md      # Performans optimizasyon yol haritası
```

## 📚 Modül Açıklamaları

### 🔧 `src/quickinsights/__init__.py` ✅ GÜNCELLENDİ
- Kütüphanenin ana giriş noktası
- **Sadeleştirilmiş import yapısı** (50+ fonksiyon → 20+ ana fonksiyon)
- **Lazy imports** ile gelişmiş özellikler
- Modüler organizasyon (core, visualization, utilities)
- Versiyon ve yazar bilgileri

### 🎯 `src/quickinsights/core.py`
- **`analyze()`**: Ana analiz fonksiyonu
- **`analyze_numeric()`**: Sayısal değişken analizi
- **`analyze_categorical()`**: Kategorik değişken analizi
- **`LazyAnalyzer`**: Lazy evaluation ile analiz
- **`parallel_analysis()`**: Paralel işleme
- **`chunked_analysis()`**: Büyük veri setleri için parçalı analiz

### 🎨 `src/quickinsights/visualizer.py`
- **`correlation_matrix()`**: Korelasyon matrisi görselleştirme
- **`distribution_plots()`**: Dağılım grafikleri
- **`summary_stats()`**: İstatistiksel özetler
- **`create_interactive_plots()`**: Plotly interaktif grafikler
- **`box_plots()`**: Kutu grafikleri

### 🛠️ `src/quickinsights/utils.py` ✅ GENİŞLETİLDİ
- **`get_data_info()`**: Veri seti bilgileri
- **`detect_outliers()`**: Aykırı değer tespiti
- **`optimize_dtypes()`**: Veri tipi optimizasyonu
- **`get_data_sample()`**: Veri örneği alma
- **`AnalysisCache`**: Caching sistemi
- **Performance utilities**: Numba, Dask, GPU desteği
- **Big data utilities**: Paralel işleme, memory mapping
- **Cloud utilities**: AWS, Azure, Google Cloud entegrasyonu

## 🚀 Kurulum ve Kullanım

### Geliştirme Kurulumu
```bash
# Projeyi klonla
git clone <repository-url>
cd QuickInsights

# Geliştirme bağımlılıklarını kur
pip install -e .

# Testleri çalıştır
python -m pytest tests/ -v

# Coverage ile test
python -m pytest --cov=quickinsights tests/
```

### Kullanıcı Kurulumu
```bash
# PyPI'dan kur
pip install quickinsights

# Gelişmiş özellikler ile
pip install quickinsights[fast,gpu,cloud,profiling]

# Veya geliştirme sürümünden
pip install git+<repository-url>
```

## 📖 Kullanım Örnekleri ✅ YENİ

### Temel Kullanım
```python
import quickinsights as qi
import pandas as pd

# Veri setini yükle
df = pd.read_csv('veri.csv')

# Tek komutla analiz et
results = qi.analyze(df)
```

### Gelişmiş Kullanım
```python
# Lazy analyzer ile
lazy_analyzer = qi.LazyAnalyzer(df)
data_info = lazy_analyzer.get_data_info()
numeric_analysis = lazy_analyzer.get_numeric_analysis()

# Performans optimizasyonu
optimized_df = qi.optimize_dtypes(df)

# Paralel analiz
parallel_results = qi.parallel_analysis(df, n_jobs=4)
```

### Örnek Dosyaları Çalıştırma
```bash
# Temel kullanım
python examples/basic_usage.py

# Performans benchmark'ları
python examples/performance_benchmarks.py

# Jupyter notebook
jupyter notebook examples/quickinsights_tutorial.ipynb
```

## 🧪 Test Sistemi ✅ GELİŞTİRİLDİ

### Test Çalıştırma
```bash
# Tüm testleri çalıştır
python -m pytest tests/ -v

# Belirli test dosyası
python -m pytest tests/test_quickinsights.py -v

# Coverage ile test
python -m pytest --cov=quickinsights tests/ --cov-report=html

# Performance testleri
python -m pytest tests/ -k "performance" -v
```

### Test Kapsamı ✅ GENİŞLETİLDİ
- ✅ Veri doğrulama ve hata yönetimi
- ✅ Aykırı değer tespiti
- ✅ İstatistiksel hesaplamalar
- ✅ Görselleştirme fonksiyonları
- ✅ Lazy analyzer ve caching
- ✅ Paralel işleme ve chunked analiz
- ✅ Veri tipi optimizasyonu
- ✅ Edge cases ve boundary conditions
- ✅ Performance features
- ✅ Memory usage tracking

## 📦 Dağıtım

### PyPI'ya Yükleme
```bash
# Build oluştur
python setup.py sdist bdist_wheel

# PyPI'ya yükle
twine upload dist/*

# Test PyPI'ya yükle
twine upload --repository testpypi dist/*
```

### Yerel Kurulum
```bash
# Geliştirme modunda kur
pip install -e .

# Normal kurulum
pip install .
```

## 🔄 Geliştirme Döngüsü ✅ GÜNCELLENDİ

1. **Kod Yazma** → `src/quickinsights/` altında
2. **Test Yazma** → `tests/` altında (kapsamlı test coverage)
3. **Test Çalıştırma** → `python -m pytest --cov=quickinsights tests/`
4. **Örnek Güncelleme** → `examples/` altında (yeni örnekler)
5. **Dokümantasyon** → README, docstring'ler ve tutorial'lar
6. **Versiyon Güncelleme** → `setup.py` ve `__init__.py`
7. **Performance Testing** → Benchmark suite çalıştırma
8. **Code Quality** → Black, flake8, mypy ile kod kalitesi

## 📋 Gereksinimler ✅ GÜNCELLENDİ

### Minimum Python Sürümü
- Python 3.8+

### Ana Bağımlılıklar
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
- scipy >= 1.7.0

### Performance Bağımlılıkları ✅ YENİ
- numba >= 0.56.0 (JIT compilation)
- dask[complete] >= 2022.1.0 (paralel işleme)

### GPU Desteği ✅ YENİ
- cupy-cuda11x >= 10.0.0 (GPU acceleration)

### Cloud Entegrasyonu ✅ YENİ
- boto3 >= 1.26.0 (AWS)
- azure-storage-blob >= 12.16.0 (Azure)
- google-cloud-storage >= 2.8.0 (Google Cloud)

### Profiling ve Monitoring ✅ YENİ
- psutil >= 5.9.0 (sistem kaynakları)

### Geliştirme Bağımlılıkları ✅ GÜNCELLENDİ
- pytest >= 6.0
- pytest-cov >= 2.0
- black >= 21.0 (kod formatı)
- flake8 >= 3.8 (kod kalitesi)
- mypy >= 0.800 (tip kontrolü)

## 🎯 Gelecek Geliştirmeler ✅ GÜNCELLENDİ

### ✅ Tamamlanan
- [x] Lazy evaluation sistemi
- [x] Paralel işleme (Dask)
- [x] GPU hızlandırma (CuPy)
- [x] Cloud entegrasyonu
- [x] Async/await desteği
- [x] Streaming analytics
- [x] Memory mapping
- [x] Performance profiling
- [x] Benchmark suite

### 🔄 Devam Eden
- [ ] Web arayüzü
- [ ] Daha fazla veri formatı desteği
- [ ] Çoklu dil desteği
- [ ] Machine learning entegrasyonu

### 📊 Yeni Özellikler
- [ ] Real-time analytics
- [ ] Automated insights
- [ ] Data quality scoring
- [ ] Interactive dashboards

---

**QuickInsights** - Veri analizi artık çok kolay! 🚀

*Son güncelleme: 2024*
