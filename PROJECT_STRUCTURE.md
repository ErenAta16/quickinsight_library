# QuickInsights Proje Yapısı 📁

Bu dosya, QuickInsights Python kütüphanesinin proje yapısını ve dosya organizasyonunu açıklar.

## 🏗️ Proje Dizin Yapısı

```
QuickInsights/
├── 📁 src/
│   └── 📁 quickinsights/
│       ├── 📄 __init__.py          # Ana kütüphane giriş noktası
│       ├── 📄 core.py              # Ana analiz fonksiyonları
│       ├── 📄 visualizer.py        # Görselleştirme modülü
│       └── 📄 utils.py             # Yardımcı fonksiyonlar
├── 📁 examples/
│   └── 📄 basic_usage.py          # Temel kullanım örneği
├── 📁 tests/
│   └── 📄 test_quickinsights.py   # Test dosyaları
├── 📄 README.md                    # Proje açıklaması
├── 📄 setup.py                     # Kurulum konfigürasyonu
├── 📄 requirements.txt             # Bağımlılıklar
└── 📄 PROJECT_STRUCTURE.md         # Bu dosya
```

## 📚 Modül Açıklamaları

### 🔧 `src/quickinsights/__init__.py`
- Kütüphanenin ana giriş noktası
- Tüm public fonksiyonları dışa aktarır
- Versiyon ve yazar bilgileri
- Import edilebilir ana fonksiyonlar

### 🎯 `src/quickinsights/core.py`
- **`analyze()`**: Ana analiz fonksiyonu
- **`analyze_numeric()`**: Sayısal değişken analizi
- **`analyze_categorical()`**: Kategorik değişken analizi
- Kapsamlı veri seti analizi ve raporlama

### 🎨 `src/quickinsights/visualizer.py`
- **`correlation_matrix()`**: Korelasyon matrisi görselleştirme
- **`distribution_plots()`**: Dağılım grafikleri
- **`summary_stats()`**: İstatistiksel özetler
- **`create_interactive_plots()`**: Plotly interaktif grafikler
- **`box_plots()`**: Kutu grafikleri

### 🛠️ `src/quickinsights/utils.py`
- **`get_data_info()`**: Veri seti bilgileri
- **`detect_outliers()`**: Aykırı değer tespiti
- **`get_correlation_strength()`**: Korelasyon gücü sınıflandırması
- **`format_number()`**: Sayı formatlama
- **`create_output_directory()`**: Çıktı dizini oluşturma
- **`validate_dataframe()`**: DataFrame doğrulama

## 🚀 Kurulum ve Kullanım

### Geliştirme Kurulumu
```bash
# Projeyi klonla
git clone <repository-url>
cd QuickInsights

# Geliştirme bağımlılıklarını kur
pip install -e .

# Testleri çalıştır
python -m pytest tests/
```

### Kullanıcı Kurulumu
```bash
# PyPI'dan kur
pip install quickinsights

# Veya geliştirme sürümünden
pip install git+<repository-url>
```

## 📖 Kullanım Örnekleri

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
# Sadece sayısal değişkenler
qi.analyze_numeric(df[['yas', 'maas']])

# Kategorik değişkenler
qi.analyze_categorical(df[['sehir', 'egitim']])

# Grafikleri kaydet
qi.analyze(df, save_plots=True, output_dir="./analiz_sonuclari")
```

## 🧪 Test Sistemi

### Test Çalıştırma
```bash
# Tüm testleri çalıştır
python -m pytest tests/

# Belirli test dosyası
python -m pytest tests/test_quickinsights.py

# Verbose mod
python -m pytest tests/ -v
```

### Test Kapsamı
- ✅ Veri doğrulama
- ✅ Aykırı değer tespiti
- ✅ İstatistiksel hesaplamalar
- ✅ Görselleştirme fonksiyonları
- ✅ Hata yönetimi
- ✅ Çıktı dosya işlemleri

## 📦 Dağıtım

### PyPI'ya Yükleme
```bash
# Build oluştur
python setup.py sdist bdist_wheel

# PyPI'ya yükle
twine upload dist/*
```

### Yerel Kurulum
```bash
# Geliştirme modunda kur
pip install -e .

# Normal kurulum
pip install .
```

## 🔄 Geliştirme Döngüsü

1. **Kod Yazma** → `src/quickinsights/` altında
2. **Test Yazma** → `tests/` altında
3. **Test Çalıştırma** → `python -m pytest`
4. **Örnek Güncelleme** → `examples/` altında
5. **Dokümantasyon** → README ve docstring'ler
6. **Versiyon Güncelleme** → `setup.py` ve `__init__.py`

## 📋 Gereksinimler

### Minimum Python Sürümü
- Python 3.8+

### Ana Bağımlılıklar
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
- scipy >= 1.7.0

### Geliştirme Bağımlılıkları
- pytest >= 6.0
- pytest-cov >= 2.0
- black >= 21.0
- flake8 >= 3.8
- mypy >= 0.800

## 🎯 Gelecek Geliştirmeler

- [ ] Daha fazla görselleştirme türü
- [ ] Makine öğrenmesi entegrasyonu
- [ ] Web arayüzü
- [ ] Daha fazla veri formatı desteği
- [ ] Performans optimizasyonları
- [ ] Çoklu dil desteği

---

**QuickInsights** - Veri analizi artık çok kolay! 🚀
