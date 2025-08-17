# 🚀 QuickInsights

Tek komutla veri seti analizi yapan, kreatif ve yenilikçi Python kütüphanesi. NumPy ve Pandas gibi temel kütüphanelerin ötesine geçerek, büyük veri analizi için gelişmiş özellikler sunar.

## ✨ Özellikler

- 🔍 **Kapsamlı Veri Analizi**: Tek komutla veri seti analizi
- 📊 **Gelişmiş Görselleştirme**: Matplotlib, Seaborn ve Plotly entegrasyonu
- 🚀 **Performans Optimizasyonu**: Lazy evaluation, caching, parallel processing
- ☁️ **Cloud Entegrasyonu**: AWS S3, Azure Blob, Google Cloud Storage
- 🤖 **AI Destekli İçgörüler**: Otomatik pattern detection ve trend analizi
- 📈 **Real-time Pipeline**: Streaming data processing
- 🔧 **Modüler Yapı**: Kolay genişletilebilir ve özelleştirilebilir

## 🚀 Kurulum

### **Test PyPI'den Kurulum (Önerilen - Güncel Versiyon):**

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quickinsights
```

### **Geliştirici Kurulumu:**

```bash
git clone https://github.com/erena6466/quickinsights.git
cd quickinsights
pip install -e .
```

## 📖 Hızlı Başlangıç

```python
import quickinsights as qi
import pandas as pd

# Örnek veri seti
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [4, 5, 6, 7, 8],
    'C': ['a', 'b', 'a', 'b', 'a']
})

# Tek komutla kapsamlı analiz
result = qi.analyze(df, show_plots=True, save_plots=True)

# Veri seti bilgileri
info = qi.get_data_info(df)

# Aykırı değer tespiti
outliers = qi.detect_outliers(df)

# Performans optimizasyonu
optimized_df = qi.memory_optimize(df)
```

## 🔧 Gelişmiş Kullanım

### **AI Destekli Analiz:**
```python
from quickinsights.ai_insights import AIInsightEngine

ai_engine = AIInsightEngine(df)
insights = ai_engine.get_insights()
trends = ai_engine.predict_trends()
```

### **Cloud Entegrasyonu:**
```python
# AWS S3'e yükleme
qi.upload_to_cloud('data.csv', 'aws', 'my-bucket/data.csv', bucket_name='my-bucket')

# Cloud'dan veri işleme
result = qi.process_cloud_data('aws', 'my-bucket/data.csv', processor_func, bucket_name='my-bucket')
```

### **Real-time Pipeline:**
```python
from quickinsights.realtime_pipeline import RealTimePipeline

pipeline = RealTimePipeline()
pipeline.add_transformation(lambda x: x * 2)
pipeline.add_filter(lambda x: x > 10)
results = pipeline.process_stream(data_stream)
```

## 📚 Dokümantasyon

Detaylı API dokümantasyonu için [docs/api.md](docs/api.md) dosyasına bakın.

Komut listesi için [COMMANDS.md](COMMANDS.md) dosyasına bakın.

## 🤝 Katkıda Bulunma

Katkıda bulunmak için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🆘 Destek

- **GitHub Issues**: [https://github.com/erena6466/quickinsights/issues](https://github.com/erena6466/quickinsights/issues)
- **Dokümantasyon**: [docs/](docs/) klasörü
- **Örnekler**: [examples/](examples/) klasörü

## 🎯 Proje Durumu

- ✅ **Core Library**: Tamamlandı
- ✅ **Modular Architecture**: Tamamlandı
- ✅ **Test Suite**: %100 başarı oranı
- ✅ **Test PyPI**: Başarıyla yüklendi
- ⏳ **Main PyPI**: Ana PyPI'ye yükleme bekleniyor
- 🔄 **CI/CD**: GitHub Actions ile otomatik test
- 📚 **Documentation**: Kapsamlı dokümantasyon

## 🚀 Gelecek Planları

- [ ] Ana PyPI'ye yükleme
- [ ] ReadTheDocs entegrasyonu
- [ ] Community building
- [ ] Performance benchmarks
- [ ] Additional ML algorithms
- [ ] Web dashboard

---

**QuickInsights** - Veri analizini basitleştiren, performansı artıran Python kütüphanesi! 🚀📊
