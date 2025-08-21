"""
QuickInsights - Quick Start Example

Bu örnek yeni başlayanlar için QuickInsights kütüphanesinin 
temel kullanımını gösterir.
"""

import pandas as pd
import numpy as np
import quickinsights as qi

def main():
    print("🚀 QuickInsights - Quick Start Example")
    print("=" * 50)
    
    # 1. Örnek veri seti oluştur
    print("\n📊 1. Örnek veri seti oluşturuluyor...")
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'yaş': np.random.normal(35, 12, n_samples).astype(int),
        'maaş': np.random.normal(75000, 20000, n_samples),
        'deneyim_yılı': np.random.normal(8, 5, n_samples).astype(int),
        'şehir': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa'], n_samples),
        'eğitim': np.random.choice(['Lise', 'Üniversite', 'Yüksek Lisans', 'Doktora'], n_samples),
        'performans_skoru': np.random.normal(7.5, 1.5, n_samples)
    }
    
    # Bazı eksik veriler ekle
    data['maaş'][np.random.choice(n_samples, 50, replace=False)] = np.nan
    data['performans_skoru'][np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    df = pd.DataFrame(data)
    
    # Yaşları mantıklı aralığa getir
    df['yaş'] = np.clip(df['yaş'], 22, 65)
    df['deneyim_yılı'] = np.clip(df['deneyim_yılı'], 0, df['yaş'] - 22)
    df['maaş'] = np.clip(df['maaş'], 30000, 200000)
    df['performans_skoru'] = np.clip(df['performans_skoru'], 1, 10)
    
    print(f"✅ Veri seti oluşturuldu: {len(df)} satır, {len(df.columns)} sütun")
    
    # 2. Basit özet
    print("\n📋 2. Veri seti özeti:")
    print(qi.easy_summary(df))
    
    # 3. Hızlı analiz
    print("\n🔍 3. Hızlı analiz yapılıyor...")
    analysis_result = qi.easy_analyze(df, target='performans_skoru')
    
    print("\n🎯 Analiz Sonuçları:")
    print(f"   • Veri kalitesi: {analysis_result['veri_kalitesi']['seviye']}")
    print(f"   • Toplam bulgu: {len(analysis_result['oto_bulgular'])}")
    
    print("\n🔍 En önemli bulgular:")
    for i, bulgu in enumerate(analysis_result['oto_bulgular'][:3], 1):
        print(f"   {i}. {bulgu}")
    
    # 4. Veri temizleme
    print("\n🧹 4. Otomatik veri temizleme...")
    cleaning_result = qi.easy_clean(df, target='performans_skoru')
    
    clean_df = cleaning_result['temiz_veri']
    print(f"   • {cleaning_result['ozet']}")
    print(f"   • Yapılan işlemler: {len(cleaning_result['yapilan_islemler'])}")
    
    # 5. Dashboard oluşturma
    print("\n📊 5. İnteraktif dashboard oluşturuluyor...")
    
    try:
        dashboard_files = qi.create_dashboard(
            clean_df, 
            title="Çalışan Performans Analizi",
            output_html="employee_dashboard.html"
        )
        print(f"   ✅ Dashboard: {dashboard_files['html']}")
        print(f"   ✅ JSON rapor: {dashboard_files['json']}")
    except Exception as e:
        print(f"   ⚠️ Dashboard oluşturulamadı: {e}")
    
    # 6. Görselleştirme önerileri
    print("\n📈 6. Görselleştirme önerileri:")
    viz_result = qi.easy_visualize(clean_df, target='performans_skoru')
    
    for i, öneri in enumerate(viz_result['grafik_onerileri'][:3], 1):
        print(f"   {i}. {öneri}")
    
    print("\n💡 Örnek kod:")
    print(f"   {viz_result['kod_ornekleri'][0]}")
    
    # 7. Veriyi kaydet
    print("\n💾 7. Temizlenmiş veriyi kaydetme...")
    output_file = qi.easy_export(clean_df, "temizlenmis_veri", "csv")
    print(f"   ✅ Kaydedildi: {output_file}")
    
    print("\n🎉 Quick Start tamamlandı!")
    print("\n📚 Sonraki adımlar:")
    print("   • qi.quick_insight() ile detaylı analiz")
    print("   • qi.smart_clean() ile gelişmiş temizleme")
    print("   • qi.create_dashboard() ile rapor paylaşımı")
    
    return clean_df

if __name__ == "__main__":
    df = main()
