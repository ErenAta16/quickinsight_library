"""
QuickInsights - Advanced Analysis Example

Bu örnek QuickInsights kütüphanesinin gelişmiş özelliklerini gösterir.
İleri düzey veri analisti kullanımı için tasarlanmıştır.
"""

import pandas as pd
import numpy as np
import quickinsights as qi

def main():
    print("🚀 QuickInsights - Advanced Analysis Example")
    print("=" * 60)
    
    # 1. Karmaşık veri seti oluştur
    print("\n📊 1. Karmaşık veri seti oluşturuluyor...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # E-ticaret satış verisi simülasyonu
    data = {
        'customer_id': range(1, n_samples + 1),
        'order_date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_samples, p=[0.3, 0.25, 0.15, 0.2, 0.1]),
        'order_amount': np.random.lognormal(4, 1, n_samples),  # Log-normal distribution
        'customer_age': np.random.normal(40, 15, n_samples).astype(int),
        'customer_location': np.random.choice(['TR-34', 'TR-06', 'TR-35', 'TR-16', 'TR-07'], n_samples),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer', 'Cash'], n_samples),
        'shipping_cost': np.random.exponential(15, n_samples),
        'discount_applied': np.random.choice([0, 5, 10, 15, 20], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'customer_satisfaction': np.random.normal(4.2, 0.8, n_samples),
        'return_flag': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Veri tiplerini düzelt ve mantıklı aralıklara getir
    df['customer_age'] = np.clip(df['customer_age'], 18, 80)
    df['order_amount'] = np.clip(df['order_amount'], 10, 5000)
    df['customer_satisfaction'] = np.clip(df['customer_satisfaction'], 1, 5)
    df['shipping_cost'] = np.clip(df['shipping_cost'], 0, 100)
    
    # Kategorik sütunlara göre order_amount'u ayarla (realistik hale getir)
    electronics_mask = df['product_category'] == 'Electronics'
    df.loc[electronics_mask, 'order_amount'] *= 2  # Electronics daha pahalı
    
    clothing_mask = df['product_category'] == 'Clothing'
    df.loc[clothing_mask, 'order_amount'] *= 0.7  # Clothing daha ucuz
    
    # Eksik veriler ekle (gerçek dünya senaryosu)
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    df.loc[missing_indices[:200], 'customer_satisfaction'] = np.nan
    df.loc[missing_indices[200:400], 'shipping_cost'] = np.nan
    df.loc[missing_indices[400:500], 'customer_age'] = np.nan
    
    print(f"✅ Karmaşık veri seti oluşturuldu: {len(df)} satır, {len(df.columns)} sütun")
    
    # 2. Comprehensive Quick Insight Analysis
    print("\n🔍 2. Kapsamlı hızlı analiz...")
    
    insight_result = qi.quick_insight(
        df, 
        target_column='customer_satisfaction',
        sample_size=2000,  # Büyük veri seti için sampling
        include_viz=False
    )
    
    print("\n📊 Executive Summary:")
    print(insight_result['executive_summary'])
    
    print("\n🎯 Hedef Değişken Analizi:")
    if 'target_analysis' in insight_result:
        target_info = insight_result['target_analysis']
        print(f"   • Tip: {target_info['variable_type']}")
        print(f"   • Eksik oran: {target_info['missing_percentage']}%")
        if 'statistics' in target_info:
            stats = target_info['statistics']
            print(f"   • Ortalama: {stats['mean']:.2f}")
            print(f"   • Medyan: {stats['median']:.2f}")
            print(f"   • Standart sapma: {stats['std']:.2f}")
    
    # 3. Advanced Data Quality Analysis
    print("\n🏆 3. Gelişmiş veri kalitesi analizi...")
    
    quality_result = qi.analyze_data_quality(df)
    
    print(f"   • Genel kalite skoru: {100 - quality_result['missing_data']['missing_percentage']:.1f}/100")
    print(f"   • Toplam eksik veri: {quality_result['missing_data']['total_missing']:,}")
    print(f"   • Kopya satır: {quality_result['duplicates']['duplicate_rows']}")
    
    if quality_result['recommendations']:
        print("\n   🔧 Ana öneriler:")
        for rec in quality_result['recommendations'][:3]:
            print(f"     - {rec}")
    
    # 4. Smart Cleaning with Advanced Options
    print("\n🧹 4. Gelişmiş otomatik temizleme...")
    
    cleaner = qi.SmartCleaner(df, target_column='customer_satisfaction')
    cleaning_result = cleaner.auto_clean(aggressive=False, preserve_original=True)
    
    print(f"   • Orijinal boyut: {cleaning_result['original_shape']}")
    print(f"   • Temizlenmiş boyut: {cleaning_result['cleaned_shape']}")
    
    quality_improvement = cleaning_result['quality_improvement']
    print(f"   • Eksik veri azalması: {quality_improvement['missing_data_reduction_pct']:.1f}%")
    print(f"   • Bellek tasarrufu: {quality_improvement['memory_reduction_pct']:.1f}%")
    
    cleaned_df = cleaning_result['cleaned_data']
    
    # 5. Advanced Feature Engineering
    print("\n🔧 5. Gelişmiş feature engineering...")
    
    # Tarih-based features
    cleaned_df['order_month'] = cleaned_df['order_date'].dt.month
    cleaned_df['order_hour'] = cleaned_df['order_date'].dt.hour
    cleaned_df['order_dayofweek'] = cleaned_df['order_date'].dt.dayofweek
    
    # Business logic features
    cleaned_df['profit_margin'] = cleaned_df['order_amount'] - cleaned_df['shipping_cost']
    cleaned_df['effective_amount'] = cleaned_df['order_amount'] * (1 - cleaned_df['discount_applied']/100)
    cleaned_df['age_group'] = pd.cut(cleaned_df['customer_age'], 
                                   bins=[0, 25, 35, 50, 100], 
                                   labels=['Young', 'Adult', 'Middle', 'Senior'])
    
    # Category encoding for high-cardinality features
    category_features = ['product_category', 'customer_location', 'payment_method']
    for col in category_features:
        if col in cleaned_df.columns:
            cleaned_df[f'{col}_encoded'] = cleaned_df[col].astype('category').cat.codes
    
    print(f"   ✅ {len(category_features)} kategorik değişken encode edildi")
    print(f"   ✅ 6 yeni özellik oluşturuldu")
    print(f"   ✅ Son boyut: {cleaned_df.shape}")
    
    # 6. Advanced Analytics
    print("\n📈 6. Gelişmiş analitik insights...")
    
    # Correlation analysis
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 2:
        corr_matrix = cleaned_df[numeric_cols].corr()
        
        # Find strongest correlations with target
        target_corr = corr_matrix['customer_satisfaction'].abs().sort_values(ascending=False)
        print(f"\n   🔗 En güçlü korelasyonlar (customer_satisfaction ile):")
        for col, corr_val in target_corr.head(5).items():
            if col != 'customer_satisfaction':
                print(f"     • {col}: {corr_val:.3f}")
    
    # Category analysis
    print(f"\n   📊 Kategori bazlı analizler:")
    for category in ['product_category', 'payment_method']:
        if category in cleaned_df.columns:
            category_stats = cleaned_df.groupby(category)['customer_satisfaction'].agg(['mean', 'count']).round(2)
            print(f"\n     {category} bazında memnuniyet:")
            for idx, row in category_stats.head(3).iterrows():
                print(f"       • {idx}: {row['mean']} (n={row['count']})")
    
    # 7. Predictive Power Assessment
    print("\n🎯 7. Tahmin gücü değerlendirmesi...")
    
    if 'predictive_power' in insight_result:
        pred_power = insight_result['predictive_power']
        print(f"   • Toplam özellik sayısı: {pred_power['feature_count']}")
        print(f"   • Sayısal özellikler: {pred_power['numeric_features']}")
        print(f"   • Kategorik özellikler: {pred_power['categorical_features']}")
        
        if 'top_correlations' in pred_power and pred_power['top_correlations']:
            print(f"\n   🏆 En güçlü korelasyonlar:")
            for col, abs_corr, actual_corr in pred_power['top_correlations'][:3]:
                print(f"     • {col}: {actual_corr:.3f}")
    
    # 8. Advanced Dashboard Generation
    print("\n📊 8. Gelişmiş dashboard oluşturuluyor...")
    
    try:
        dashboard_files = qi.create_dashboard(
            cleaned_df,
            title="E-Ticaret Müşteri Memnuniyeti Analizi",
            output_html="advanced_ecommerce_dashboard.html",
            output_json="advanced_analysis_data.json"
        )
        
        print(f"   ✅ İnteraktif Dashboard: {dashboard_files['html']}")
        print(f"   ✅ Veri raporu: {dashboard_files['json']}")
        
    except Exception as e:
        print(f"   ⚠️ Dashboard oluşturulamadı: {e}")
    
    # 9. Export Advanced Results
    print("\n💾 9. Gelişmiş sonuçları export etme...")
    
    # Export cleaned data
    qi.easy_export(cleaned_df, "advanced_cleaned_ecommerce_data", "csv")
    
    # Export analysis summary
    analysis_summary = {
        'dataset_info': insight_result['dataset_info'],
        'data_quality': quality_result,
        'cleaning_summary': cleaning_result,
        'key_insights': insight_result['auto_insights'],
        'recommendations': insight_result['recommendations']
    }
    
    import json
    with open('advanced_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print("   ✅ Temizlenmiş veri: advanced_cleaned_ecommerce_data.csv")
    print("   ✅ Analiz özeti: advanced_analysis_summary.json")
    
    # 10. Performance Benchmarking
    print("\n⚡ 10. Performans analizi...")
    
    # Memory optimization
    optimized_df = qi.optimize_for_speed(cleaned_df)
    
    original_memory = cleaned_df.memory_usage(deep=True).sum() / 1024**2
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
    memory_saving = ((original_memory - optimized_memory) / original_memory) * 100
    
    print(f"   • Orijinal bellek: {original_memory:.1f} MB")
    print(f"   • Optimize bellek: {optimized_memory:.1f} MB")
    print(f"   • Tasarruf: {memory_saving:.1f}%")
    
    print("\n🎉 Gelişmiş analiz tamamlandı!")
    print("\n🚀 Sonraki seviye öneriler:")
    print("   • Machine Learning pipeline ile model oluşturma")
    print("   • Real-time streaming analysis")
    print("   • Advanced visualization ile deep insights")
    print("   • A/B testing framework entegrasyonu")
    
    return {
        'original_data': df,
        'cleaned_data': cleaned_df,
        'optimized_data': optimized_df,
        'analysis_results': insight_result,
        'quality_assessment': quality_result
    }

if __name__ == "__main__":
    results = main()
