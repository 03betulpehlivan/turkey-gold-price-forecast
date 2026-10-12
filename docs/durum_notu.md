# Durum Notu (12 Ekim 2026)

## Tamamlanan
- **FINAL_EXAM_V1 (2026, bağımsız, tek seferlik):** nokta tahmini "yarın = bugün" ile **pratikte eşdeğer** (MAPE ~%1,35); aralık tahmininde A1 ve A3 geçti, **A2 (Christoffersen %80) geçmedi**, yani taşmalar kümelendi. Sonuçlar `research/final_exam_v1/` altında.
- **Zamanlama bulgusu:** Yahoo günlük serisinde altın ~18:00 UTC'ye, USD/TRY ~02–05 UTC'ye ait (`fx_close_time`). Eski "teorik gram" bu yüzden ~13–16 saat farklı anların çarpımı.
- **Eşzamanlı seri (LIVE_V1 altyapısı):** saatlik arşiv ve tam 14:00 NY kuralı; 498 gün (2024-10-11 – 2026-10-09). `-TahminYok` modu gerçek veride doğrulandı; ana panel ve sınav dosyaları değişmedi.

## Açık sorun: vadeli kontrat değişimi
- GC=F (sürekli vadeli) saatlik ve günlük serilerde kontratı **farklı günlerde** değiştiriyor. Örnekler: 2025-07-30 / 08-01 ve 2026-07-29 / 07-31; sıçrama ~%1,5–1,9.
- Eşzamanlı seride sahte getiriler oluşuyor; vadeli fiyat ayrıca spottan taşıma maliyeti kadar yüksek.
- 2026-01-30 (~−%9–11'lik çok büyük hareket) belirsiz: aşırı oynaklık mı, kontrat değişimi mi, ayırt edilemedi.
- **Canlı tahmin bu çözülene kadar BEKLETİLİYOR.** Arşiv her gün `-TahminYok` ile toplanmaya devam etmeli.

## Sıradaki adım
1. `python -m src.analysis.spot_probe`: XAUUSD=X, GLD ve IAU taraması (yalnızca teşhis). Ön kayıtlı kural: kontrat günlerinde z ≥ 3 ve korelasyon > 0,95.
2. Uygun aday çıkarsa eşzamanlı seriyi spot kaynakla yeniden tanımlamak (yeni sürüm, testlerle). Çıkmazsa kontrat günlerini işaretleyip dışlamak.
3. Ardından ilk canlı tahmin (TSİ 21:00 sonrası, aynı NY günü içinde).
