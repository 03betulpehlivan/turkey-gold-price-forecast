# Durum Notu (15 Ekim 2026)

Ayrıntılı ara rapor: [danisman_raporu.md](danisman_raporu.md). Bu not yalnızca güncel durumu özetler.

## Tamamlanan

- **Bağımsız sınavlar (tek seferlik, tekrar çalıştırılmaz):**
  - `FINAL_EXAM_V1` (teorik, 2026): nokta tahmini "yarın = bugün" ile **pratikte eşdeğer**. Aralıklarda Christoffersen %80 koşulu **sağlanmadı**.
  - `FINAL_EXAM_PHYS_V1` (fiziki, 2023-01 – 2026-10): Boşluk modeli RW'den **%9,1 daha kötü** (p = 0,03).
- **Canlı protokoller (ön kayıtlı, 15.10.2026'dan beri):**
  - `LIVE_PHYS_V1` ve `LIVE_PHYS_INT_V1`: sabah fiziki tahmin ve aralıklar.
  - `LIVE_V1`: akşam eşanlı teorik tahmin. Değerlendirme kuralı, ilk sonuç görülmeden önce `research/preregistration/live_v1.json` ile donduruldu (`src/live/forecast_eval.py`). Aynı ön kayıtla `forecast.py`, `sync_dataset.py`, `adaptive.py`, `intervals.py`, `implied_vol.py` ve `train.py` kilitlendi. Bunlardan biri değişirse sonraki teorik tahminler değerlendirmeye girmez.
- **Otomasyon:** Windows Görev Zamanlayıcı ile hafta içi 12:30 / 13:30 (sabah) ve 22:30 (akşam).
- **Veri yedeği:** Her zamanlanmış çalışmanın sonunda `data\raw`, `data\processed` ve `logs` klasörleri `ALTIN_YEDEK` hedefine kopyalanır (`scripts/yedekle.ps1`). Hedefte hiçbir şey silinmez.
- **Sürekli test:** GitHub Actions her kod değişikliğinde bütün birim testlerini Ubuntu'da çalıştırır.

## Açık sorun: vadeli kontrat değişimi (teorik seri)

- GC=F kontrat değişim günlerinde %1,5–1,9'luk sahte sıçrama gösteriyor (örneğin 2025-07-30 ve 2026-07-29).
- Bir sonraki değişim Kasım sonunda bekleniyor. `LIVE_V1` ön kaydı bu günleri yalnızca ikincil duyarlılık analizinde dışlıyor; birincil sonuç bütün günleri kapsıyor.
- **Karar adımı:** `python -m src.analysis.spot_probe`. Spot kaynağa geçilirse yeni protokol kimliği gerekir (`LIVE_V2`); `LIVE_V1` kodu değişmez.

## Sıradaki adımlar

1. Spot ons taraması (Kasım sonundan önce).
2. 2001–2004 geriye dönük bağımsız sınav ("yalnız ons" modeli; daha önce hiçbir seçimde kullanılmamış dönem).
3. SQL Server, REST API ve web arayüzü.
4. Haftalık/aylık trend, makro göstergeler, sızıntı gösterimi ve LSTM (keşif).
5. Nisan 2027: canlı protokollerin tek seferlik değerlendirmesi.
