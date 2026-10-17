# Durum Notu (17 Ekim 2026)

Ayrıntılı ara rapor: [danisman_raporu.md](danisman_raporu.md). Bu not yalnızca güncel durumu özetler.

## Çalışan sistem

- **Canlı protokoller (15.10.2026'dan beri, ön kayıtlı):** `LIVE_PHYS_V1` ve `LIVE_PHYS_INT_V1` (sabah fiziki tahmin ve aralıklar), `LIVE_V1` (akşam teorik tahmin, vadeli ons) ve `LIVE_V2` (aynı model, ons kaynağı GLD). Değerlendirme 120 tahminden sonra bir kez yapılacak (~Nisan 2027).
- **Otomasyon:** hafta içi 12:30 (yedek 13:30) ve 22:30; her çalışmanın sonunda veritabanı yüklemesi, yedek, GitHub eşitlemesi ve sistem durumu.
- **Sistem katmanı:** SQL Server (LocalDB) aynası, salt okunur FastAPI ve web panosu (`scripts\pano.ps1`).
- **Koruma:** her push'ta ve her gün bütünlük denetimi (`python -m src.audit`), GitHub'da bekçi (görev çalışmazsa e-posta), korumalı `main` dalı, SHA ile sabit eylemler, haftalık bağımlılık taraması.
- **Yeniden üretim:** `python -m src.repro`, bütün keşif çalışmalarını ve geriye dönük sınavı baştan hesaplayıp kayıtlı sonuçlarla karşılaştırır.

## Tamamlanan çalışmalar

- Bağımsız sınavlar: `FINAL_EXAM_V1`, `FINAL_EXAM_PHYS_V1`, `BACK_EXAM_ONS_V1` (hiçbirinde model RW'yi geçemedi).
- Keşif çalışmaları: sızıntı gösterimi ve LSTM, oynaklık, TL oynaklığı bileşen modeli, haftalık/aylık trend ve makro göstergeler, fiziki prim, güç analizi.

## Takvim

- **Kasım 2026 sonu:** vadeli kontrat değişimi; `LIVE_V1` ve `LIVE_V2` karşılaştırması.
- **~Nisan 2027:** canlı protokollerin tek seferlik değerlendirmesi.
