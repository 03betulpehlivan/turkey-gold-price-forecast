# Durum Notu (22 Ekim 2026)

Ayrıntılı rapor: [danisman_raporu.md](danisman_raporu.md)

## Şu an çalışan

- **Ana ekran:** 1, 3 ve 5 gün sonrası için %80 ve %95 fiyat aralıkları, hareketli gün uyarısı, başarı karnesi (`scripts\pano.ps1`).
- **Model yarışması:** 16 model; kazananlar 1 günde ARIMA-GARCH, 3 ve 5 günde ARIMA-GARCH + Chronos.
- **İleriye dönük karne:** her sabah tahminler, gerçek fiyat belli olmadan önce kaydediliyor.
- **Otomatik görevler:** hafta içi her gün sabah 12:30 ve akşam 22:30; veritabanı, yedek, GitHub eşitlemesi, bekçi.

## Son denemeler

- Kronos: 1 günde ARIMA-GARCH ile aynı, 3 ve 5 günde daha kötü.
- Fed toplantı günleri (olay takvimi): ek bilgi katmadı.

## Sırada

- İleriye dönük karnenin dolması (birkaç ay).
- TCMB faiz kararlarıyla olay takvimi.
