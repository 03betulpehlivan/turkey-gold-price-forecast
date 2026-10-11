# Veri Platformu (Adım 1)

Bu belge, çok kaynaklı günlük veri toplama ve piyasa paneli adımını açıklar.
Mevcut model hattı (`build_dataset.py` → `model_dataset.csv` → `train.py`) bu adımdan etkilenmez.

## Komutlar (proje kökünden, PowerShell)

```powershell
python -m src.data.collect                 # 10 seri, 2015-01-01 .. dün; yeni anlık görüntü
python -m src.data.build_panel             # en yeni görüntüden panel; yalnızca rapor
python -m src.data.build_panel --write     # data/processed/market_panel.csv + rapor
python -m unittest tests.test_data_platform -v
```

## Seriler

Sembol ve açıklamalar genel piyasa bilgisidir; bu kod yazılırken Yahoo'ya erişilemediği için
**hiçbiri doğrulanmadı**. İlk `collect` çalıştırmasının `manifest.json` dosyası her serinin
gerçek durumunu, satır sayısını ve tarih aralığını kayda geçirir.

| Ad | Yahoo sembolü | Açıklama | Zorunlu |
| --- | --- | --- | --- |
| gold | GC=F | COMEX altın vadeli, sürekli en yakın kontrat (USD/ons); spot değil | Evet |
| usdtry | USDTRY=X | USD/TRY kuru | Evet |
| silver | SI=F | COMEX gümüş vadeli (USD/ons) | Hayır |
| dxy | DX-Y.NYB | ICE ABD dolar endeksi | Hayır |
| us10y | ^TNX | ABD 10 yıllık getiri endeksi; birim doğrulanmalı | Hayır |
| vix | ^VIX | S&P 500 oynaklık endeksi | Hayır |
| sp500 | ^GSPC | S&P 500 | Hayır |
| bist100 | XU100.IS | BIST 100 (TRY) | Hayır |
| brent | BZ=F | Brent petrol vadeli (USD/varil) | Hayır |
| eurtry | EURTRY=X | EUR/TRY kuru | Hayır |
| gvz | ^GVZ | CBOE altın ETF oynaklık endeksi (opsiyon zımni oynaklığı, yıllık %); kapanış saati altından sonra olabilir | Hayır |

## Dosya düzeni

```
data/raw/snapshots/<YYYYMMDDTHHMMSSZ>/   # her collect çalıştırması yeni klasör; asla üzerine yazılmaz
    manifest.json                       # indirme zamanı, istek, yfinance sürümü, seri durumları, md5
    gold.csv, usdtry.csv, ...           # yfinance çıktısı (auto_adjust=False), mevcut ham dosyalarla aynı biçim
data/processed/market_panel.csv          # build_panel --write
data/processed/market_panel_report.json  # kapsam, bayatlık, mevcut veriyle karşılaştırma
```

`data/raw/gold_futures.csv` ve `data/raw/usdtry.csv` (Faz A/B girdileri) değişmez.

## Panel şeması ve zaman kuralları

- **Temel takvim:** altın ve USD/TRY'nin ortak günleri. Teorik fiyat `build_dataset.build` ile, mevcut veriyle bit düzeyinde aynı formül ve işlem sırasıyla hesaplanır.
- **Ek seriler (as-of):** her panel günü için o gün veya öncesindeki **en son** kapanış. Panel günü sonrasındaki hiçbir değer kullanılmaz (kesme değişmezliği testiyle sınanır).
- **`<seri>_age_days`:** kullanılan değerin panel gününden kaç takvim günü eski olduğu.
- **Bayatlık sınırı:** 7 takvim günü. Daha eski değerler boş bırakılır ve raporlanır; bayram gibi uzun tatillerde eski değer taşınmaz.
- **Varsayım (doğrulanmadı):** bir güne ait kapanışlar, o gün için yapılan tahminde bilinir; tahmin tüm piyasalar kapandıktan sonra (yaklaşık gece yarısı TSİ) üretilir. Yahoo'nun günlük çubuk saatleri belgelenmemiştir; bu varsayım tanı adımında sınanacak.

## Bilinen sınırlamalar

- Vadeli seriler (altın, gümüş, Brent) kontrat değişim günlerinde yapay sıçrama içerebilir.
- Yahoo geçmiş değerleri düzeltebilir. Panel raporu, yeni indirilen altın/kur değerlerini mevcut `theoretical_gold.csv` ile karşılaştırır ve farkları gösterir. Bu bir hata değil, kaynağın değiştiğinin bilgisidir.
- Panel henüz modele bağlı değildir. Ek seriler, ayrı ve önceden kayda geçirilmiş bir özellik deneyiyle değerlendirilecek.
