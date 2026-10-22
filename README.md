# Gram Altın Fiyat Aralığı Tahmini

Gram altının **1, 3 ve 5 gün sonra hangi fiyat aralığında olacağını** tahmin eden ve bu tahminlerin ne kadar tuttuğunu açıkça gösteren bir sistem.

> *English summary: A forecasting system for the Turkish gram gold price. It predicts 80% and 95% price ranges for 1, 3 and 5 trading days ahead. On unseen 2023–2026 data, its 95% ranges contained the actual price on 95–96% of days, while being 25–30% narrower than a naive baseline.*

**Selçuk Üniversitesi Bilgisayar Mühendisliği bitirme projesi** · Fatma Betül Pehlivan
📄 Ayrıntılı rapor: [docs/danisman_raporu.md](docs/danisman_raporu.md)

---

## Ne yapıyor?

Örnek bir çıktı:

> 20 Ekim 2026 kapanışı 6.560 TL.
> Bir sonraki işlem günü **%80 olasılıkla 6.490 – 6.675 TL**, **%95 olasılıkla 6.414 – 6.753 TL** arasında olacak.

Sistem fiyatın yönünü tahmin etmez; çünkü testlerde hiçbir model "yarın altın yükselecek mi?" sorusunda yazı-turadan iyi olmadı. Bunun yerine fiyatın ne kadar oynayacağını tahmin eder ve bunu bir aralığa çevirir. Tıpkı hava durumunun "yarın kesin yağmur" demek yerine "%70 olasılıkla yağmur" demesi gibi.

![Ana ekran](docs/figures/ana_ekran.png)

## Sonuçlar

16 model, 2012–2022 arasında yarıştı. Her ufuk için en iyi model seçildi ve **hiç görmediği 2023–2026 döneminde** sınandı:

| Tahmin | Kazanan model | "%95" dediği aralık tuttu | "%80" dediği aralık tuttu |
| --- | --- | --- | --- |
| 1 gün sonra | ARIMA-GARCH | **%94,7** | %77,3 |
| 3 gün sonra | ARIMA-GARCH + Chronos | **%96,2** | %84,3 |
| 5 gün sonra | ARIMA-GARCH + Chronos | **%96,2** | %84,0 |

Bu sistemin aralıkları, aynı güvenle, hiçbir şey öğrenmeyen basit yöntemin aralıklarından **%25–30 daha dar**. Fark istatistiksel olarak anlamlı (p < 0,01).

![Aralık genişliği karşılaştırması](docs/figures/yarisma.png)

**İşe yaramayanlar** (onları da raporluyorum): fiyatın yönünü tahmin etmek, XGBoost ile aralık tahmini, petrol ve dolar gibi dış piyasa verileri, Fed toplantı günleri ve finans için hazırlanmış Kronos modeli. Ayrıntılar raporda.

## Nasıl test edildi?

- **Model yalnızca geçmişi görür.** Her ayın başında, o güne kadarki son 3 yılın verisiyle yeniden kurulur ve o ayın günlerini tahmin eder.
- **En iyi model son sınavdan önce seçilir.** Seçim 2012–2022'de yapıldı; 2023–2026 yalnızca son sınav.
- **İleriye dönük karne.** Ekim 2026'dan beri sistem her sabah tahminini, gerçek fiyat belli olmadan önce değiştirilemez biçimde kaydediyor.

## Kurulum

Python 3.11 ve Git gerekir.

```powershell
git clone https://github.com/03betulpehlivan/turkey-gold-price-forecast.git
cd turkey-gold-price-forecast
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Chronos modeli isteğe bağlıdır (birkaç yüz MB indirir): `pip install chronos-forecasting`. Kurulu değilse sistem kendiliğinden ARIMA-GARCH'ı kullanır.

Fiziki altın verisi için ücretsiz bir TCMB EVDS anahtarı gerekir. Anahtar yalnızca ortam değişkeninde tutulur, dosyalara yazılmaz:

```powershell
setx EVDS_API_KEY "anahtarınız"
```

## Kullanım

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\pano.ps1   # web arayüzü: http://127.0.0.1:8000
python -m src.tool.yarisma --chronos                                   # model yarışması (10-15 dakika)
python -m src.tool.canli_karne karne                                   # ileriye dönük karne
python -m unittest discover -s tests                                   # bütün testler (690'dan fazla)
```

## Proje yapısı

```
src/tool/      tahmin araçları: model yarışması, ileriye dönük karne, pano servisi
src/web/       web arayüzü (ana ekran ve ayrıntılı araştırma panosu)
src/api/       REST API (FastAPI)
src/db/        Microsoft SQL Server / SQLite veritabanı katmanı
src/data/      veri indirme (TCMB EVDS, Yahoo Finance, Fed takvimi)
src/models/    ilk aşamanın modelleri ve sınavları
src/analysis/  ilk aşamanın araştırma çalışmaları
research/      sonuç dosyaları, tahmin kayıtları
scripts/       her gün kendiliğinden çalışan görevler
tests/         otomatik testler
docs/          rapor ve grafikler
```

## Teknik özet

- **Veri:** TCMB EVDS (fiziki gram altın, 1995'ten beri), Yahoo Finance (ons, kur, piyasa göstergeleri).
- **Modeller:** GARCH ailesi, ARIMA-GARCH, HAR, Markov rejim modeli, XGBoost, Amazon Chronos, Kronos ve birleşimleri.
- **Altyapı:** Python, Microsoft SQL Server, FastAPI, Windows Görev Zamanlayıcı, GitHub Actions.
- **Güvenilirlik:** her kod değişikliğinde otomatik testler; kayıtların değiştirilmediğini her gün kontrol eden bütünlük denetimi; görev çalışmazsa e-posta gönderen bekçi.

## Lisans

MIT. Ayrıntılar `LICENSE` dosyasında.

**Bu proje araştırma ve eğitim amaçlıdır. Gösterilen aralıklar yatırım tavsiyesi değil, istatistiksel risk aralığıdır.**
