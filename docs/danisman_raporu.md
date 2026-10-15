# Türkiye'de Gram Altın Fiyatlarının Makine Öğrenmesi Tabanlı Tahmini ve Analiz Sistemi

**Ara rapor ve pilot gösterim planı — 15 Ekim 2026**

Fatma Betül Pehlivan · Selçuk Üniversitesi Bilgisayar Mühendisliği · github.com/03betulpehlivan/turkey-gold-price-forecast

---

## Özet

Bu proje, Türkiye'deki **fiziki** gram altın fiyatı ile ons altın ve USD/TRY'den hesaplanan **teorik** gram fiyatı için kısa vadeli tahmin üretir, tahminlerin ne kadar güvenilir olduğunu ölçer ve sonuçları sızıntısız, önceden kayıtlı testlerle değerlendirir.

Ana sonuçlar:

1. **Günlük fiyat seviyesini basit "yarın = bugün" yönteminden (Random Walk, RW) daha iyi tahmin etmek zordur.** Teorik fiyatta 2026 bağımsız sınavında model RW ile pratikte eşdeğer çıktı. Fiziki fiyatta ise geliştirme döneminde güçlü görünen model, 2023–2026 bağımsız sınavında RW'den daha kötü sonuç verdi.
2. **Fiziki fiyat için sabah tahmininde gerçek bir bilgi kaynağı bulundu.** Borsa İstanbul kapandıktan sonra gece boyunca ons ve kurda olan hareket, ertesi günün fiziki kapanışına büyük ölçüde yansıyor. 2012–2022 geliştirme döneminde yalnızca ons bilgisi, veri zamanlaması varsayımından bağımsız olarak RW'ye göre %5 daha az hata sağladı (11 yılın 11'inde). Kur bilgisiyle birlikte kazanç %22'ye çıkıyor.
3. **Yaygın kullanılan veri kaynağında ciddi zamanlama sorunları tespit edildi.** Yahoo Finance günlük verisinde ons ve kur 13–16 saat farklı anlara ait. 2005–2010 kur verisi ise sabah tahmini için geleceği sızdırıyor. Vadeli ons serisinde kontrat değişimlerinden kaynaklanan sahte sıçramalar var.
4. **Oynaklığa duyarlı tahmin aralıkları** geliştirme döneminde iyi kalibre edildi (%80 aralıkta kapsam 0,800, kümelenme testi geçti).
5. **Canlı sistem kuruldu ve ilk gerçek tahminini üretti.** 15 Ekim 2026'da ilk canlı tahmin ve tahmin aralıkları değiştirilemez kayda geçti. Sistem, her tahmini ertesi gün gerçekleşen fiyatla puanlayacak şekilde tasarlandı ve hafta içi otomatik çalışacak biçimde zamanlandı. Önceden kaydedilmiş değerlendirme, 120 tahmin dolunca (yaklaşık Nisan 2027) bir kez yapılacak.

---

## 1. Araştırma problemi

Türkiye'de gram altın hem yatırım hem birikim aracıdır. Fiyatı esas olarak uluslararası ons fiyatı ve USD/TRY kuruna bağlıdır. Ancak yurt içi fiziki piyasa, zaman zaman uluslararası fiyattan ayrışır.

Araştırma soruları:

- **S1.** Günlük gram altın fiyatı, basit referans yöntemlerden daha isabetli tahmin edilebilir mi?
- **S2.** Hangi bilgi kaynakları (piyasa göstergeleri, gece fiyat hareketleri, yurt içi prim) tahmine gerçek bir katkı sağlar?
- **S3.** Tahminin belirsizliği (fiyatın hangi aralıkta olacağı) güvenilir biçimde ölçülebilir mi?
- **S4.** Sonuçlar, geliştirme sırasında hiç görülmemiş veride ve canlı ortamda da geçerli mi?

## 2. Teorik fiyat ile fiziki fiyat

| | Teorik gram fiyatı | Fiziki gram fiyatı |
| --- | --- | --- |
| Tanım | Ons (USD) × USD/TRY ÷ 31,1035 | Borsa İstanbul Kıymetli Madenler Piyasası günlük kapanışı (TL/kg ÷ 1.000) |
| Kaynak | Yahoo Finance | TCMB EVDS |
| Niteliği | Uluslararası fiyatın TL karşılığı | Türkiye organize piyasasında gerçekleşen **toptan** fiyat (kuyumcu fiyatı değil) |

İki seri projede **ayrı değişkenler** olarak tutulur. Aralarındaki fark (prim) bir bulgu olarak incelenir.

**Fiziki altın primi** (fiziki ÷ teorik − 1, yıllık medyan):

| 2015–2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- |
| −%0,1 ile +%0,2 arası | **+%1,17** | **+%1,38** | +%0,45 | ~%0 |

2023–2024'te fiziki altın, uluslararası fiyatın belirgin biçimde üzerinde işlem gördü; bazı günlerde prim %5'i aştı. Bu dönemde altın ithalatına getirilen kısıtlamalar olası bir açıklamadır; bu proje kapsamında doğrulanmamıştır.

## 3. Veri

| Veri | Kaynak | Dönem | Miktar |
| --- | --- | --- | --- |
| Fiziki altın kapanışı ve ağırlıklı ortalama (TL), Borsa İstanbul USD/ons | EVDS: İstanbul Altın Borsası arşivi (1995–2018) ve Borsa İstanbul (2018–) | 1995-07 → bugün | 7.039 gün |
| Ons (GC=F), USD/TRY ve 9 piyasa göstergesi (gümüş, dolar endeksi, ABD 10 yıllık faizi, VIX, S&P 500, BIST 100, Brent, EUR/TRY, altın oynaklık endeksi GVZ) | Yahoo Finance, günlük | 2005 → bugün | 5.461 gün |
| Ons ve USD/TRY, saatlik | Yahoo Finance | 2024-10 → bugün | Her gün arşivleniyor |

**Veri kalite kontrolleri:**

- **Fiziki serinin birleşme noktası:** 29.06.2018 ile 02.07.2018 arasında +%0,66; olağan bir hafta sonu hareketi.
- **"İşlem yok" günleri:** 7 günde TL fiyatı 0 olarak kayıtlı (çoğu arife günü). Fiyat sayılmadı, boş gün kabul edildi.
- **%15'ten büyük günlük hareketler:** 23.02.2001, 17.09.2001, 21.12.2021. Üçü de gerçek piyasa olayları.
- **EVDS'nin 1.000 satır sınırı:** bu sınırın veriyi sessizce kestiği tespit edildi. İndirme, sınıra ulaşan parçaları otomatik bölecek şekilde düzeltildi.
- **Değiştirilemez anlık görüntüler:** her indirme md5 özetiyle saklanıyor.

## 4. Projeye özgün veri bulguları

**4.1 Yahoo günlük verisinde zamanlama uyumsuzluğu.** Saatlik veriyle yapılan analiz, Yahoo'nun günlük ons kapanışının ~18:00 UTC'ye (COMEX uzlaşma saati), USD/TRY kapanışının ise ~02:00–05:00 UTC'ye ait olduğunu gösterdi. Yaygın kullanılan "teorik gram fiyatı" bu nedenle **13–16 saat farklı anların çarpımıdır.** Proje, iki değeri aynı saatlik çubuktan alan eşzamanlı bir seri kurdu.

**4.2 2005–2010 kur verisinde sızıntı.** Borsa İstanbul'un aynı andaki TL ve USD kapanışlarından türetilen örtük kurla yapılan testte, Yahoo'nun 2005–2010 günlük kur verisinin Borsa İstanbul kapanışından **sonraki** bir saate ait olduğu bulundu (ertesi gün korelasyonu 0,25; bant 0,06). Bu dönem verisiyle yapılan sabah tahminleri geleceği görür; bu yüzden eğitimden çıkarıldı. 2011–2015'te kur verisi kapanıştan önceye ait; 2016–2022 için test kararsız.

**4.3 Vadeli kontrat sıçramaları.** Sürekli vadeli ons serisi (GC=F), kontrat değişim günlerinde saatlik ve günlük veride **farklı günlerde** %1,5–1,9'luk sahte sıçramalar yapıyor (örneğin 30.07.2025 / 01.08.2025 ve 29.07.2026 / 31.07.2026).

## 5. Yöntem

- **Zamana göre ilerleyen değerlendirme (walk-forward):** genişleyen eğitim penceresi, her 63 işlem gününde yeniden eğitim, eğitim ile test arasında 1 günlük boşluk. Rastgele bölme hiç kullanılmadı.
- **Sızıntı kontrolleri:** her girdi için açık zaman damgası; tahmin anında bilinmeyen verinin kullanılmadığını doğrulayan otomatik testler.
- **Ön kayıt:** her kesin test için hipotez, karar kuralı, kod ve veri özetleri sonuçlardan önce yazılıp Git'e kaydedildi. Kesin testler **yalnızca bir kez** çalıştırılabiliyor; her deneme `research/registry.jsonl` dosyasına kaydediliyor.
- **İstatistik:**
  - Nokta tahminleri: Diebold–Mariano testi (küçük örneklem düzeltmesiyle), çoklu karşılaştırma için Holm düzeltmesi, eşleştirilmiş blok bootstrap güven aralıkları.
  - Tahmin aralıkları: Kupiec ve Christoffersen testleri, Winkler skoru.
- **Yazılım kalitesi:** 429 otomatik test, tamamı geçiyor (tam paket, 15.10.2026). Testlerin gerçekten hata yakaladığı, koda bilinçli hatalar sokularak (mutasyon testi) doğrulandı. Testler sentetik veriyle çalışır; piyasa başarısının kanıtı değildir.

**Modeller.** Random Walk, Mean-Return, Linear Regression, Random Forest, XGBoost, faktör modelleri, GARCH, GVZ ile bileşen oynaklığı, uyarlanabilir konformal çıkarım (ACI). Fiziki fiyat için "boşluk" modelleri: sabah teorik fiyat ile dünkü fiziki fiyat arasındaki farka dayanıyor.

## 6. Sonuçlar

Her sonucun türü belirtilmiştir:

- **Keşif:** geliştirme verisiyle yapılan çalışma; model seçimi burada yapılır, kanıt değeri sınırlıdır.
- **Bağımsız sınav:** önceden kaydedilmiş, tek seferlik, hiç görülmemiş veri.
- **Canlı:** ileriye dönük kayıt.

### 6.1 Teorik fiyat, ertesi gün tahmini

| Değerlendirme | Tür | Sonuç |
| --- | --- | --- |
| 2018–2023, 6 model ve 8 faktör grubu | Keşif | Hiçbir model RW'yi anlamlı biçimde geçemedi. RW hatası %1,07 |
| 2026, 194 tahmin (`FINAL_EXAM_V1`) | **Bağımsız sınav** | Mean-Return %1,345, RW %1,347. **Pratikte eşdeğer** (fark −%0,21 [−1,34, +1,17]) |

### 6.2 Teorik fiyat, tahmin aralıkları

| Değerlendirme | Tür | Sonuç |
| --- | --- | --- |
| 2023/10–2025/12 teyit testi | **Bağımsız** | %80 ve %95 kapsam 0,788 / 0,940. **Hipotez desteklendi** |
| 2026 | **Bağımsız sınav** | Kapsam 0,794 / 0,938, ama %80 aralıkta hatalar kümelendi (Christoffersen p ≈ 0,05). **Desteklenmedi** |

### 6.3 Fiziki fiyat, sabah tahmini

**Kurulum.** Hedef, bugünün fiziki kapanışı. Tahmin anında bilinenler: dünkü fiziki kapanış, dün akşamki ons kapanışı ve bu sabahın kuru.

| Değerlendirme | Tür | Sonuç |
| --- | --- | --- |
| 2016–2022, 1.741 tahmin | Keşif | Boşluk modeli RW'den **%22,8** daha az hata; yön isabeti %71 |
| Hata analizi (2016–2022) | Keşif | Model iyi kalibre (eğim 1,10 [0,91, 1,28]). Kontrat günleri, prim seviyesi ve aşırı günler başarıyı bozmuyor |
| **2023–2026, 874 tahmin (`FINAL_EXAM_PHYS_V1`)** | **Bağımsız sınav** | Boşluk modeli **RW'den %9,1 daha kötü** [−20,8, −0,3], p = 0,03 → **desteklenmedi.** Yön isabeti %65. 2023–2024'te kötü, 2025'te Boşluk + Prim modeli RW'den %14 iyi |
| 2006–2022 tarihsel sağlamlık | Keşif | 17 yılın 17'sinde iyileşme, ama 2005–2010 kur verisinde sızıntı tespit edildi (bkz. 4.2) |

**Yorum.** 2023–2024'te fiziki fiyat uluslararası fiyattan ayrıştı (yurt içi prim). Geçmişten öğrenilen "fiziki fiyat teorik fiyata yakınsar" ilişkisi bu rejimde bozuldu. Model yönü çoğu gün doğru tahmin etti, ama değişimin büyüklüğünü abarttı.

### 6.4 Kazancın kaynağı (2012–2022, keşif)

Canlı protokolün kodu, geçmişin her günü için birebir çalıştırıldı.

| Bilgi | RW'ye göre | Güven aralığı | Zamanlama varsayımına bağlı mı? |
| --- | --- | --- | --- |
| **Yalnız gece ons hareketi** | **+%5,1** | [+3,0, +7,6] | **Hayır.** 11 yılın 11'inde iyileşme |
| Yalnız gece kur hareketi | +%8,2 | [+4,3, +12,4] | Evet |
| İkisi birlikte | +%21,7 | [+17,9, +25,2] | Evet |
| Kur bir gün bayat verilirse | +%0,7 | [−7,6, +8,9] | — |

Gece ons hareketinin katkısı, verinin saat damgasından bağımsız olarak kanıtlanmıştır. Kur katkısının geçmişteki büyüklüğü ise Yahoo kur verisinin eski yıllarda da sabah saatine ait olduğu varsayımına dayanıyor; bu varsayım yalnızca 2024–2025 için doğrulanabildi. Canlı sistem kuru, saati açıkça belli bir çubuktan (05:00 UTC) aldığı için bu soruyu ileriye dönük olarak cevaplayacak.

### 6.5 Fiziki tahmin aralıkları (2012–2022, keşif)

Boşluk modeli, %80 aralık:

| Yöntem | Kapsam | Genişlik | Winkler | Kupiec p | Christoffersen p |
| --- | --- | --- | --- | --- | --- |
| RW, statik | 0,784 | %2,85 | 17,65 | 0,063 | 0,000 |
| Boşluk, statik | 0,778 | %2,13 | 13,51 | 0,011 | 0,000 |
| **Boşluk, oynaklık ölçekli + ACI** | **0,800** | %2,42 | **12,94** | **0,975** | 0,043 |

- **%95 aralıkta:** ölçekli + ACI yöntemi kapsam 0,949 verdi ve kümelenme testini geçti (p = 0,354).
- **Prim-Çapa modelinde:** iki düzeyde de bütün kalibrasyon testleri geçiyor.
- **Statik RW aralığına göre:** Winkler skoru %27 daha iyi.

## 7. Canlı sistem (pilot)

Üç canlı protokol var. Fiziki fiyat protokolleri 15 Ekim 2026'da ilk tahminlerini üretti; teorik fiyat protokolünün ilk zamanlanmış çalıştırması 15 Ekim 22:30:

| Protokol | Ne üretir | Değerlendirme |
| --- | --- | --- |
| `LIVE_PHYS_V1` | Bugünün fiziki kapanışı için 4 model: RW, Boşluk, Boşluk-Kayan, Prim-Çapa | 120 tahmin sonra bir kez; DM testi ve Holm düzeltmesi |
| `LIVE_PHYS_INT_V1` | Her modelin etrafında oynaklığa duyarlı %80 ve %95 aralıkları. Kalibrasyon, canlı kodun gerçek saatlik kur arşiviyle geçmişe dönük çalıştırılmasıyla yapılıyor | 120 gün sonra bir kez; kapsam, Kupiec, Christoffersen |
| `LIVE_V1` | Eşzamanlı teorik gram için akşam tahmini | Canlı karne |

**Güvenceler:**

- **Zaman penceresi:** tahmin, yalnızca hedef günün sabahı ve belirli bir saat aralığında kabul ediliyor. Veri eski ya da eksikse sistem tahmin üretmiyor ve nedenini yazıyor.
- **Değiştirilemez kayıt:** her tahmin satırının özeti tutuluyor ve sonradan yapılan değişiklikler reddediliyor. Kayıtlar her gün Git'e gönderiliyor.
- **Kilitli kod:** ön kayıttaki kod özeti değişirse, o tahminler değerlendirmeye sayılmıyor.
- **Otomasyon:** Windows Görev Zamanlayıcı; hafta içi 12:30 (yedek 13:30) ve 22:30. Görevler kuruldu ve **elle tetiklenen** bir çalıştırmayla arka planda doğru çalıştıkları doğrulandı (15.10.2026; bütün adımlar çalıştı, çıkış kodu 0). Zamanlanmış saatte kendiliğinden çalışma ilk kez 15.10 22:30 ve 16.10 12:30'da gözlenecek.
- **Kayıt bütünlüğü kontrolü (15.10.2026):** iki kaydın satır özetleri doğrulandı; bu bölümdeki değerler doğrudan kayıt dosyalarından okundu.

**İlk canlı tahmin (15.10.2026).** Son bilinen fiziki kapanış 14.10.2026: 6.500,00 TL. Ons 4.107,90 $, kur 49,3968 (05:00 UTC çubuğu), sabah teorik fiyat 6.523,93 TL.

| Model | Tahmin | %80 aralık | %95 aralık |
| --- | --- | --- | --- |
| RW | 6.500,00 | 6.417 – 6.627 | 6.332 – 6.688 |
| Boşluk | 6.531,46 | 6.417 – 6.667 | 6.354 – 6.743 |
| Boşluk-Kayan | 6.531,58 | 6.411 – 6.666 | 6.344 – 6.735 |
| Prim-Çapa | 6.466,45 | 6.374 – 6.552 | 6.348 – 6.615 |

## 8. Pilot gösterim planı (toplantıda)

1. **Veri ve doğrulama:** `python -m src.data.physical_gold --snapshots ...` çıktısı. 1995–2026 serisi, birleşme noktası, prim tablosu.
2. **Canlı kayıt:** `python -m src.live.physical_forecast card` ve `python -m src.live.physical_intervals card`. Kaç tahmin yapıldı, kaçı puanlandı, hata ve kapsam.
3. **Kaydın değiştirilemezliği:** `git log --oneline -- research/live/` ile her günün tahmininin zaman damgalı commit'leri. Ayrıca `research/preregistration/` altındaki ön kayıtlar.
4. **Bağımsız sınavlar:** `research/final_exam_v1/result.json` ve `research/final_exam_phys_v1/result.json`. Olumsuz sonuçların da kayıtlı olduğunun gösterilmesi.
5. **Bir günün tahmini:** `logs\` klasöründeki son sabah kaydı. Girdiler, zaman damgaları, dört model ve aralıklar.

## 9. Sınırlılıklar

- **Fiziki fiyat toptan piyasa fiyatıdır;** kuyumcu alış/satış fiyatı ve işçilik bu çalışmanın kapsamında değil.
- **Fiziki tahmindeki kazancın büyük kısmı** geçmişte doğrulanamayan bir kur zamanlama varsayımına dayanıyor. Zamanlamadan bağımsız kısım %5. Kesin sonuç canlı kayıtla belli olacak.
- **Kilitli dönemin kullanımı:** 2023 sonrası fiziki veri bağımsız sınavda tüketildi. Sınavdan sonra önerilen modeller (Boşluk-Kayan, Prim-Çapa) yalnızca ileriye dönük olarak değerlendirilebilir.
- **Kaynak sınırları:** saatlik veri Yahoo'da yalnızca son ~2 yıl için var; canlı sistem bu yüzden her gün arşivliyor.
- **Teknik not (gözlendi ve doğrulandı):** puanlama adımı kaydı okuyup yeniden yazarken, pandas'ın varsayılan CSV okuyucusu bazı ondalık sayıların yalnızca son basamaklarını farklı geri okuyabiliyor. 15.10.2026 kaydında üç ara değişkenin (boşluk, prim, katsayı) gösterimi 16.–17. anlamlı basamakta değişti; tahmin değerleri ve satır özeti birebir aynı kaldı. Satır özetleri 10 anlamlı basamakla hesaplandığı için bütünlük kontrolü bu farktan etkilenmiyor. Sentetik denemede fark en fazla iki yeniden yazımda duruyor.

## 10. Sonraki adımlar

1. **Canlı kaydın birikmesi:** Nisan 2027'de iki protokolün tek seferlik değerlendirmesi.
2. **Teorik seri için spot ons kaynağı:** vadeli kontrat sorununun çözümü, Kasım sonundan önce.
3. **Makro göstergeler ve uzun ufuk:** TCMB politika faizi, enflasyon; haftalık ve aylık trend analizi; LSTM karşılaştırması.
4. **Sistem katmanı:** Microsoft SQL Server, REST API ve web arayüzü.
5. **Kuyumcu fiyatları:** veri bulunabilirse fiziki toptan fiyat ile perakende fiyat arasındaki farkın analizi.
