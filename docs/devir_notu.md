# PROJE DEVİR NOTU — Türkiye Gram Altın Tahmini (15.10.2026)

Bu belge, projenin bütün durumunu yeni bir sohbette sıfırdan devam edebilmek için yazıldı. **Yeni sohbetteki Claude'a:** önce bu belgenin tamamını oku. Burada "kilitli" ya da "değiştirilmez" denen hiçbir şeye dokunma. Commit veya push öncesi kullanıcıdan onay iste.

---

## 0. Kullanıcı ve çalışma biçimi

- **Kullanıcı:** Fatma Betül Pehlivan, Selçuk Üniversitesi Bilgisayar Mühendisliği 4. sınıf. Bitirme projesi. Türkçe konuşuyor ve yazım hataları olabiliyor; anlamı esas al.
- **Proje başlığı (danışmanın kapsamı):** "Türkiye'de Gram Altın Fiyatlarının Makine Öğrenmesi Tabanlı Tahmini ve Analiz Sistemi". Fiziki altın fiyatları, geçmiş fiyatlar, USD/TRY, ons ve ekonomik göstergelerle kısa vadeli fiyat ve trend tahmini. Teknolojiler: Python, pandas, scikit-learn, XGBoost/LSTM, görselleştirme, REST API, web arayüzü, Microsoft SQL Server.
- **Ayrıca:** TÜBİTAK 2209-A ve HAVELSAN SUIT başvuruları için hocalara mail atıldı. Mailler ölçülü; abartılı iddia yok.
- **Ortam:** Windows 11, Visual Studio 2022 Developer PowerShell, Python 3.11, proje kökünde `.venv`.
  - Proje yolu: `C:\Users\pehlivan\source\repos\turkey-gold-price-forecast`
  - Oturum başı: `cd ...` ve ardından `.\.venv\Scripts\Activate.ps1`
- **Dosya alışverişi:** Claude'un ürettiği dosyalar kullanıcının **Masaüstüne** (bazen `Desktop\gold` ya da İndirilenler) iner; aynı adlı dosyalar `README (1).md` gibi adlandırılır. Komutları buna göre ver; gerekirse önce `Get-ChildItem` ile bulma komutu yaz.
- **Kullanıcının tercihleri:**
  - Adım adım, kopyala-yapıştır komutlar.
  - Kısa ve net açıklama, gerçekçi değerlendirme.
  - "Optimum, profesyonel iyileştirme" istiyor, ama **imkânsızı vaat etme.**
- **Terminal çıktıları:** mümkünse **metin olarak** istensin (görsel sınırı var). Örnek: `komut | Set-Clipboard`.
- **Claude'un ortamı:** `/home/claude` sandbox. Ağ erişimi yalnızca GitHub, PyPI, npm, Ubuntu. **EVDS'ye, Yahoo'ya ve kullanıcının verisine erişim YOK;** gerçek veri çalıştırmalarını kullanıcı yapar. Depoyu indirmek için:
  `curl -sL -o r.zip https://codeload.github.com/03betulpehlivan/turkey-gold-price-forecast/zip/refs/heads/main`
  Çıktıyı dosyaya yönlendirip `iconv -t ascii//TRANSLIT` ile oku (Türkçe karakterler terminali bozabiliyor). PowerShell sandbox'ta YOK; `.ps1` dosyaları yalnızca yapısal Python testleriyle sınanır.

## 1. Değişmez kurallar (çok önemli)

1. **Kullanıcı onayı olmadan commit ve push talimatı verilmez.** Kullanıcı genelde "devam" der; yine de commit komutlarını açıkça listele.
2. **Bağımsız sınavlar tamamlandı ve tekrar çalıştırılamaz:** `research/final_exam_v1/`, `research/final_exam_phys_v1/` ve bunların ön kayıtları değiştirilmez.
3. **Fiziki veride hedef tarihi ≥ 2023-01-01 olan dönem KİLİTLİ:** model, özellik veya hiperparametre seçimi ya da strateji değişikliği için kullanılmaz. (Eğitim verisi olarak canlı protokollerde kullanılması serbest; seçim için değil.) Teorik 2026 dönemi de tüketildi.
4. **Canlı protokol kodları ön kayıtla kilitli:**
   - `src/live/physical_forecast.py` (md5 `95b49469009c34a87334a416c60c70cd`)
   - `src/live/physical_intervals.py`, `src/models/interval_study.py` (LIVE_PHYS_INT_V1 birleşik özeti)
   Değiştirilirse tahminler değerlendirme dışı kalır. Değişiklik gerekiyorsa **yeni protokol kimliğiyle** yapılır (örneğin `LIVE_PHYS_V2`).
5. **Ana `data/processed/market_panel.csv` ve `data/processed/physical/physical_gold_daily.csv`'nin üzerine yazılmaz;** bağımsız sınavların md5 güvencesi bunlara bağlı. Canlı sistem kendi kopyalarını kullanır: `data/processed/live/`. Genişletilmiş tarih paneli: `data/processed/history/`.
6. **Sonuçlar her zaman "keşif / bağımsız sınav / canlı" diye etiketlenir.** Olumsuz sonuçlar saklanmaz ya da yumuşatılmaz. Sentetik test başarısı piyasa başarısı diye sunulmaz.
7. **API anahtarı** yalnızca `EVDS_API_KEY` ortam değişkeninde (`setx`); kodda, Git'te ya da kayıtlarda asla yer almaz.
8. **Kod standartları** (yeni dosyalarda da korunacak):
   - Python kaynakları **ASCII** (Türkçe karakterler için `\uXXXX` kaçışları).
   - Testler `unittest`, **sentetik veri**, gerçek klasörleri koruyan md5 korumaları (`setUpModule` / `tearDownModule`).
   - Her yeni özellik için **mutasyon kontrolü:** koda bilinçli hata sok, testin yakaladığını göster. Yakalanmayan mutasyon için testi güçlendir ya da nedenini açıkla.
   - `.gitattributes`: `*.py text eol=lf`. Kod özeti satır sonu dönüşümünden etkilenmesin diye eklendi.
9. **Yeni modüllerde** varsayılan kayıt klasörü çalışma anında okunmalı. (Eski bir hata: fonksiyon tanımında sabitlenmiş varsayılan, testleri gerçek klasöre yazdırıyordu.)

## 2. Depo yapısı (önemli dosyalar)

```
src/data/
  collect.py, build_panel.py, build_dataset.py, validate.py   Yahoo günlük, panel, doğrulama
  collect_intraday.py, sync_dataset.py                         saatlik arşiv, eşzamanlı teorik seri (14:00 NY)
  evds.py                                                      EVDS istemcisi (keşif, parçalı indirme)
  physical_gold.py                                             fiziki altın serisi 1995+, prim analizi
src/analysis/
  fx_close_time.py, timing_check.py, sync_prices.py, data_diagnostics.py
  spot_probe.py                                                spot ons adayları (XAUUSD=X, GLD, IAU) - HENÜZ ÇALIŞTIRILMADI
  physical_errors.py                                           fiziki tahmin hata analizi (keşif)
  fx_timing_history.py                                         Yahoo kur çubuğu saati, yıllara göre (BIST örtük kur)
  gain_decomposition.py                                        kazancın kaynağı: gece ons ve kur (keşif)
src/models/
  train.py, point_eval.py, factors.py, intervals.py, implied_vol.py, adaptive.py
  final_exam.py                                                FINAL_EXAM_V1 (TAMAMLANDI)
  physical_eval.py                                             fiziki sabah tahmini, 2016-2022 (keşif; 2023+ kilit)
  physical_exam.py                                             FINAL_EXAM_PHYS_V1 (TAMAMLANDI)
  physical_history.py                                          rejim sağlamlığı 2006-2022 (keşif)
  live_backtest.py                                             canlı protokolün 2012-2022 geriye dönük testi (iki kur sürümü)
  interval_study.py                                            oynaklık ölçekli ve ACI aralıklar (keşif) - LIVE_PHYS_INT_V1'e bağımlı, KİLİTLİ
src/live/
  forecast.py                                                  LIVE_V1 (teorik, akşam)
  physical_forecast.py                                         LIVE_PHYS_V1 - KİLİTLİ
  physical_intervals.py                                        LIVE_PHYS_INT_V1 - KİLİTLİ
scripts/
  gunluk_tahmin.ps1        akşam teorik (-TahminYok seçeneği var)
  sabah_fiziki.ps1         sabah: collect -> live panel -> intraday -> EVDS refresh -> score -> issue -> card ->
                           aralık score -> aralık issue -> aralık card -> iki kaydın commit+push
  zamanli_calistir.ps1     Görev Zamanlayıcı sarmalayıcısı; çıktı logs\<gorev>_<tarih>.log
  gorev_kur.ps1            iki görevi kaydeder
research/
  preregistration/         final_exam_v1.json, final_exam_phys_v1.json, live_phys_v1.json, live_phys_int_v1.json
  final_exam_v1/, final_exam_phys_v1/   sınav sonuçları
  live/physical_log.csv, live/physical_intervals_log.csv   (canlı teorik kayıt: research/live/forecast_log.csv)
  registry.jsonl           bütün protokol olayları (prepared, started, completed, refused, evaluated)
docs/   danisman_raporu.md (danışman ara raporu), durum_notu.md, veri_platformu.md, devir_notu.md (bu belge)
logs/   zamanlanmış çalıştırma kayıtları (.gitignore'da)
```

**Testler:** 429 test, hepsi geçiyor (`python -m unittest discover -s tests`, ~2 dakika). Bilinen zararsız uyarılar: `adaptive.py` ve `intervals.py` içinde kapatılmamış dosya (`ResourceWarning`); pandas `invalid value in log`; `test_physical_intervals.py`'nin `read_bytes` yamasında kapatılmamış dosya. İleride temizlenebilir.

## 3. Veri

| Veri | Ayrıntı |
| --- | --- |
| **Yahoo günlük** | `data/raw/snapshots/<zaman>/`. 11 seri: gold GC=F, usdtry USDTRY=X, silver, dxy, us10y, vix, sp500, bist100, brent, eurtry, gvz. Varsayılan başlangıç 2015; `--start 2005-01-01` ile 2005+ indirildi |
| **Paneller** | Ana: `data/processed/market_panel.csv` (2015+, **dokunma**). Canlı: `data/processed/live/market_panel.csv` (her sabah ve akşam yeniden kurulur). Tarih: `data/processed/history/market_panel.csv` (2005-01-03 → 2026-10-13, 5.461 gün) |
| **Panelin yapısı** | Temel tarihler, altın ve kurun **aynı tarihte** iç birleşimi (`build_dataset.build`, `how="inner"`). Panelde T günü varsa `usdtry`, Yahoo'nun T tarihli gerçek çubuğu. Ek seriler `merge_asof` ile taşınıyor |
| **Yahoo saatlik** | `data/raw/snapshots_intraday/`; 2024-10'dan beri, her gün arşivleniyor. Yahoo yalnızca ~730 gün tutar, bu yüzden arşiv değerli. `sync_dataset.merge_snapshots()` bütün görüntüleri birleştirir ve tamamlanmamış çubukları atar |
| **EVDS (fiziki altın)** | Ayrıntılar aşağıda |

**EVDS ayrıntıları:**

- **Erişim:** adres `https://evds3.tcmb.gov.tr/igmevdsms-dis/`; anahtar istek başlığında (`key`); eski TLS ayarı gerekiyor.
- **Gerçek çalıştırmada gözlenen davranışlar:**
  - `endDate` boş olursa 400 döner.
  - Yanıt başına **en fazla 1.000 satır;** fazlası sessizce kesiliyor. `evds.py` istekleri parçalara bölüyor.
- **Seriler:**
  - Arşiv (1995-07-27 – 2018-06-29): `TP.MK.T.KAP.Y` (kapanış TL/kg), `TP.MK.T.AOF.Y` (ağırlıklı ortalama), `TP.MK.D.KAP.Y` (USD/ons).
  - Güncel (2018-07-02 – ): `TP.ALTINPIYASA.KAP02`, `.AGORT02`, `.KAP03`.
  - KAP05 (TL/gr) seyrek olduğu için kullanılmıyor.
  - USD kuru: `TP.DK.USD.A.YTL`.
- **Fiziki seri:** `data/processed/physical/physical_gold_daily.csv`. 7.038 gün (sonra 7.039); kaynak dağılımı İAB arşivi 4.981, BIST 2.057.
- **Kalite kontrolleri:**
  - 7 günde TL fiyatı 0 (işlem yok; boş sayıldı; 2009–2010 arifeleri).
  - Birleşme noktası 184,13 → 185,35 (+%0,66).
  - %15'ten büyük günlük hareketler: 23.02.2001, 17.09.2001, 21.12.2021 (gerçek olaylar).
- **Canlı kopya:** `data/processed/live/physical_gold_daily.csv` (`physical_forecast refresh`).
- **Yayın zamanı:** EVDS bir günün kapanışını ertesi gün yayımlıyor. 15.10 sabahı 12:54'te 14.10 verisi vardı; daha erken saatler bilinmiyor.

## 4. Araştırma sonuçları

| Konu | Tür | Sonuç |
| --- | --- | --- |
| Teorik fiyat, ertesi gün, 2018–2023/10 | Keşif | Hiçbir model RW'yi geçemedi; RW MAE 6,06 TL, MAPE %1,07 |
| **Teorik, `FINAL_EXAM_V1`, 2026, 194 tahmin** | **Bağımsız** | Mean-Return MAPE %1,345, RW %1,347 → **pratikte eşdeğer** (fark −%0,21 [−1,34, +1,17]) |
| Teorik aralıklar, 2023/10–2025/12 teyit | Bağımsız | Kapsam 0,788 / 0,940 → **desteklendi** |
| Teorik aralıklar, 2026 | Bağımsız | Kapsam 0,794 / 0,938; Christoffersen %80 ≈ 0,05 → **desteklenmedi** |
| Prim (fiziki ÷ teorik − 1, yıllık medyan) | Betimsel | 2015–2022 ~%0; 2023 +%1,17; 2024 +%1,38 (bazı günler +%5); 2025 +%0,45; 2026 ~%0; son 30 gün −%0,9 |
| Fiziki sabah tahmini, 2016–2022, 1.741 tahmin | Keşif | Boşluk +%22,8 (MAE 3,22 / RW 4,18 TL; MAPE %0,78 / %1,01), yön %71,2; İhtiyatlı +%6,5; Boşluk+Prim +%22,5; XGB +%16,6 |
| Hata analizi, 2016–2022 | Keşif | Ayrıntılar aşağıda |
| **Fiziki, `FINAL_EXAM_PHYS_V1`, hedef 2023-01-04 … 2026-10-09, 874 tahmin** | **Bağımsız** | Ayrıntılar aşağıda |
| Tarihsel sağlamlık, 2006–2022, 3.656 tahmin | Keşif | Boşluk +%24,3; 17/17 yıl; rejimler +%30,1 / +%21,5 / +%24,3; b 0,68–0,78. 272 bayat satır dışarıda |
| **Kur saati, TCMB referansı** | Teşhis | **TUTARSIZ** (EVDS geçerlilik günü tarihlendirmesi olası); kullanılmıyor |
| **Kur saati, BIST örtük kur referansı (yalnızca ≤ 2022)** | Teşhis | 2005–2010 **KAPANIŞ_SONRASI** (lead 0,247, bant 0,061) → **sızıntı; eğitimden çıkarıldı.** 2011–2015 KAPANIŞ_ÖNCESİ. 2016–2022 GEÇERSİZ (kontrol 0,050, bant 0,048) |
| Kur saati, saatlik, 2024–2025 | Teşhis | Yahoo günlük kur ~00–06 UTC (en iyi 05:00, 3,10 baz puan; 07+ belirgin kötü) |
| `live_backtest`, 2012–2022, 2.502 tahmin; 236/2.738 red (çoğu tatil) | Keşif | Ayrıntılar aşağıda |
| **Kazanç ayrıştırma, 2012–2022, 2.523 tahmin** | Keşif | Ayrıntılar aşağıda |
| Aralıklar (`interval_study`, aynı gün kuru) | Keşif | Ayrıntılar aşağıda |

**Hata analizi (2016–2022, keşif):**

- Boşluk eğimi b = 1,10 [0,91, 1,28]: abartı yok. XGB b = 0,73: abartıyor.
- Kontrat günlerinde iyileşme +%38,9; zarar yok.
- Prim beşlilerinin hepsinde +%16 ile +%26 arası iyileşme.
- En kötü %1'lik gün çıkarılınca +%20,9.
- Büyük tahmin üçlüsünde yön isabeti %87, iyileşme +%38.

**`FINAL_EXAM_PHYS_V1` ayrıntıları (bağımsız):**

- Boşluk **−%9,07** [−20,77, −0,27], DM-HLN +2,15, p = 0,032 → **DAHA_KÖTÜ.** Yön isabeti %64,8.
- İkincil modeller: İhtiyatlı +%1,98, Boşluk+Prim +%2,19 (ikisi de anlamsız); XGB −%9,71.
- Yıllara göre MAPE (RW / Boşluk): 2023: 1,63 / 1,94; 2024: 1,26 / 1,60; 2025: 1,03 / 0,99 (+Prim 0,89); 2026: 1,29 / 1,36 (+Prim 1,24).
- Tazelik kuralıyla 66 satır dışarıda.

**`live_backtest` ayrıntıları (keşif):**

- **Aynı gün kuru:** Boşluk +%24,4 [20,8, 28,0], Kayan +%22,5, Prim-Çapa +%22,0; yön ~%70.
- **Önceki gün kuru:** +%0,7 (anlamsız). Bu "alt sınır", eğitim ile kullanım tutarsızlığı yüzünden olduğundan düşük olabilir.

**Kazanç ayrıştırma ayrıntıları (keşif):**

- **Yalnız ons:** **+%5,1** [3,0, 7,6], **11/11 yıl,** yön %61,9. **Zamanlamadan bağımsız.**
- Yalnız kur: +%8,2 [4,3, 12,4].
- İkisi birden: +%21,7 [17,9, 25,2].

**Aralık ayrıntıları (`interval_study`, aynı gün kuru, keşif):**

- **Boşluk %80:**
  - Statik: kapsam 0,778, Winkler 13,51, Kupiec 0,011, Christoffersen 0,000.
  - Ölçekli: kapsam 0,795, Winkler 12,88.
  - Ölçekli+ACI: kapsam 0,800, Winkler 12,94, Kupiec 0,975, Christoffersen 0,043.
- **Boşluk %95, Ölçekli+ACI:** kapsam 0,949, Christoffersen 0,354.
- **Prim-Çapa, Ölçekli+ACI:** %80'de 0,799 / Christoffersen 0,373; %95'te 0,949 / 0,379.
- **RW statik:** Winkler 17,65, yani ölçekli Boşluk %27 daha iyi.

**Veri bulguları:**

- Yahoo zamanlama uyumsuzluğu: ons ~18 UTC, kur ~02–05 UTC.
- Vadeli kontrat sıçramaları %1,5–1,9: 30.07.2025 / 01.08.2025 ve 29.07.2026 / 31.07.2026.
- 30.01.2026'da ~%9–11'lik büyük düşüş.

## 5. Canlı protokoller

**LIVE_PHYS_V1** (ön kayıt `research/preregistration/live_phys_v1.json`, kod md5 `95b49469…`):

- **Hedef:** bugünün fiziki kapanışı (TL/gr).
- **Bilgi:**
  - EVDS'de yayımlanmış son kapanış (en fazla 4 gün eski).
  - Ons: panelde bugünden kesinlikle önceki son kapanış; tarihi ≥ son fiziki gün olmalı.
  - Kur: saatlik USD/TRY, bitişi bugün **05:00 UTC** olan çubuk.
- **Modeller:**
  - RW.
  - Boşluk: OLS, 2011'den beri bütün taze satırlar. Canlı panel 2015'ten başladığı için fiilen 2015+.
  - Boşluk-Kayan: son 250 satır.
  - Prim-Çapa: M × exp(prim EWMA, yarı ömür 20).
- **Zaman penceresi:** hedef günün İstanbul tarihinde, 05:00–11:00 UTC (08:00–14:00 TSİ).
- **Reddetme:** hafta sonu; erken; geç; hedef günün kapanışı yayımlanmışsa; EVDS bayatsa; panel bayatsa; kur çubuğu yoksa.
- **Kayıt:** `research/live/physical_log.csv`; satır başına `row_hash` (sayılar 10 anlamlı basamak).
- **Değerlendirme:** 120 puanlanmış tahminden sonra (yalnızca ön kayıttaki md5 ile üretilenler), DM-HLN, Holm (3), bootstrap.
- **Komutlar:** `python -m src.live.physical_forecast prepare|refresh|score|issue|card|evaluate`

**LIVE_PHYS_INT_V1** (ön kayıt `live_phys_int_v1.json`; birleşik kod özeti üç dosyayı kapsıyor, satır sonundan bağımsız):

- **Merkez:** V1 kaydındaki tahminler; yeniden hesaplanıp birebir eşleştiği doğrulanır.
- **Kalibrasyon:** son en fazla 500 gün için canlı kod, gerçek saatlik kurla (D 04:00 UTC çubuğu) yeniden çalıştırılır; en az 300 gün gerekir.
- **Yöntem:** Ölçekli+ACI (λ 0,94, son 250 bilinen hata, η 0,02).
- **Kayıt:** `research/live/physical_intervals_log.csv`.
- **Değerlendirme:** 120 gün sonra; "kalibre" = Kupiec p > 0,05 ve Christoffersen p > 0,05.

**LIVE_V1** (teorik, akşam): `src/live/forecast.py`, eşzamanlı 14:00 NY serisi. Kayıt `research/live/forecast_log.csv`. Tahmin, NY takvim gününde ve 14:00 NY çubuğu kapandıktan sonra kabul edilir. Ana risk: vadeli kontrat değişimi (bir sonraki ~Kasım sonu).

**İlk canlı tahmin (15.10.2026, 09:55 UTC):**

- **Girdiler:** son fiziki kapanış 14.10: 6.500,00 TL. Ons 14.10: 4.107,90 $. Kur (05:00 UTC): 49,3968. Sabah teorik 6.523,93 TL; boşluk +%0,37; prim −%0,89.
- **Tahminler:** RW 6.500,00 · Boşluk 6.531,46 · Kayan 6.531,58 · Prim-Çapa 6.466,45.
- **Aralıklar** (476 kalibrasyon günü, 2024-10-14 … 2026-10-14):

| Model | %80 | %95 |
| --- | --- | --- |
| RW | 6.417,31 – 6.626,72 | 6.331,50 – 6.687,59 |
| Boşluk | 6.416,53 – 6.666,61 | 6.353,87 – 6.743,27 |
| Boşluk-Kayan | 6.411,22 – 6.665,75 | 6.344,18 – 6.735,29 |
| Prim-Çapa | 6.374,47 – 6.551,81 | 6.348,15 – 6.614,69 |

## 6. Otomasyon (Windows Görev Zamanlayıcı)

- **Görevler:**
  - `AltinTahmin-Sabah`: hafta içi 12:30 ve 13:30. İkinci çalışma yedek; ilki başarılıysa "zaten kayıtlı" der.
  - `AltinTahmin-Aksam`: hafta içi 22:30. Kış saatinde de güvenli.
- **Zincir:** Görev Zamanlayıcı → `powershell.exe` → `scripts\zamanli_calistir.ps1 -Gorev sabah|aksam` → `sabah_fiziki.ps1` / `gunluk_tahmin.ps1` → Python modülleri.
- **Ayarlar:** yalnızca kullanıcı oturumu açıkken çalışır; parola saklanmaz. `StartWhenAvailable`, `IgnoreNew`, 1 saat zaman sınırı.
- **Doğrulama durumu:** 15.10 14:55'te elle tetiklendi; sonuç 0, transcript kaydı oluştu, push çalıştı. **Zamanlanmış saatte kendiliğinden çalışma henüz gözlenmedi:** ilkleri 15.10 22:30 ve 16.10 12:30.
- **Komutlar:**
  - Durum: `Get-ScheduledTask -TaskName "AltinTahmin-*" | Get-ScheduledTaskInfo | Select-Object TaskName, LastRunTime, LastTaskResult, NextRunTime`
  - Kayıtlar: `Get-ChildItem logs | Sort-Object LastWriteTime | Select-Object -Last 2 | ForEach-Object { "== $($_.Name)"; Get-Content $_.FullName -Tail 15 }`
  - Durdurma ve kaldırma: `Disable-ScheduledTask`, `Unregister-ScheduledTask`. Yeniden kurulum: `scripts\gorev_kur.ps1`.
  - 267011 kodu "henüz çalışmadı" demek.
- **Gereksinimler:** bilgisayar açık ve oturum açık; `EVDS_API_KEY` ayarlı (`setx`); Git kimlik bilgileri kayıtlı.

## 7. Bilinen notlar ve riskler

1. **CSV son basamak farkı:** pandas'ın varsayılan CSV okuyucusu bazı ondalık sayıları son basamakta farklı geri okuyor (değerlerin ~%9'u). Puanlama yeniden yazınca küçük farklar oluşuyor.
   - Doğrulandı (`cae9a2b`): yalnızca boşluk, prim ve katsayı 16.–17. basamakta değişti. Tahminler ve satır özeti aynı kaldı.
   - En fazla iki yeniden yazımda duruyor.
   - **Uç risk:** bir değer tam 10. basamak yuvarlama sınırındaysa özet değişebilir (~milyonda bir). Olursa sabah görevi "KAYIT BÜTÜNLÜĞÜ HATASI" ile durur; asıl kayıt Git geçmişinde.
   - Kilitli kod olduğu için düzeltilmedi; bilinçli karar.
2. **Teorik seri ve vadeli kontrat sıçraması:** `spot_probe.py` hazır ama çalıştırılmadı. Kasım sonundan önce karar verilmeli.
3. **EVDS anahtarı sohbette paylaşıldı:** yenilenmesi önerildi. Yenilenirse `setx EVDS_API_KEY "yeni"` gerekir; aksi hâlde sabah görevi 401 alır.
4. **README kararları kullanıcıda:**
   - Contributors bölümünde Enes Karaca var (TÜBİTAK'a "bireysel" yazılmıştı).
   - Depoda `LICENSE` dosyası yok, ama README MIT diyor.
5. **Satır sonu:** `.ps1`, `.md` ve `.csv` dosyalarında Git LF→CRLF uyarısı veriyor; `.py` için `.gitattributes` var.
6. **Yahoo verisi:** ons değerleri float32 kaynaklı olabilir (`4107.89990234375`); sorun değil.

## 8. Açık işler (öncelik sırasıyla)

1. **Otomasyonun kendiliğinden çalıştığını doğrulamak** (15.10 22:30 ve 16.10 12:30 kayıtları). İlk karneyi yorumlamak: 15.10 tahminleri ve aralık kapsamı. Doğrulanınca danışman raporundaki otomasyon cümlesini güncellemek.
2. **Danışman toplantısı:** `docs/danisman_raporu.md` Bölüm 8'deki gösterim planı.
3. **Spot ons kararı** (teorik seri): `python -m src.analysis.spot_probe`.
4. **Danışmanın kapsamındaki eksikler:**
   - Makro göstergeler (TCMB faizi, enflasyon; EVDS'den alınabilir).
   - Haftalık ve aylık trend.
   - LSTM karşılaştırması (yalnızca gerekçeli bir deney olarak).
   - Sonra Microsoft SQL Server, REST API (FastAPI) ve web arayüzü.
5. **Bakım:** test uyarılarının temizlenmesi; canlı modüllerde değişiklik ancak yeni protokol kimliğiyle.
6. **Nisan 2027:** `python -m src.live.physical_forecast evaluate` ve `python -m src.live.physical_intervals evaluate`. Tek sefer; sonuçlar commit edilir.

## 9. Önemli commit'ler

- `96f8437`: FINAL_EXAM_PHYS_V1 sonucu
- `5023ec2`: LIVE_PHYS_V1 kodu
- `4418ac8`: LIVE_PHYS_V1 ön kaydı
- `97c134c`: `.gitattributes`
- `ab16059`: ilk canlı tahmin
- `31cdbc6`: LIVE_PHYS_INT_V1 ön kaydı
- `01c8382`: Görev Zamanlayıcı
- `cae9a2b`: CSV son basamak farkı commit'i
- `3ffc483`: danışman raporu
- `f563802`: README

## 10. Yeni sohbette ilk adım

Kullanıcı yeni sohbete bu belgeyle birlikte şu çıktıyı **metin olarak** yapıştırır:

```powershell
Get-ScheduledTask -TaskName "AltinTahmin-*" | Get-ScheduledTaskInfo | Select-Object TaskName, LastRunTime, LastTaskResult | Format-Table -AutoSize
Get-ChildItem logs | Sort-Object LastWriteTime | Select-Object -Last 2 | ForEach-Object { "== $($_.Name)"; Get-Content $_.FullName -Tail 15 }
python -m src.live.physical_forecast card
python -m src.live.physical_intervals card
```

Claude bu çıktıdan şunları kontrol eder:

- İki görev de zamanlanmış saatte sonuç 0 ile çalıştı mı?
- Kayıtlarda HATA var mı?
- 15.10 tahminleri puanlandı mı?
- Hangi model gerçek kapanışa daha yakın çıktı ve aralıklar tuttu mu? Tek günlük sonuç yalnızca betimleyicidir.
