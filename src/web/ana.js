"use strict";
/* Ana ekran: gram altin icin 1, 3, 5 islem gunu sonrasi fiyat araligi ve aciklamalar.
   CSP geregi: satir ici stil ve HTML dizesi yok; ogeler createElement ile kurulur, konumlar element.style ile verilir. */
const NS = "http://www.w3.org/2000/svg";
const AY = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"];
const AD = { "Sabit aralik": "Sabit aralık (basit yöntem)", "Gecmis benzer": "Geçmiş benzer günler", "EWMA": "EWMA",
  "GARCH-t": "GARCH-t", "GJR-t": "GJR-GARCH", "EWMA+ACI": "EWMA + ayar", "GARCH-t+ACI": "GARCH-t + ayar",
  "XGBoost-kantil": "XGBoost (makine öğrenmesi)", "ARIMA-GARCH": "ARIMA-GARCH", "HAR": "HAR", "Rejim (Markov)": "Rejim (Markov)",
  "Birlesim": "Birleşim (3 model)", "HAR-X": "HAR-X (dış piyasa verisi)", "GARCH-t+olay": "GARCH-t + Fed günleri",
  "Chronos": "Chronos (yapay zekâ)", "Birlesim-2": "ARIMA-GARCH + Chronos" };
const SADE = { "ARIMA-GARCH": "klasik istatistik modeli; fiyatın son günlerdeki oynaklığından önümüzdeki günlerin oynaklığını hesaplar",
  "Birlesim-2": "klasik ARIMA-GARCH ile Amazon’un yapay zekâ modeli Chronos’un ortalaması",
  "Chronos": "Amazon’un çok sayıda zaman serisiyle eğitilmiş yapay zekâ modeli",
  "GARCH-t": "oynaklığı son günlere göre güncelleyen klasik istatistik modeli" };
const ad = m => AD[m] || m;
const NE = { "Sabit aralik": "Son 3 yılın ortalama hareketine bakar. Karşılaştırma için basit yöntem.",
  "Gecmis benzer": "Son 3 yılda gerçekten yaşanmış değişimleri olduğu gibi kullanır.",
  "EWMA": "Yakın günlere daha çok ağırlık veren oynaklık ortalaması.",
  "GARCH-t": "Oynaklığı her gün güncelleyen klasik istatistik modeli; büyük sıçramalara izin verir.",
  "GJR-t": "GARCH’ın düşüşlere ayrı tepki veren türü.",
  "EWMA+ACI": "EWMA; aralık ıskaladıkça genişleyen, fazla tuttukça daralan otomatik ayarla.",
  "GARCH-t+ACI": "GARCH-t; aynı otomatik ayarla.",
  "XGBoost-kantil": "Makine öğrenmesi; aralığın sınırlarını doğrudan öğrenir.",
  "ARIMA-GARCH": "GARCH-t’ye bir önceki günün değişiminin etkisini ekler.",
  "HAR": "Son günün, son haftanın ve son ayın oynaklığını birleştirir.",
  "Rejim (Markov)": "Piyasayı sakin ve çalkantılı iki duruma ayırır.",
  "Birlesim": "GARCH-t, EWMA ve geçmiş benzer günlerin ortalaması.",
  "HAR-X": "HAR’a dış piyasa verilerini ekler (GVZ, VIX, dolar/TL, Brent, dolar endeksi).",
  "GARCH-t+olay": "GARCH-t; Fed karar günlerinde aralığı genişletir.",
  "Chronos": "Amazon’un çok sayıda zaman serisiyle eğitilmiş yapay zekâ modeli.",
  "Birlesim-2": "ARIMA-GARCH ile Chronos’un ortalaması." };
const AZ_HAREKET = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const c of kids) if (c !== null && c !== undefined && c !== "") e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  if (tag === "svg" && attrs.viewBox) {
    const [, , w, h] = attrs.viewBox.split(" ");
    e.setAttribute("width", w);
    e.setAttribute("height", h);
  }
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
async function al(url) {
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* bos */ }
    throw new Error(msg);
  }
  return r.json();
}
const sayi = (v, d = 0) => Number(v).toLocaleString("tr-TR", { minimumFractionDigits: d, maximumFractionDigits: d });
const yuzde = (v, d = 1) => "%" + sayi(v, d);
const tarih = iso => { const [y, m, g] = iso.split("-").map(Number); return `${g} ${AY[m - 1]} ${y}`; };
const kisa = iso => { const [, m, g] = iso.split("-").map(Number); return `${g} ${AY[m - 1]}`; };
const genislik = id => Math.max(300, Math.round(document.getElementById(id).clientWidth || 900));
const pdeger = p => (p == null ? "—" : p < 0.0001 ? "p < 0,0001" : `p = ${sayi(p, 4)}`);
const sans = p => (p == null ? "" : p < 0.05 ? `Bu fark tesadüf değil (${pdeger(p)})` : `Bu fark tesadüf de olabilir (${pdeger(p)})`);
const AY_EK = ["’ta", "’ta", "’ta", "’da", "’ta", "’da", "’da", "’ta", "’de", "’de", "’da", "’ta"];
const kisaDe = iso => kisa(iso) + AY_EK[Number(iso.split("-")[1]) - 1];   // "28 Ekim’de"
const yorum80 = t => (t < 78 ? "Hedefin biraz altında: aralık biraz dar."
  : t > 82 ? "Hedefin biraz üstünde: aralık biraz geniş, yani temkinli." : "Hedefe çok yakın.");

/* Sayilari eski degerden yeni degere akarak degistir (hareket azaltma tercihinde aninda). */
function sayiAkit(e, a, b) {
  if (AZ_HAREKET || a == null || !isFinite(a) || Math.round(a) === Math.round(b)) { e.textContent = sayi(b); return; }
  const t0 = performance.now(), sure = 650;
  const adim = t => {
    const x = Math.min(1, (t - t0) / sure), k = 1 - Math.pow(1 - x, 3);
    e.textContent = sayi(a + (b - a) * k);
    if (x < 1) requestAnimationFrame(adim);
  };
  requestAnimationFrame(adim);
}
const ufukAdi = h => (h === 1 ? "Bir sonraki işlem günü" : `${h} işlem günü sonra`);
const kac = d => (d === 80 ? "10 benzer günün yaklaşık 8’inde" : "100 benzer günün yaklaşık 95’inde");
const icinde = (v, a, b) => v >= a && v <= b;
const bekle = ms => new Promise(r => setTimeout(r, AZ_HAREKET ? 0 : ms));
const sonraki = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

let MERKEZ = {};
const merkezTahmin = f => (f.aralik80[0] + f.aralik80[1]) / 2;
const VERI = { u: null, y: null, g: null };

/* Gorunur olunca bir kez calistir (animasyonlar icin). */
function gorununce(e, fn) {
  if (!("IntersectionObserver" in window) || AZ_HAREKET) { fn(); return; }
  const io = new IntersectionObserver(es => { if (es.some(x => x.isIntersecting)) { io.disconnect(); fn(); } }, { threshold: 0.25 });
  io.observe(e);
}

/* ---------- Aralik cetveli: fiyat ekseni uzerinde %95 ve %80 bantlari ---------- */
function cetvel(kutu, olcekli) {
  const eksen = el("div", { class: "c-eksen" });
  const b95 = el("div", { class: "c-bant c95" }), b80 = el("div", { class: "c-bant c80" });
  const isaret = el("div", { class: "c-isaret" }, el("span", { class: "c-etiket" }));
  const gercek = el("div", { class: "c-gercek gizli" }, el("span", { class: "c-etiket" }));
  const altE = el("span", { class: "c-sinir s-alt" }), ustE = el("span", { class: "c-sinir s-ust" });
  eksen.append(b95, b80, isaret, gercek, altE, ustE);
  const olcek = el("div", { class: "c-olcek", "aria-hidden": "true" });
  kutu.replaceChildren(eksen, olcek);
  kutu.setAttribute("role", "img");
  let alan = [0, 1];
  const yer = v => `${(100 * (v - alan[0]) / (alan[1] - alan[0])).toFixed(3)}%`;
  const bant = (e, a, b) => { e.style.left = yer(a); e.style.width = `${(100 * (b - a) / (alan[1] - alan[0])).toFixed(3)}%`; };
  function olcekCiz() {
    const span = alan[1] - alan[0];
    const adimlar = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000];
    const adim = adimlar.find(s => span / s <= 6) || 5000;
    const cizgiler = [];
    for (let v = Math.ceil(alan[0] / adim) * adim; v <= alan[1]; v += adim) {
      const t = el("span", { class: "c-cizgi" }, sayi(v));
      t.style.left = yer(v);
      cizgiler.push(t);
    }
    olcek.replaceChildren(...cizgiler);
  }
  async function ciz(v) {
    alan = v.alan;
    const secili = v.duzey === 95 ? [v.lo95, v.hi95] : [v.lo80, v.hi80];
    kutu.classList.toggle("s95", v.duzey === 95);
    kutu.setAttribute("aria-label", v.etiket || "");
    if (v.ilk && !AZ_HAREKET) {                                     // ilk acilista bantlar merkezden acilir
      const m = (v.lo80 + v.hi80) / 2;
      [b95, b80].forEach(b => bant(b, m, m));
      altE.classList.add("gizli"); ustE.classList.add("gizli");
      await sonraki();
    }
    bant(b95, v.lo95, v.hi95);
    bant(b80, v.lo80, v.hi80);
    altE.textContent = sayi(secili[0]); ustE.textContent = sayi(secili[1]);
    altE.style.left = yer(secili[0]); ustE.style.left = yer(secili[1]);
    altE.classList.toggle("gizli", !!v.etiketsiz); ustE.classList.toggle("gizli", !!v.etiketsiz);
    if (v.isaret != null) {
      isaret.classList.remove("gizli");
      isaret.style.left = yer(v.isaret);
      isaret.firstChild.textContent = v.isaretYazi || "";
    } else isaret.classList.add("gizli");
    if (v.gercek != null) {
      gercek.classList.add("gizli");
      gercek.classList.remove("ic80", "ic95", "dis");
      gercek.style.left = yer(v.gercek);
      gercek.firstChild.textContent = v.gercekYazi || "";
      await sonraki();
      gercek.classList.add(icinde(v.gercek, v.lo80, v.hi80) ? "ic80" : (icinde(v.gercek, v.lo95, v.hi95) ? "ic95" : "dis"));
      gercek.classList.remove("gizli");
    } else gercek.classList.add("gizli");
    if (olcekli) olcekCiz();
  }
  return { ciz };
}

/* ---------- 1. Bugunku tahmin ---------- */
const H = { ufuk: 1, duzey: 80, cet: null, ilk: true };

function heroAlan(u) {
  const lo = Math.min(u.son_kapanis, ...u.ufuklar.map(f => f.aralik95[0]));
  const hi = Math.max(u.son_kapanis, ...u.ufuklar.map(f => f.aralik95[1]));
  const pad = (hi - lo) * 0.3;
  return [lo - pad, hi + pad];
}

function heroCiz() {
  const u = VERI.u;
  const f = u.ufuklar.find(x => x.ufuk_gun === H.ufuk) || u.ufuklar[0];
  const [lo, hi] = H.duzey === 80 ? f.aralik80 : f.aralik95;
  const ne = `${ufukAdi(f.ufuk_gun)} (${kisa(f.hedef_gun_yaklasik)})`;
  H.cet.ciz({ alan: heroAlan(u), lo80: f.aralik80[0], hi80: f.aralik80[1], lo95: f.aralik95[0], hi95: f.aralik95[1],
    duzey: H.duzey, isaret: u.son_kapanis, isaretYazi: `son kapanış ${sayi(u.son_kapanis)}`, ilk: H.ilk,
    etiket: `${ne}: %${H.duzey} olasılıkla ${sayi(lo)} ile ${sayi(hi)} TL arası. Son kapanış ${sayi(u.son_kapanis)} TL.` });
  H.ilk = false;
  const ba = document.getElementById("buyuk-aralik");
  if (!H.alt) {
    H.alt = el("span", { class: "sayi" }); H.ust = el("span", { class: "sayi" });
    ba.replaceChildren(H.alt, el("span", { class: "tire" }, "–"), H.ust, el("span", { class: "tl" }, "TL"));
  }
  sayiAkit(H.alt, H.onceki ? H.onceki[0] : null, lo);
  sayiAkit(H.ust, H.onceki ? H.onceki[1] : null, hi);
  H.onceki = [lo, hi];
  document.getElementById("cumle").textContent =
    `${ne} fiyatın bu aralıkta olma olasılığı %${H.duzey}. Yani ${kac(H.duzey)} fiyat bu aralıkta kalır.`;
  const fin = f.final ? (H.duzey === 80 ? f.final.tutma80 : f.final.tutma95) : null;
  const m = MERKEZ[f.ufuk_gun];
  const satir = (dt, dd, alt) => [el("div", {}, el("dt", {}, dt), el("dd", {}, dd, alt ? el("small", {}, alt) : ""))];
  document.getElementById("olgular").replaceChildren(
    ...satir("Aralığın genişliği", `${sayi(hi - lo)} TL`, `alt ve üst sınır arası`),
    ...satir("Merkez tahmin", `${sayi(merkezTahmin(f))} TL`, m ? `tek fiyat; ortalama sapma ±${sayi(m.hata_tl)} TL` : "aralığın tam ortası"),
    ...(fin != null ? satir("Geçmişte ne kadar tuttu?", yuzde(fin), `hiç görmediği 2023–2026’da, ${sayi(f.final.gun)} gün`) : []),
    ...satir("Model", ad(f.kullanilan), f.yedek ? `asıl model ${ad(f.model)}; Chronos bu bilgisayarda yok, yedek kullanıldı` : "yarışmanın kazananı"));
  document.querySelectorAll("#ufuk-sec button").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.ufuk) === H.ufuk)));
  document.querySelectorAll("#duzey-sec button").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.duzey) === H.duzey)));
}

function yelpaze(u, gecmis, W) {
  const H2 = 300, L = 66, R = W > 600 ? 70 : 52, T = 14, B = 30;
  const son = gecmis.slice(-30);
  const n = son.length, maxh = Math.max(...u.ufuklar.map(x => x.ufuk_gun));
  const Xs = L + (W > 600 ? 0.6 : 0.45) * (W - L - R);
  const xi = i => (i <= n - 1 ? L + (i / (n - 1)) * (Xs - L) : Xs + ((i - (n - 1)) / maxh) * (W - R - Xs));
  const lo = Math.min(...son.map(r => r.gercek), ...u.ufuklar.map(x => x.aralik95[0]));
  const hi = Math.max(...son.map(r => r.gercek), ...u.ufuklar.map(x => x.aralik95[1]));
  const pad = (hi - lo) * 0.06;
  const y = v => T + (1 - (v - (lo - pad)) / ((hi + pad) - (lo - pad))) * (H2 - T - B);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H2}`, role: "img",
    "aria-label": "Son 30 işlem gününün kapanışları ve 1, 3, 5 gün sonrası için tahmin aralıkları" });
  const x0 = xi(n - 1), P = u.son_kapanis;
  const nokta = k => {
    const uc = [[x0, y(P)]].concat(u.ufuklar.map(f => [xi(n - 1 + f.ufuk_gun), y(f[k][1])]));
    const ac = u.ufuklar.slice().reverse().map(f => [xi(n - 1 + f.ufuk_gun), y(f[k][0])]);
    return uc.concat(ac).map(p => p.join(",")).join(" ");
  };
  s.append(svg("polygon", { points: nokta("aralik95"), class: "f95" }));
  s.append(svg("polygon", { points: nokta("aralik80"), class: "f80" }));
  s.append(svg("polyline", { points: son.map((r, i) => `${xi(i)},${y(r.gercek)}`).join(" "), class: "fiyat-cizgi" }));
  s.append(svg("circle", { cx: x0, cy: y(P), r: 4, class: "fiyat-nokta" }));
  s.append(svg("line", { x1: x0, x2: x0, y1: T, y2: H2 - B, class: "kesik" }));
  s.append(svg("text", { x: x0 - 6, y: T + 10, "text-anchor": "end" }, "bugüne kadar"));
  s.append(svg("text", { x: x0 + 6, y: T + 10 }, "tahmin"));
  u.ufuklar.forEach((f, k) => {
    const xf = xi(n - 1 + f.ufuk_gun);
    s.append(svg("line", { x1: xf, x2: xf, y1: y(f.aralik95[1]), y2: y(f.aralik95[0]), class: "ince" }));
    s.append(svg("text", { x: xf, y: H2 - 8, "text-anchor": k === u.ufuklar.length - 1 ? "end" : "middle" },
                 W > 600 ? `+${f.ufuk_gun} gün` : `+${f.ufuk_gun}`));
  });
  const fs = u.ufuklar[u.ufuklar.length - 1], xs = xi(n - 1 + fs.ufuk_gun);
  s.append(svg("text", { x: xs + 6, y: y(fs.aralik80[1]) + 4, class: "vurgu" }, sayi(fs.aralik80[1])));
  s.append(svg("text", { x: xs + 6, y: y(fs.aralik80[0]) + 4, class: "vurgu" }, sayi(fs.aralik80[0])));
  s.append(svg("text", { x: xi(0), y: H2 - 8 }, kisa(son[0].tarih)));
  s.append(svg("text", { x: L - 8, y: y(hi) + 4, "text-anchor": "end" }, `${sayi(hi)} TL`));
  s.append(svg("text", { x: L - 8, y: y(lo) + 4, "text-anchor": "end" }, `${sayi(lo)} TL`));
  return s;
}

function lejant(...ogeler) {
  return el("div", { class: "lejant" }, ...ogeler.map(([c, t]) => el("span", { class: c }, t)));
}

async function tahmin() {
  try {
    const u = await al("/api/tahmin/ufuklar");
    VERI.u = u;
    document.getElementById("ust").textContent =
      `En son yayımlanan fiyat: ${tarih(u.bilgi_gunu)} kapanışı, ${sayi(u.son_kapanis, 2)} TL. TCMB fiyatı bir sonraki iş günü yayımlar; ` +
      "tahminler bu fiyattan sonraki işlem günleri için. Hafta sonu ve resmî tatiller sayılmaz.";
    const us = document.getElementById("ufuk-sec");
    us.replaceChildren(...u.ufuklar.map(f => {
      const b = el("button", { type: "button", "data-ufuk": String(f.ufuk_gun), "aria-pressed": "false" },
        `${f.ufuk_gun} gün sonra`, el("small", {}, kisa(f.hedef_gun_yaklasik)));
      b.addEventListener("click", () => { H.ufuk = f.ufuk_gun; heroCiz(); });
      return b;
    }));
    H.cet = cetvel(document.getElementById("cetvel"), true);
    heroCiz();
    const f1 = u.ufuklar[0];
    document.getElementById("hava-ornek").textContent =
      `“${kisaDe(f1.hedef_gun_yaklasik)} %80 olasılıkla ${sayi(f1.aralik80[0])} TL ile ${sayi(f1.aralik80[1])} TL arasında olacak” diyoruz.`;
    const hk = document.getElementById("hareket");
    if (u.hareketli_olasiligi_pct != null) {
      const p = u.hareketli_olasiligi_pct;
      hk.className = p >= 50 ? "not uyari-kutu" : "not";
      hk.textContent = (p >= 50 ? "Hareketli gün uyarısı: " : "") +
        `Bir sonraki işlem gününde fiyatın %${sayi(u.esik_pct, 0)}'den fazla değişme olasılığı ${yuzde(p, 0)}. ` +
        "Bu bir yatırım tavsiyesi değil, istatistiksel bir risk aralığıdır.";
    }
    if (VERI.g && VERI.g.length > 2) yelpazeCiz();
  } catch (e) {
    document.getElementById("ust").textContent = "Tahmin şu an hesaplanamıyor.";
    document.getElementById("cetvel").replaceChildren(el("p", { class: "bos" }, `Güncel fiyat verisi ve yarışma sonucu gerekiyor (${e.message}).`));
  }
}
function yelpazeCiz() {
  document.getElementById("yelpaze").replaceChildren(yelpaze(VERI.u, VERI.g, genislik("yelpaze")),
    lejant(["lg", "gerçekleşen kapanış"], ["l80", "%80 aralık"], ["l95", "%95 aralık"]));
}

/* ---------- 2. Aralik ne demek ---------- */
function noktalar() {
  const doldur = (kutu, n, disarida) => {
    const k = kutu.querySelector(".noktalar");
    k.replaceChildren(...Array.from({ length: n }, (_, i) => {
      const d = el("span", { class: disarida.includes(i) ? "nk bos-nk" : "nk" });
      d.style.transitionDelay = `${Math.round(i * (n > 10 ? 12 : 90))}ms`;
      return d;
    }));
    gorununce(kutu, () => k.classList.add("dolu"));
  };
  doldur(document.getElementById("nokta80"), 10, [3, 8]);
  doldur(document.getElementById("nokta95"), 100, [7, 33, 58, 71, 94]);
}

function sakinCalkanti() {
  const u1 = VERI.y.ufuklar[0], P = VERI.u ? VERI.u.son_kapanis : null;
  const sec = u1.yillar.filter(r => r.yil < 2023);
  const tum = u1.yillar;
  const sakin = sec.reduce((a, b) => (b.genislik80 < a.genislik80 ? b : a));
  const calk = tum.reduce((a, b) => (b.genislik80 > a.genislik80 ? b : a));
  const enGenis = calk.genislik80;
  const satir = (r, ad2) => {
    const bar = el("div", { class: "sc-bar" });
    const ic = el("div", { class: "sc-ic" }, el("span", { class: "sc-nokta" }));
    bar.append(ic);
    gorununce(bar, () => { ic.style.width = `${(100 * r.genislik80 / enGenis).toFixed(1)}%`; });
    return el("div", { class: "sc-satir" },
      el("p", { class: "sc-ad" }, `${ad2} (${r.yil})`), bar,
      el("p", { class: "sc-deger" }, `${yuzde(r.genislik80)} genişlik`,
         el("small", {}, P ? `bugünkü fiyatla yaklaşık ${sayi(P * r.genislik80 / 100)} TL` : "fiyata oranla")));
  };
  document.getElementById("sakin-calkanti").replaceChildren(
    satir(sakin, "En sakin yıl"), satir(calk, "En çalkantılı yıl"),
    el("p", { class: "not" }, "Bir sonraki gün için %80 aralığın o yıl ortalama genişliği, fiyata oranla. Aynı sistem, çalkantılı yılda yaklaşık iki kat geniş aralık verdi."));
}

function darGenis() {
  const P = VERI.u.son_kapanis, kutu = document.getElementById("dar-genis");
  const satirlar = VERI.y.ufuklar.filter(u => u.final && u.sabit && u.sabit.final).map(u => {
    const biz = P * u.final.genislik80 / 100, sabit = P * u.sabit.final.genislik80 / 100;
    const mx = P * Math.max(...VERI.y.ufuklar.filter(x => x.sabit && x.sabit.final).map(x => x.sabit.final.genislik80)) / 100;
    const cubuk = (deger, sinif, yazi) => {
      const ic = el("div", { class: `dg-ic ${sinif}` });
      const b = el("div", { class: "dg-bar" }, ic);
      gorununce(b, () => { ic.style.width = `${(100 * deger / mx).toFixed(1)}%`; });
      return el("div", { class: "dg-cubuk" }, el("span", { class: "dg-ad" }, yazi), b, el("span", { class: "dg-tl" }, `${sayi(deger)} TL`));
    };
    return el("div", { class: "dg-grup" },
      el("p", { class: "dg-baslik" }, u.ufuk_gun === 1 ? "1 gün sonrası" : `${u.ufuk_gun} gün sonrası`,
         el("span", { class: "dg-fark" }, `${yuzde(100 * (1 - biz / sabit), 0)} daha dar`)),
      cubuk(biz, "biz", "Bu sistem"), cubuk(sabit, "sabit", "Sabit aralık"));
  });
  kutu.replaceChildren(...satirlar);
  const u1 = VERI.y.ufuklar[0];
  if (u1.sabit && u1.sabit.final) document.getElementById("dar-genis-not").textContent =
    `%80 aralığın ortalama genişliği, hiç görülmemiş 2023–2026 döneminde, bugünkü ${sayi(P)} TL fiyata çevrilmiş hâli. ` +
    `Bir sonraki gün için sistemin %80 aralığı ${yuzde(u1.final.tutma80)} tuttu; sabit aralık ise ${yuzde(u1.sabit.final.tutma80)} tuttu. ` +
    "Sabit aralık %80 dediği hâlde daha sık tutuyor, çünkü gereğinden geniş. " +
    (() => {
      const f = VERI.u.ufuklar[0], w = f.aralik80[1] - f.aralik80[0], ort = P * u1.final.genislik80 / 100;
      return `Bugünkü 1 günlük %80 aralık ${sayi(w)} TL; bu, ${sayi(ort)} TL’lik ortalamanın ${w < ort ? "altında, çünkü son günler ortalamadan sakin" : "üstünde, çünkü son günler ortalamadan hareketli"}.`;
    })();
  const m = MERKEZ[1];
  if (m) document.getElementById("merkez-aciklama").textContent =
    `Evet. Aralığın tam ortasına merkez tahmin diyoruz. 2023–2026’da bir sonraki gün için merkez tahmin ortalama ${sayi(m.hata_tl)} TL saptı. ` +
    `“Yarın = bugün” demek ise ortalama ${sayi(m.rw_hata_tl)} TL sapıyor. Fark çok küçük, çünkü altının yönü tahmin edilemiyor. Bu yüzden asıl bilgi aralıktır.`;
}

/* ---------- 3. Tuttu mu ---------- */
function sinav() {
  const d = VERI.y;
  document.getElementById("guven-ozet").textContent =
    "Modeller hazırlık döneminde (2012–2022) denendi ve en iyisi seçildi. Sonra sınav döneminde (2023–2026), bu yılları hiç görmeden, her gün yalnızca o güne kadarki fiyatlarla sınandılar. " +
    "Hedef: %80 aralığın yaklaşık %80, %95 aralığın yaklaşık %95 tutması.";
  document.getElementById("guven").replaceChildren(...d.ufuklar.filter(u => u.final).map(u => {
    const a = (u.anlamlilik || {}).final;
    return el("div", { class: "blok" },
      el("p", { class: "baslik" }, u.ufuk_gun === 1 ? "1 gün sonrası" : `${u.ufuk_gun} gün sonrası`),
      el("div", { class: "cift" },
        el("p", {}, el("span", { class: "buyuk" }, yuzde(u.final.tutma80)), el("span", { class: "alt" }, "%80 aralık tuttu")),
        el("p", {}, el("span", { class: "buyuk" }, yuzde(u.final.tutma95)), el("span", { class: "alt" }, "%95 aralık tuttu"))),
      el("p", { class: "satir yorum" }, yorum80(u.final.tutma80)),
      el("p", { class: "satir" }, `${sayi(u.final.gun)} gün sınandı.`),
      a ? el("p", { class: "satir" }, `Basit yönteme (sabit aralık) göre ${yuzde(a.iyilesme_pct)} daha iyi. ${sans(a.p_degeri)}.`) : "");
  }));
}

const T = { gunler: [], i: 0, oyn: false, cet: null, iptal: 0 };

function tekrarGoster(i) {
  const r = T.gunler[i];
  T.i = i;
  const m = (r.lo80 + r.hi80) / 2, yari = Math.max((r.hi95 - r.lo95) / 2 * 1.5, Math.abs(r.gercek - m) * 1.25);
  const ok80 = icinde(r.gercek, r.lo80, r.hi80), ok95 = icinde(r.gercek, r.lo95, r.hi95);
  const sonuc = ok80 ? "tuttu" : (ok95 ? "%80 aralık tutmadı, %95 aralık tuttu" : "tutmadı");
  T.cet.ciz({ alan: [m - yari, m + yari], lo80: r.lo80, hi80: r.hi80, lo95: r.lo95, hi95: r.hi95, duzey: 80,
    gercek: r.gercek, gercekYazi: `gerçekleşen ${sayi(r.gercek)}`,
    etiket: `${tarih(r.tarih)}: %80 aralık ${sayi(r.lo80)}–${sayi(r.hi80)} TL, gerçekleşen ${sayi(r.gercek)} TL, ${sonuc}.` });
  document.getElementById("tekrar-yazi").replaceChildren(
    el("b", {}, tarih(r.tarih)), ` için bir gün önce verilen %80 aralık ${sayi(r.lo80)} – ${sayi(r.hi80)} TL. Gerçekleşen kapanış ${sayi(r.gercek)} TL: `,
    el("span", { class: ok80 ? "iyi" : (ok95 ? "orta" : "kotu") }, sonuc), ".");
  const seritler = document.querySelectorAll("#tekrar-seritler button");
  let tut = 0;
  seritler.forEach((b, k) => {
    const rr = T.gunler[k], acik = k <= i;
    b.classList.toggle("acik", acik);
    b.classList.toggle("secili", k === i);
    b.setAttribute("aria-current", k === i ? "true" : "false");
    if (acik && icinde(rr.gercek, rr.lo80, rr.hi80)) tut += 1;
  });
  document.getElementById("tekrar-sayac").textContent = `Gösterilen: ${i + 1} / ${T.gunler.length} gün. %80 aralık: ${tut} gün tuttu, ${i + 1 - tut} gün tutmadı.`;
}

async function oynat() {
  const dugme = document.getElementById("oynat");
  if (T.oyn) { T.oyn = false; T.iptal += 1; dugme.textContent = "Oynat"; return; }
  T.oyn = true; const benim = ++T.iptal; dugme.textContent = "Durdur";
  for (let k = 0; k < T.gunler.length; k++) {
    if (benim !== T.iptal) return;
    tekrarGoster(k);
    await bekle(1100);
  }
  T.oyn = false; dugme.textContent = "Baştan oynat";
}

function tekrarKur(g) {
  T.gunler = g.slice(-15);
  if (!T.gunler.length) return;
  T.cet = cetvel(document.getElementById("tekrar-cetvel"), false);
  document.getElementById("tekrar-seritler").replaceChildren(...T.gunler.map((r, k) => {
    const ok80 = icinde(r.gercek, r.lo80, r.hi80), ok95 = icinde(r.gercek, r.lo95, r.hi95);
    const b = el("button", { type: "button", class: `serit-gun ${ok80 ? "iyi" : (ok95 ? "orta" : "kotu")}`,
      "aria-label": `${tarih(r.tarih)}: ${ok80 ? "tuttu" : "tutmadı"}` }, el("span", { class: "sg-tarih" }, kisa(r.tarih).split(" ")[0]));
    b.addEventListener("click", () => { if (T.oyn) oynat(); tekrarGoster(k); });
    return b;
  }));
  tekrarGoster(T.gunler.length - 1);
}

function tablolar(g) {
  const son = g.slice(-15).reverse();
  const t80 = son.filter(r => icinde(r.gercek, r.lo80, r.hi80)).length;
  const tbl = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Gün"), el("th", {}, "%80 aralık (TL)"),
    el("th", {}, "%95 aralık (TL)"), el("th", {}, "Gerçekleşen"), el("th", {}, "Sonuç"))));
  const gov = el("tbody");
  son.forEach(r => {
    const ok = icinde(r.gercek, r.lo80, r.hi80), ok95 = icinde(r.gercek, r.lo95, r.hi95);
    gov.append(el("tr", {}, el("td", {}, kisa(r.tarih)), el("td", {}, `${sayi(r.lo80)} – ${sayi(r.hi80)}`),
      el("td", {}, `${sayi(r.lo95)} – ${sayi(r.hi95)}`), el("td", {}, sayi(r.gercek)),
      el("td", { class: ok ? "tuttu" : "tutmadi" }, ok ? "tuttu" : (ok95 ? "tutmadı (%95 tuttu)" : "tutmadı"))));
  });
  tbl.append(gov);
  document.getElementById("panel-aralik").replaceChildren(
    el("p", { class: "giris" }, `${son.length} günün ${t80} tanesinde fiyat %80 aralığın içinde kaldı. ` +
      "Bu sonuçlar sonradan hesaplandı, ama her gün için model yalnızca o güne kadarki fiyatları gördü."), tbl);
  const sapmalar = son.map(r => r.gercek - (r.lo80 + r.hi80) / 2);
  const ort = sapmalar.reduce((a, b) => a + Math.abs(b), 0) / Math.max(sapmalar.length, 1);
  const mt = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Gün"), el("th", {}, "Merkez tahmin"),
    el("th", {}, "Gerçekleşen"), el("th", {}, "Sapma"))));
  const mg = el("tbody");
  son.forEach((r, k) => {
    const s = sapmalar[k], yon = Math.round(s) === 0 ? "tam isabet" : (s > 0 ? `${sayi(Math.abs(s))} TL eksik` : `${sayi(Math.abs(s))} TL fazla`);
    mg.append(el("tr", {}, el("td", {}, kisa(r.tarih)), el("td", {}, `${sayi((r.lo80 + r.hi80) / 2)} TL`),
      el("td", {}, `${sayi(r.gercek)} TL`), el("td", { class: "sapma" }, yon)));
  });
  mt.append(mg);
  const m = MERKEZ[1];
  document.getElementById("panel-merkez").replaceChildren(
    el("p", { class: "giris" }, `Merkez tahmin, %80 aralığın tam ortasıdır. “Eksik”: tahmin gerçekleşenden düşük kaldı. “Fazla”: tahmin gerçekleşenden yüksek kaldı. ` +
      `Bu ${son.length} günde ortalama sapma ${sayi(ort)} TL.` + (m ? ` 2023–2026 ortalaması ±${sayi(m.hata_tl)} TL.` : "")), mt,
    el("p", { class: "not" }, "Tek fiyat tahmini her gün bir miktar sapar; altının yönü tahmin edilemediği için bu beklenen bir durumdur. Sistemin asıl iddiası aralıktır."));
}

function sekmeler() {
  const s = [["sekme-aralik", "panel-aralik"], ["sekme-merkez", "panel-merkez"]];
  const sec = id => s.forEach(([t, p]) => {
    const on = t === id, b = document.getElementById(t);
    b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1;
    document.getElementById(p).hidden = !on;
  });
  s.forEach(([t], k) => {
    const b = document.getElementById(t);
    b.addEventListener("click", () => sec(t));
    b.addEventListener("keydown", e => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const n = s[(k + (e.key === "ArrowRight" ? 1 : s.length - 1)) % s.length][0];
        sec(n); document.getElementById(n).focus();
      }
    });
  });
}

async function buYil() {
  const ozet = document.getElementById("buyil-ozet");
  try {
    const d = await al("/api/tahmin/bu-yil-yarisma");
    const g = d.gunler;
    VERI.g = g;
    if (!g.length) { ozet.textContent = `${d.yil} için henüz sınanmış gün yok.`; return g; }
    ozet.textContent = `${d.yil} yılında şimdiye kadar ${d.gun} işlem gününün ${d.aralik80_tuttu} tanesinde fiyat, bir gün önceden ` +
      `verilen %80 aralığın içinde kaldı (${yuzde(d.aralik80_pct)}). %95 aralık ${yuzde(d.aralik95_pct)} tuttu. Kırmızı noktalar %80 aralığın dışında kalan günler.`;
    const W = genislik("buyil"), H2 = 250, L = 66, R = 8, Tp = 12, B = 28;
    const ymin = Math.min(...g.map(r => r.lo95)), ymax = Math.max(...g.map(r => r.hi95));
    const x = i => L + (i / Math.max(g.length - 1, 1)) * (W - L - R);
    const y = v => Tp + (1 - (v - ymin) / (ymax - ymin || 1)) * (H2 - Tp - B);
    const bant = (a, b) => g.map((r, i) => `${x(i)},${y(r[a])}`).join(" ") + " " +
      g.map((r, i) => [i, r]).reverse().map(([i, r]) => `${x(i)},${y(r[b])}`).join(" ");
    const s = svg("svg", { viewBox: `0 0 ${W} ${H2}`, role: "img", "aria-label": ozet.textContent });
    s.append(svg("polygon", { points: bant("hi95", "lo95"), class: "f95" }));
    s.append(svg("polygon", { points: bant("hi80", "lo80"), class: "f80" }));
    s.append(svg("polyline", { points: g.map((r, i) => `${x(i)},${y(r.gercek)}`).join(" "), class: "fiyat-cizgi ince-cizgi" }));
    g.forEach((r, i) => { if (!icinde(r.gercek, r.lo80, r.hi80)) s.append(svg("circle", { cx: x(i), cy: y(r.gercek), r: 2.6, class: "disari" })); });
    s.append(svg("line", { x1: L, x2: L, y1: Tp, y2: H2 - B, class: "eksen" }));
    s.append(svg("text", { x: L - 8, y: Tp + 4, "text-anchor": "end" }, `${sayi(ymax)} TL`));
    s.append(svg("text", { x: L - 8, y: H2 - B, "text-anchor": "end" }, `${sayi(ymin)} TL`));
    s.append(svg("text", { x: L, y: H2 - 6 }, kisa(g[0].tarih)));
    s.append(svg("text", { x: W - R, y: H2 - 6, "text-anchor": "end" }, kisa(g[g.length - 1].tarih)));
    document.getElementById("buyil").replaceChildren(s,
      lejant(["lg", "gerçekleşen kapanış"], ["l80", "%80 aralık"], ["l95", "%95 aralık"], ["ld", "%80 aralığın dışında kalan gün"]));
    tablolar(g);
    tekrarKur(g);
    return g;
  } catch (e) {
    ozet.textContent = `Bu yılın tahminleri hesaplanamadı (${e.message}).`;
    return [];
  }
}

async function canliKarne() {
  const ozet = document.getElementById("canli-ozet");
  try {
    const d = await al("/api/tahmin/canli-karne");
    if (!d.kayit) {
      ozet.textContent = "Bu karne, her sabah verilen 1, 3 ve 5 günlük aralıkları gerçek fiyat belli olmadan önce kaydeder ve " +
        "sonra gerçekleşen fiyatla puanlar. Henüz kayıt yok; ilk kayıt bir sonraki sabah görevinde yapılacak.";
      return;
    }
    ozet.textContent = "Yukarıdaki sonuçlar geriye dönük hesaplandı. Buradakiler ise gerçek: her sabah verilen aralık, fiyat belli olmadan önce " +
      `değiştirilemez biçimde kaydediliyor (${tarih(d.baslangic)} kapanışından beri). ` +
      (d.son_fiyat_gunu ? `En son yayımlanan kapanış ${tarih(d.son_fiyat_gunu)}. ` : "") +
      "“Bekliyor” yazan tahminler hata değil: hedef günün kapanışı yayımlanınca kendiliğinden puanlanır.";
    document.getElementById("canli").replaceChildren(...d.ufuklar.map(u => el("div", { class: "blok" },
      el("p", { class: "baslik" }, u.ufuk_gun === 1 ? "1 gün sonrası" : `${u.ufuk_gun} gün sonrası`),
      el("p", { class: "buyuk" }, u.puanlanan ? `${u.tuttu80}/${u.puanlanan}` : "—"),
      el("p", { class: "alt" }, u.puanlanan ? "%80 aralık tuttu" : "henüz sonuçlanan tahmin yok"),
      el("p", { class: "satir" }, u.puanlanan ? `%95 aralık: ${u.tuttu95}/${u.puanlanan} tuttu. ${u.kayit - u.puanlanan} tahmin bekliyor.` : `${u.kayit} tahmin bekliyor`))));
    const tbl = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Tahmin günü"), el("th", {}, "Ne kadar sonrası"),
      el("th", {}, "%80 aralık (TL)"), el("th", {}, "Gerçekleşen"), el("th", {}, "Sonuç"))));
    const gov = el("tbody");
    d.son_kayitlar.slice().reverse().forEach(r => gov.append(el("tr", {}, el("td", {}, kisa(r.bilgi_gunu)),
      el("td", {}, `${r.ufuk_gun} gün`), el("td", {}, `${sayi(r.lo80)} – ${sayi(r.hi80)}`),
      el("td", {}, r.gercek == null ? (r.beklenen_gun ? `bekliyor (${kisa(r.beklenen_gun)} kapanışı)` : "bekliyor") : sayi(r.gercek)),
      el("td", { class: r.in80 === true ? "tuttu" : (r.in80 === false ? "tutmadi" : "") },
         r.in80 == null ? "—" : (r.in80 ? "tuttu" : "tutmadı")))));
    tbl.append(gov);
    document.getElementById("canli-son").replaceChildren(tbl);
  } catch (e) {
    ozet.textContent = `İleriye dönük karne okunamadı (${e.message}).`;
  }
}

/* ---------- 4. Kriz yillari: genislik ve tutma, yil yil ---------- */
const KRIZ = { 2018: "kur krizi", 2020: "pandemi", 2021: "kur krizi", 2023: "fiyat sıçraması" };

function krizGrafik(rows, sabit, W) {
  const genis = W > 640, L = genis ? 46 : 34, R = 6, T = 26, H1 = 150, ARA = 34, H2b = 110, B = 24;
  const H2 = T + H1 + ARA + H2b + B;
  const bw = (W - L - R) / rows.length;
  const sb = Object.fromEntries((sabit || []).map(r => [r.yil, r]));
  const gmax = Math.max(...rows.map(r => r.genislik80), ...(sabit || []).map(r => r.genislik80)) * 1.08;
  const yg = v => T + H1 - (v / gmax) * H1;
  const y0 = T + H1 + ARA, yt = v => y0 + H2b - ((v - 60) / 40) * H2b;          // tutma: %60-100 arasi
  const s = svg("svg", { viewBox: `0 0 ${W} ${H2}`, role: "img",
    "aria-label": "Yıllara göre %80 aralığın genişliği ve tutma oranı; kriz yılları işaretli" });
  rows.forEach((r, i) => {
    if (!KRIZ[r.yil]) return;
    const x0 = L + i * bw;
    s.append(svg("rect", { x: x0, y: T - 4, width: bw, height: H2 - T - B + 4, class: "kriz-zemin" }));
    if (genis) s.append(svg("text", { x: x0 + bw / 2, y: T - 10, "text-anchor": "middle", class: "kriz-yazi" }, KRIZ[r.yil]));
  });
  s.append(svg("text", { x: L, y: 12, class: "panel-baslik" }, "%80 aralığın genişliği (fiyatın yüzdesi)"));
  [0, 2, 4, 6].filter(v => v < gmax).forEach(v => {
    s.append(svg("line", { x1: L, x2: W - R, y1: yg(v), y2: yg(v), class: "izgara" }));
    s.append(svg("text", { x: L - 6, y: yg(v) + 4, "text-anchor": "end" }, `%${v}`));
  });
  rows.forEach((r, i) => {
    const x0 = L + i * bw + bw * 0.2, w = bw * 0.6, fin = r.yil >= 2023;
    const rc = svg("rect", { x: x0, y: yg(r.genislik80), width: w, height: H1 - (yg(r.genislik80) - T), class: fin ? "cubuk-final" : "cubuk-secim" });
    rc.append(svg("title", {}, `${r.yil}: sistem ${yuzde(r.genislik80)}${sb[r.yil] ? `, sabit aralık ${yuzde(sb[r.yil].genislik80)}` : ""}`));
    s.append(rc);
    if (sb[r.yil]) s.append(svg("line", { x1: L + i * bw + bw * 0.08, x2: L + (i + 1) * bw - bw * 0.08,
      y1: yg(sb[r.yil].genislik80), y2: yg(sb[r.yil].genislik80), class: "sabit-cizgi" }));
  });
  s.append(svg("text", { x: L, y: y0 - 10, class: "panel-baslik" }, "%80 aralığın tuttuğu günler"));
  [60, 80, 100].forEach(v => {
    s.append(svg("line", { x1: L, x2: W - R, y1: yt(v), y2: yt(v), class: v === 80 ? "hedef-cizgi" : "izgara" }));
    s.append(svg("text", { x: L - 6, y: yt(v) + 4, "text-anchor": "end" }, `%${v}`));
  });
  const pts = rows.map((r, i) => [L + (i + 0.5) * bw, yt(Math.max(60, r.tutma80))]);
  s.append(svg("polyline", { points: pts.map(p => p.join(",")).join(" "), class: "tutma-cizgi" }));
  rows.forEach((r, i) => {
    const c = svg("circle", { cx: pts[i][0], cy: pts[i][1], r: genis ? 4 : 3, class: r.yil >= 2023 ? "nokta-final" : "nokta-secim" });
    c.append(svg("title", {}, `${r.yil}: %80 aralık ${yuzde(r.tutma80)} tuttu, %95 aralık ${yuzde(r.tutma95)}`));
    s.append(c);
    s.append(svg("text", { x: L + (i + 0.5) * bw, y: H2 - 6, "text-anchor": "middle", class: KRIZ[r.yil] ? "kriz-yil" : "" },
      genis ? String(r.yil) : String(r.yil).slice(2)));
  });
  return s;
}

function krizCiz() {
  const u1 = VERI.y.ufuklar[0];
  document.getElementById("yillar").replaceChildren(krizGrafik(u1.yillar, u1.sabit ? u1.sabit.yillar : null, genislik("yillar")),
    lejant(["lgr", "hazırlık dönemi 2012–2022"], ["lf", "sınav dönemi 2023–2026 (hiç görülmemiş)"], ["ls", "sabit aralık"], ["lk", "kriz yılı"]));
  document.getElementById("yillar-ozet").textContent =
    `Bir sonraki gün için, ${ad(u1.kazanan)} modeli. Üstte: aralık her yıl fiyatın yüzde kaçı kadar genişti (çubuk: bu sistem, çizgi: sabit aralık). ` +
    "Altta: %80 aralık o yıl günlerin yüzde kaçında tuttu (kesikli çizgi hedef %80).";
}

/* ---------- 5. Model yarismasi ---------- */
function yarismaCiz() {
  const d = VERI.y;
  document.getElementById("yarisma-ozet").textContent =
    `${d.model_sayisi} yöntem, hazırlık dönemi (2012–2022) verisiyle yarıştı. Her yöntem her ay yalnızca o güne kadarki fiyatlarla yeniden kuruldu ` +
    "ve 1, 3, 5 gün sonrası için aralık verdi. Sonra iki şeye bakıldı: aralık söylediği kadar sık tuttu mu, ve ne kadar dar? Kazananlar:";
  const kaz = {};
  d.ufuklar.forEach(u => { (kaz[u.kazanan] = kaz[u.kazanan] || []).push(u.ufuk_gun); });
  document.getElementById("kazananlar").replaceChildren(...Object.entries(kaz).map(([m, hs]) =>
    el("li", {}, el("b", {}, `${hs.join(" ve ")} gün için: ${ad(m)}.`), SADE[m] ? ` ${SADE[m][0].toUpperCase()}${SADE[m].slice(1)}.` : "")));
  document.getElementById("yarisma-aciklama").textContent =
    `Hedefe yakınlık: %80 aralık 80 ± ${sayi(d.hedef_sapma["80"], 0)}, %95 aralık 95 ± ${sayi(d.hedef_sapma["95"], 0)} arasında tutmalı. ` +
    "Satırlar 1 günlük puana göre sıralı. Kazananın sınav dönemi sonuçları “Tuttu mu?” bölümünde.";
  const u1 = d.ufuklar[0];
  const modeller = u1.modeller.map(r => r.model);
  const bul = (u, m) => u.modeller.find(r => r.model === m);
  const tbl = el("table", { class: "yarisma-tablo" }, el("thead", {},
    el("tr", {}, el("th", { rowspan: "2" }, "Model"),
      ...d.ufuklar.map(u => el("th", { colspan: "3", class: "ufuk-bas" }, u.ufuk_gun === 1 ? "1 gün sonrası" : `${u.ufuk_gun} gün sonrası`))),
    el("tr", {}, ...d.ufuklar.flatMap(() => [el("th", {}, "%80"), el("th", {}, "%95"), el("th", { class: "puan-bas" }, "puan")]))));
  const gov = el("tbody");
  modeller.forEach(m => {
    const kazandi = d.ufuklar.some(u => u.kazanan === m);
    gov.append(el("tr", { class: kazandi ? "kazanan-satir" : "" },
      el("td", { class: "model-ad" }, el("b", {}, ad(m)), NE[m] ? el("small", {}, NE[m]) : ""),
      ...d.ufuklar.flatMap(u => {
        const r = bul(u, m);
        if (!r) return [el("td", {}, "—"), el("td", {}, "—"), el("td", {}, "—")];
        const cls = u.kazanan === m ? "kazanan" : (r.uygun ? "" : "disarida");
        return [el("td", { class: cls }, sayi(r.tutma80, 1)), el("td", { class: cls }, sayi(r.tutma95, 1)),
                el("td", { class: `${cls} puan-h` }, el("span", { class: "puan" }, sayi(r.puan, 2)))];
      })));
  });
  tbl.append(gov);
  document.getElementById("yarisma").replaceChildren(tbl);
}

async function yarisma() {
  try {
    VERI.y = await al("/api/tahmin/yarisma");
    sinav(); yarismaCiz(); yarisKur(); krizCiz(); sakinCalkanti();
  } catch (e) {
    document.getElementById("guven-ozet").textContent = `Yarışma sonucu okunamadı (${e.message}).`;
  }
}

/* ---------- Nasil calisiyor: adim adim gosteri (gercek bugunku sayilarla) ---------- */
const G = { adim: 0, cet: null, zam: null, adimlar: [] };

function gosteriAdimlari() {
  const u = VERI.u, f = u.ufuklar[0], P = u.son_kapanis;
  const m = (f.aralik80[0] + f.aralik80[1]) / 2;
  const y80 = (f.aralik80[1] - f.aralik80[0]) / 2, y95 = (f.aralik95[1] - f.aralik95[0]) / 2;
  const temel = { alan: [m - y95 * 1.6, m + y95 * 1.6], isaret: P, isaretYazi: `son kapanış ${sayi(P)}` };
  const adimlar = [
    { yazi: `1. Başlangıç: en son kapanış fiyatı ${sayi(P)} TL (${kisa(u.bilgi_gunu)}).`,
      v: { ...temel, lo80: m, hi80: m, lo95: m, hi95: m, duzey: 80, etiketsiz: true } },
    { yazi: `2. Model, ${kisa(f.hedef_gun_yaklasik)} kapanışının ne kadar oynayabileceğini hesaplar ve %80 aralığı koyar: ortadan iki yana yaklaşık ±${sayi(y80)} TL.`,
      v: { ...temel, lo80: f.aralik80[0], hi80: f.aralik80[1], lo95: f.aralik80[0], hi95: f.aralik80[1], duzey: 80 } },
    { yazi: `3. Daha emin olmak için %95 aralık: iki yana yaklaşık ±${sayi(y95)} TL. Daha çok günü kapsaması gerektiği için daha geniş.`,
      v: { ...temel, lo80: f.aralik80[0], hi80: f.aralik80[1], lo95: f.aralik95[0], hi95: f.aralik95[1], duzey: 95 } },
  ];
  const g = VERI.g && VERI.g.length ? VERI.g[VERI.g.length - 1] : null;
  if (g) {
    const mg = (g.lo80 + g.hi80) / 2, yg = Math.max((g.hi95 - g.lo95) / 2 * 1.6, Math.abs(g.gercek - mg) * 1.25);
    const ok = icinde(g.gercek, g.lo80, g.hi80);
    adimlar.push({ yazi: `4. Gün bitince gerçek fiyat gelir ve aralığın içinde mi diye bakılır. Örnek: ${kisa(g.tarih)} için verilen %80 aralık ` +
        `${sayi(g.lo80)} – ${sayi(g.hi80)} TL, gerçekleşen ${sayi(g.gercek)} TL: ${ok ? "tuttu" : "tutmadı"}.`,
      v: { alan: [mg - yg, mg + yg], lo80: g.lo80, hi80: g.hi80, lo95: g.lo95, hi95: g.hi95, duzey: 80,
           gercek: g.gercek, gercekYazi: `gerçekleşen ${sayi(g.gercek)}` } });
  }
  return adimlar;
}
function gosteriDur() { if (G.zam) { clearInterval(G.zam); G.zam = null; } }
function gosteriGit(i) {
  G.adim = Math.max(0, Math.min(G.adimlar.length - 1, i));
  const a = G.adimlar[G.adim];
  G.cet.ciz({ ...a.v, etiket: a.yazi });
  document.getElementById("gosteri-adim").textContent = a.yazi;
  document.querySelectorAll("#gosteri-noktalar span").forEach((n, k) => n.classList.toggle("acik", k <= G.adim));
  document.getElementById("gosteri-geri").disabled = G.adim === 0;
  document.getElementById("gosteri-ileri").disabled = G.adim === G.adimlar.length - 1;
}
function gosteriKur() {
  G.adimlar = gosteriAdimlari();
  document.getElementById("gosteri-noktalar").replaceChildren(...G.adimlar.map(() => el("span")));
  if (!G.cet) {
    G.cet = cetvel(document.getElementById("gosteri-cetvel"), false);
    document.getElementById("gosteri-geri").addEventListener("click", () => { gosteriDur(); gosteriGit(G.adim - 1); });
    document.getElementById("gosteri-ileri").addEventListener("click", () => { gosteriDur(); gosteriGit(G.adim + 1); });
    gorununce(document.getElementById("gosteri"), () => {
      if (AZ_HAREKET) return;                                       // hareket azaltma: kendiliginden ilerlemez
      G.zam = setInterval(() => { if (G.adim >= G.adimlar.length - 1) gosteriDur(); else gosteriGit(G.adim + 1); }, 2800);
    });
  }
  gosteriGit(0);
}

/* ---------- Model yarismasi: kim kazandi? (cubuk = kazanandan ne kadar kotu) ---------- */
const Y = { ufuk: 1, goruldu: false };

function yarisKur() {
  document.getElementById("yaris-sec").replaceChildren(...VERI.y.ufuklar.map(u => {
    const b = el("button", { type: "button", "data-ufuk": String(u.ufuk_gun), "aria-pressed": "false" }, `${u.ufuk_gun} gün sonrası`);
    b.addEventListener("click", () => { Y.ufuk = u.ufuk_gun; yarisCiz(); });
    return b;
  }));
  yarisCiz();
}
function yarisCiz() {
  const d = VERI.y, u = d.ufuklar.find(x => x.ufuk_gun === Y.ufuk) || d.ufuklar[0];
  document.querySelectorAll("#yaris-sec button").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.ufuk) === u.ufuk_gun)));
  const kaz = u.modeller.find(r => r.model === u.kazanan);
  const sira = r => (r.model === u.kazanan ? 0 : r.uygun ? 1 : 2);                // kazanan, sonra yarisanlar, en altta elenenler
  const sirali = u.modeller.slice().sort((a, b) => sira(a) - sira(b) || a.puan - b.puan);
  const enKotu = Math.max(0.01, ...sirali.filter(r => r.uygun && r.model !== u.kazanan).map(r => r.puan / kaz.puan - 1));
  const h80 = d.hedef_sapma["80"];
  const kutu = document.getElementById("yaris-satirlar");
  kutu.replaceChildren(...sirali.map((r, k) => {
    const kazanan = r.model === u.kazanan, fark = r.puan / kaz.puan - 1;
    const ic = el("div", { class: "yr-ic" });
    ic.style.transitionDelay = `${k * 45}ms`;
    ic.dataset.hedef = kazanan ? "2" : (r.uygun ? (100 * fark / enKotu).toFixed(1) : "0");
    let yazi;
    if (kazanan) yazi = "● kazanan";
    else if (!r.uygun) yazi = `elendi: ${Math.abs(r.tutma80 - 80) > h80 ? `%80 aralık ${sayi(r.tutma80, 1)} tuttu` : `%95 aralık ${sayi(r.tutma95, 1)} tuttu`}`;
    else if (100 * fark < 0.5) yazi = "kazananla neredeyse aynı";
    else yazi = `kazanandan %${sayi(100 * fark, 1)} kötü`;
    return el("div", { class: `yr-satir${kazanan ? " kazanan" : ""}${r.uygun ? "" : " elendi"}` },
      el("span", { class: "yr-ad" }, ad(r.model)), el("div", { class: "yr-bar" }, ic), el("span", { class: "yr-deger" }, yazi));
  }));
  const uygula = () => sonraki().then(() => kutu.querySelectorAll(".yr-ic").forEach(i => { i.style.width = `${i.dataset.hedef}%`; }));
  if (Y.goruldu) uygula(); else gorununce(kutu, () => { Y.goruldu = true; uygula(); });
}

/* ---------- Ust menu: bulunulan bolumu isaretle ---------- */
function menu() {
  if (!("IntersectionObserver" in window)) return;
  const linkler = [...document.querySelectorAll("#menu a")];
  const io = new IntersectionObserver(es => {
    es.forEach(x => {
      if (!x.isIntersecting) return;
      linkler.forEach(a => a.classList.toggle("burada", a.getAttribute("href") === `#${x.target.id}`));
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("main > section").forEach(s => io.observe(s));
}

async function hepsi() {
  noktalar(); sekmeler(); menu();
  try { (await al("/api/tahmin/merkez")).ufuklar.forEach(m => { MERKEZ[m.ufuk_gun] = m; }); } catch (e) { MERKEZ = {}; }
  document.getElementById("oynat").addEventListener("click", oynat);
  document.querySelectorAll("#duzey-sec button").forEach(b => b.addEventListener("click", () => {
    H.duzey = Number(b.dataset.duzey); if (VERI.u) heroCiz();
  }));
  await Promise.all([buYil(), tahmin(), yarisma()]);
  if (VERI.u && VERI.g && VERI.g.length > 2) yelpazeCiz();
  if (VERI.u && VERI.y) { darGenis(); sakinCalkanti(); }
  if (VERI.u) gosteriKur();
  canliKarne();
}
hepsi();

let sonGenislik = window.innerWidth, zamanlayici = null;
window.addEventListener("resize", () => {
  clearTimeout(zamanlayici);
  zamanlayici = setTimeout(() => {
    if (Math.abs(window.innerWidth - sonGenislik) <= 40) return;
    sonGenislik = window.innerWidth;
    if (VERI.u && VERI.g && VERI.g.length > 2) yelpazeCiz();
    if (VERI.y) krizCiz();
  }, 250);
});
