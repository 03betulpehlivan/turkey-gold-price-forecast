"use strict";
const NS = "http://www.w3.org/2000/svg";
const AY = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"];
const MODEL_ADI = { "Yarin=bugun": "Sabit aralık", "GARCH": "Oynaklık modeli", "GARCH-t+ACI": "Bu sistem" };
const MODEL_ACIKLAMA = { "Yarin=bugun": "her gün aynı genişlik", "GARCH": "tek seferlik eğitim",
                         "GARCH-t+ACI": "pencere ve ayar ile" };

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const c of kids) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
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
const kisaTarih = iso => { const [, m, g] = iso.split("-").map(Number); return `${g} ${AY[m - 1]}`; };

function havaSimgesi(p) {
  const s = svg("svg", { viewBox: "0 0 64 64", role: "img", "aria-hidden": "true" });
  const gunes = (cx, cy, r) => {
    s.append(svg("circle", { cx, cy, r, fill: "var(--gunes)" }));
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      s.append(svg("line", { x1: cx + Math.cos(a) * (r + 4), y1: cy + Math.sin(a) * (r + 4), x2: cx + Math.cos(a) * (r + 9),
                             y2: cy + Math.sin(a) * (r + 9), stroke: "var(--gunes)", "stroke-width": 3, "stroke-linecap": "round" }));
    }
  };
  const bulut = (x, y, k, renk) => s.append(svg("path", {
    d: `M${x} ${y} a${8 * k} ${8 * k} 0 0 1 ${15 * k} -${4 * k} a${10 * k} ${10 * k} 0 0 1 ${19 * k} ${4 * k} a${6 * k} ${6 * k} 0 0 1 0 ${12 * k} h-${34 * k} a${6 * k} ${6 * k} 0 0 1 0 -${12 * k}z`,
    fill: renk }));
  if (p < 25) gunes(32, 32, 12);
  else if (p < 50) { gunes(24, 24, 10); bulut(22, 38, 1, "var(--soluk)"); }
  else {
    bulut(14, 26, 1.1, "var(--soluk)");
    s.append(svg("path", { d: "M34 42 l-7 11 h7 l-4 10 l11 -14 h-7 l4 -7z", fill: "var(--uyari)" }));
  }
  return s;
}
function havaMetni(p) {
  if (p < 25) return "Sakin bir gün bekleniyor";
  if (p < 50) return "Biraz hareketli olabilir";
  return "Hareketli bir gün bekleniyor";
}

const genislik = id => Math.max(300, Math.round(document.getElementById(id).clientWidth || 900));

function cetvel(k, W) {
  const H = 74, P = 10;
  const [l95, h95] = k.aralik95, [l80, h80] = k.aralik80;
  const pad = (h95 - l95) * 0.12, lo = l95 - pad, hi = h95 + pad;
  const x = v => P + ((v - lo) / (hi - lo)) * (W - 2 * P);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `%80 aralık ${sayi(l80)} ile ${sayi(h80)} TL, %95 aralık ${sayi(l95)} ile ${sayi(h95)} TL, son kapanış ${sayi(k.son_kapanis)} TL` });
  s.append(svg("rect", { x: x(l95), y: 22, width: x(h95) - x(l95), height: 22, rx: 11, fill: "var(--bant95)" }));
  s.append(svg("rect", { x: x(l80), y: 22, width: x(h80) - x(l80), height: 22, rx: 11, fill: "var(--bant80)" }));
  const xs = x(k.son_kapanis);
  s.append(svg("line", { x1: xs, x2: xs, y1: 14, y2: 52, stroke: "var(--murekkep)", "stroke-width": 2.5 }));
  s.append(svg("text", { x: xs, y: 9, "text-anchor": "middle", class: "vurgu" }, `son kapanış ${sayi(k.son_kapanis)}`));
  s.append(svg("text", { x: x(l95), y: 68, "text-anchor": "start" }, sayi(l95)));
  s.append(svg("text", { x: x(h95), y: 68, "text-anchor": "end" }, sayi(h95)));
  s.append(svg("text", { x: x(l80) + 4, y: 68, "text-anchor": "start", class: "vurgu" }, sayi(l80)));
  s.append(svg("text", { x: x(h80) - 4, y: 68, "text-anchor": "end", class: "vurgu" }, sayi(h80)));
  return s;
}

async function tahmin() {
  const kutu = document.getElementById("tahmin");
  try {
    const k = await al("/api/tahmin/yarin");
    document.getElementById("ust").textContent = `${tarih(k.bilgi_gunu)} kapanışına göre, bir sonraki işlem günü için`;
    const p = k.hareketli_olasiligi_pct;
    const sol = el("div", {},
      el("p", { class: "etiket" }, "%80 olasılıkla"),
      el("p", { class: "aralik" }, `${sayi(k.aralik80[0])} – ${sayi(k.aralik80[1])}`, el("span", { class: "tl" }, "TL")),
      el("p", { class: "ikinci" }, "%95 olasılıkla ", el("b", {}, `${sayi(k.aralik95[0])} – ${sayi(k.aralik95[1])} TL`)),
      el("p", { class: "ikinci" }, "Son kapanış ", el("b", {}, `${sayi(k.son_kapanis, 2)} TL`), ` (${tarih(k.bilgi_gunu)})`));
    const sag = el("div", { class: "hava" }, havaSimgesi(p), el("div", { class: "olasilik" }, yuzde(p, 0)),
      el("div", { class: "ne" }, havaMetni(p)),
      el("div", { class: "tanim" }, `Hareketli gün olasılığı: fiyatın bir günde %${sayi(k.esik_pct, 0)}'den fazla değişme olasılığı.`));
    kutu.replaceChildren(el("div", { class: "tahmin-ust" }, sol, sag), el("div", { class: "cetvel", id: "cetvel" }),
      el("p", { class: "not" }, "Koyu bant %80, açık bant %95 olasılık aralığı. Hafta sonu ve resmî tatiller işlem günü sayılmaz."));
    document.getElementById("cetvel").append(cetvel(k, genislik("cetvel")));
  } catch (e) {
    document.getElementById("ust").textContent = "Yarın tahmini şu an hesaplanamıyor.";
    kutu.replaceChildren(el("p", { class: "bos" }, `Tahmin için güncel fiyat verisi gerekiyor (${e.message}).`));
  }
}

async function buYil() {
  const kutu = document.getElementById("buyil"), ozet = document.getElementById("buyil-ozet");
  try {
    const d = await al("/api/tahmin/bu-yil");
    const g = d.gunler;
    if (!g.length) { ozet.textContent = `${d.yil} için henüz sınanmış gün yok.`; return; }
    ozet.textContent = `${d.yil} yılında şimdiye kadar ${d.gun} işlem gününün ${d.aralik80_tuttu}'inde fiyat %80'lik aralığın ` +
      `içinde kaldı (tutma oranı ${yuzde(d.aralik80_pct)}). %95'lik aralığın tutma oranı ${yuzde(d.aralik95_pct)}. ` +
      "Her gün, yalnızca o güne kadar bilinen fiyatlarla tahmin edildi.";
    const W = genislik("buyil"), H = 260, L = 74, R = 8, T = 12, B = 28;
    const ymin = Math.min(...g.map(r => r.lo95)), ymax = Math.max(...g.map(r => r.hi95));
    const x = i => L + (i / Math.max(g.length - 1, 1)) * (W - L - R);
    const y = v => T + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - T - B);
    const bant = (a, b) => g.map((r, i) => `${x(i)},${y(r[a])}`).join(" ") + " " + g.map((r, i) => [i, r]).reverse().map(([i, r]) => `${x(i)},${y(r[b])}`).join(" ");
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ozet.textContent });
    s.append(svg("polygon", { points: bant("hi95", "lo95"), fill: "var(--bant95)" }));
    s.append(svg("polygon", { points: bant("hi80", "lo80"), fill: "var(--bant80)" }));
    s.append(svg("polyline", { points: g.map((r, i) => `${x(i)},${y(r.gercek)}`).join(" "), fill: "none", stroke: "var(--murekkep)", "stroke-width": 1.6 }));
    g.forEach((r, i) => { if (r.gercek < r.lo80 || r.gercek > r.hi80) s.append(svg("circle", { cx: x(i), cy: y(r.gercek), r: 2.6, fill: "var(--uyari)" })); });
    s.append(svg("text", { x: L, y: H - 6 }, kisaTarih(g[0].tarih)));
    s.append(svg("text", { x: W - R, y: H - 6, "text-anchor": "end" }, kisaTarih(g[g.length - 1].tarih)));
    s.append(svg("text", { x: L - 8, y: T + 4, "text-anchor": "end" }, `${sayi(ymax)} TL`));
    s.append(svg("text", { x: L - 8, y: H - B, "text-anchor": "end" }, `${sayi(ymin)} TL`));
    s.append(svg("line", { x1: L, x2: L, y1: T, y2: H - B, stroke: "var(--cizgi)", "stroke-width": 1 }));
    kutu.replaceChildren(s, el("div", { class: "lejant" }, el("span", { class: "lg" }, "gerçekleşen kapanış"),
      el("span", { class: "l80" }, "%80 aralık"), el("span", { class: "l95" }, "%95 aralık"),
      el("span", { class: "ld" }, "%80 aralığın dışında kalan gün")));
  } catch (e) {
    ozet.textContent = `Bu yılın tahminleri hesaplanamadı (${e.message}).`;
  }
}

function cubuk(ad, aciklama, v, hedef, biz) {
  const dolu = el("div", { class: "dolu" }), cizgi = el("div", { class: "hedef" });
  dolu.style.width = `${Math.max(0, Math.min(100, v))}%`;          // CSSOM: CSP satir ici stile izin vermez
  cizgi.style.left = `${hedef}%`;
  const yol = el("div", { class: "yol" }, dolu, cizgi);
  return el("div", { class: biz ? "cubuk biz" : "cubuk", title: aciklama },
    el("div", {}, ad), yol, el("div", { class: "sayi" }, yuzde(v)));
}

function guvenGrafik(rows, W) {
  const H = 230, L = 34, B = 40, T = 22;
  const n = rows.length, bw = (W - L) / n;
  const y = v => T + (1 - v / 100) * (H - T - B);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Söylenen olasılık ile gerçekleşen oran" });
  [0, 50, 100].forEach(v => {
    s.append(svg("line", { x1: L, x2: W, y1: y(v), y2: y(v), stroke: "var(--cizgi)", "stroke-width": 1 }));
    s.append(svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, `%${v}`));
  });
  rows.forEach((r, i) => {
    const az = r.gun < 20, x0 = L + i * bw + bw * 0.18, w = bw * 0.3;
    const op = az ? 0.35 : 1;
    s.append(svg("rect", { x: x0, y: y(r.ortalama_tahmin_pct), width: w, height: y(0) - y(r.ortalama_tahmin_pct), fill: "var(--pirinc)", opacity: op }));
    s.append(svg("rect", { x: x0 + w + 4, y: y(r.gerceklesen_pct), width: w, height: y(0) - y(r.gerceklesen_pct), fill: "var(--gercek)", opacity: op }));
    s.append(svg("text", { x: x0 + w / 2, y: y(r.ortalama_tahmin_pct) - 5, "text-anchor": "middle" }, `%${sayi(r.ortalama_tahmin_pct)}`));
    s.append(svg("text", { x: x0 + w * 1.5 + 4, y: y(r.gerceklesen_pct) - 5, "text-anchor": "middle" }, `%${sayi(r.gerceklesen_pct)}`));
    s.append(svg("text", { x: x0 + w + 2, y: H - 22, "text-anchor": "middle" }, `${r.tahmin_araligi.replace("-", "–")}`));
    s.append(svg("text", { x: x0 + w + 2, y: H - 6, "text-anchor": "middle" }, az ? `${r.gun} gün (az)` : `${sayi(r.gun)} gün`));
  });
  return s;
}

function yilGrafik(rows, W) {
  const H = 220, L = 34, B = 28, T = 22;
  const bw = (W - L) / rows.length;
  const y = v => T + (1 - v / 100) * (H - T - B);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Yıllara göre %80 aralığın tutma oranı" });
  s.append(svg("line", { x1: L, x2: W, y1: y(0), y2: y(0), stroke: "var(--cizgi)", "stroke-width": 1 }));
  s.append(svg("text", { x: L - 6, y: y(0) + 4, "text-anchor": "end" }, "%0"));
  s.append(svg("text", { x: L - 6, y: y(80) + 4, "text-anchor": "end" }, "%80"));
  rows.forEach((r, i) => {
    const x0 = L + i * bw + bw * 0.25, w = bw * 0.5;
    s.append(svg("rect", { x: x0, y: y(r.aralik80_tutma_pct), width: w, height: y(0) - y(r.aralik80_tutma_pct), fill: "var(--gercek)" }));
    s.append(svg("text", { x: x0 + w / 2, y: y(r.aralik80_tutma_pct) + 22, "text-anchor": "middle", class: "ici" }, yuzde(r.aralik80_tutma_pct)));
    s.append(svg("text", { x: x0 + w / 2, y: H - 8, "text-anchor": "middle" }, String(r.yil)));
  });
  s.append(svg("line", { x1: L, x2: W, y1: y(80), y2: y(80), stroke: "var(--murekkep)", "stroke-width": 1.5, "stroke-dasharray": "5 5" }));
  return s;
}

async function karne() {
  try {
    const d = await al("/api/tahmin/karne");
    const biz = d.karsilastirma.find(r => r.model === d.model);
    document.getElementById("karne-ozet").textContent =
      `Sistem ${d.egitim[0]}–${d.egitim[1]} verisiyle eğitildi ve ${d.test[0]}–${d.test[1]} arasındaki ${sayi(d.test_gun)} işlem gününde, ` +
      `Türk lirası krizleri ve pandemi dahil, sınandı. %80'lik aralığın tutma oranı ${yuzde(biz.aralik80)}, ` +
      `%95'lik aralığınki ${yuzde(biz.aralik95)}. Hedefe ne kadar yakınsa o kadar iyi.`;
    const grup = (baslik, alan, hedef) => el("div", {}, el("h3", {}, baslik),
      ...d.karsilastirma.map(r => cubuk(MODEL_ADI[r.model] || r.model, MODEL_ACIKLAMA[r.model] || "", r[alan], hedef, r.model === d.model)));
    document.getElementById("kiyas").replaceChildren(grup("%80 dediğimiz aralık (hedef %80)", "aralik80", 80),
                                                     grup("%95 dediğimiz aralık (hedef %95)", "aralik95", 95));
    document.getElementById("guven").replaceChildren(guvenGrafik(d.guvenilirlik, genislik("guven")),
      el("div", { class: "lejant" }, el("span", { class: "lp" }, "sistemin söylediği olasılık"),
         el("span", { class: "lgr" }, "gerçekte hareketli geçen günler")));
    document.getElementById("yillar").replaceChildren(yilGrafik(d.yillar, genislik("yillar")));
  } catch (e) {
    document.getElementById("karne-ozet").textContent = `Başarı karnesi okunamadı (${e.message}).`;
  }
}

function hepsi() { tahmin(); buYil(); karne(); }
hepsi();
let sonGenislik = window.innerWidth, zamanlayici = null;
window.addEventListener("resize", () => {
  clearTimeout(zamanlayici);
  zamanlayici = setTimeout(() => { if (Math.abs(window.innerWidth - sonGenislik) > 40) { sonGenislik = window.innerWidth; hepsi(); } }, 250);
});
