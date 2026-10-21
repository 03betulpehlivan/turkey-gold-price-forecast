"use strict";
const NS = "http://www.w3.org/2000/svg";
const AY = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"];
const AD = { "Sabit aralik": "Sabit aralık (basit yöntem)", "Gecmis benzer": "Geçmiş benzer günler", "EWMA": "EWMA",
  "GARCH-t": "GARCH-t", "GJR-t": "GJR-GARCH", "EWMA+ACI": "EWMA + ayar", "GARCH-t+ACI": "GARCH-t + ayar",
  "XGBoost-kantil": "XGBoost (makine öğrenmesi)", "ARIMA-GARCH": "ARIMA-GARCH", "HAR": "HAR", "Rejim (Markov)": "Rejim (Markov)",
  "Birlesim": "Birleşim (3 model)", "HAR-X": "HAR-X (dış piyasa verisi)", "Chronos": "Chronos (yapay zekâ)",
  "Birlesim-2": "ARIMA-GARCH + Chronos" };
const ad = m => AD[m] || m;

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
const kisa = iso => { const [, m, g] = iso.split("-").map(Number); return `${g} ${AY[m - 1]}`; };
const genislik = id => Math.max(300, Math.round(document.getElementById(id).clientWidth || 900));
const pdeger = p => (p == null ? "—" : p < 0.0001 ? "p < 0,0001" : `p = ${sayi(p, 4)}`);

function yelpaze(u, gecmis, W) {
  const H = 300, L = 66, R = W > 600 ? 70 : 52, T = 14, B = 30;
  const son = gecmis.slice(-30);
  const n = son.length, maxh = Math.max(...u.ufuklar.map(x => x.ufuk_gun));
  const Xs = L + (W > 600 ? 0.6 : 0.45) * (W - L - R);                // bugunun konumu; sag tarafi gelecek
  const xi = i => (i <= n - 1 ? L + (i / (n - 1)) * (Xs - L) : Xs + ((i - (n - 1)) / maxh) * (W - R - Xs));
  const lo = Math.min(...son.map(r => r.gercek), ...u.ufuklar.map(x => x.aralik95[0]));
  const hi = Math.max(...son.map(r => r.gercek), ...u.ufuklar.map(x => x.aralik95[1]));
  const pad = (hi - lo) * 0.06;
  const y = v => T + (1 - (v - (lo - pad)) / ((hi + pad) - (lo - pad))) * (H - T - B);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Son 30 işlem gününün kapanışları ve 1, 3, 5 gün sonrası için tahmin aralıkları" });
  const x0 = xi(n - 1), P = u.son_kapanis;
  const nokta = (alt, ust) => {
    const uc = [[x0, y(P)]].concat(u.ufuklar.map(f => [xi(n - 1 + f.ufuk_gun), y(f[ust][1])]));
    const ac = u.ufuklar.slice().reverse().map(f => [xi(n - 1 + f.ufuk_gun), y(f[alt][0])]);
    return uc.concat(ac).map(p => p.join(",")).join(" ");
  };
  s.append(svg("polygon", { points: nokta("aralik95", "aralik95"), fill: "var(--bant95)" }));
  s.append(svg("polygon", { points: nokta("aralik80", "aralik80"), fill: "var(--bant80)" }));
  s.append(svg("polyline", { points: son.map((r, i) => `${xi(i)},${y(r.gercek)}`).join(" "), fill: "none",
                             stroke: "var(--murekkep)", "stroke-width": 2 }));
  s.append(svg("circle", { cx: x0, cy: y(P), r: 4, fill: "var(--murekkep)" }));
  s.append(svg("line", { x1: x0, x2: x0, y1: T, y2: H - B, stroke: "var(--cizgi)", "stroke-dasharray": "3 4" }));
  s.append(svg("text", { x: x0 - 6, y: T + 10, "text-anchor": "end" }, "bugüne kadar"));
  s.append(svg("text", { x: x0 + 6, y: T + 10 }, "tahmin"));
  u.ufuklar.forEach((f, k) => {
    const xf = xi(n - 1 + f.ufuk_gun);
    s.append(svg("line", { x1: xf, x2: xf, y1: y(f.aralik95[1]), y2: y(f.aralik95[0]), stroke: "var(--soluk)", "stroke-width": 1 }));
    s.append(svg("text", { x: xf, y: H - 8, "text-anchor": k === u.ufuklar.length - 1 ? "end" : "middle" },
                 W > 600 ? `+${f.ufuk_gun} gün` : `+${f.ufuk_gun}`));
  });
  const fs = u.ufuklar[u.ufuklar.length - 1], xs = xi(n - 1 + fs.ufuk_gun);
  s.append(svg("text", { x: xs + 6, y: y(fs.aralik80[1]) + 4, class: "vurgu" }, sayi(fs.aralik80[1])));
  s.append(svg("text", { x: xs + 6, y: y(fs.aralik80[0]) + 4, class: "vurgu" }, sayi(fs.aralik80[0])));
  s.append(svg("text", { x: xi(0), y: H - 8 }, kisa(son[0].tarih)));
  s.append(svg("text", { x: L - 8, y: y(hi) + 4, "text-anchor": "end" }, `${sayi(hi)} TL`));
  s.append(svg("text", { x: L - 8, y: y(lo) + 4, "text-anchor": "end" }, `${sayi(lo)} TL`));
  return s;
}

function kart(f) {
  const etiket = f.ufuk_gun === 1 ? "Bir sonraki işlem günü" : `${f.ufuk_gun} işlem günü sonra`;
  const fin = f.final && f.final.tutma95 != null ? `2023–2026'da %95 aralık ${yuzde(f.final.tutma95)} tuttu.` : "";
  return el("div", { class: "kart" },
    el("p", { class: "ne" }, `${etiket} (${kisa(f.hedef_gun_yaklasik)} civarı), %80 olasılıkla`),
    el("p", { class: "aralik" }, `${sayi(f.aralik80[0])} – ${sayi(f.aralik80[1])}`, el("span", { class: "tl" }, "TL")),
    el("p", { class: "ikinci" }, "%95 olasılıkla ", el("b", {}, `${sayi(f.aralik95[0])} – ${sayi(f.aralik95[1])} TL`)),
    el("p", { class: "model" }, f.yedek
      ? `Model: ${ad(f.kullanilan)}. Asıl model ${ad(f.model)}; Chronos bu bilgisayarda kurulu olmadığı için yedek kullanıldı. ${fin}`
      : `Model: ${ad(f.kullanilan)}. ${fin}`));
}

async function tahmin(gecmis) {
  try {
    const u = await al("/api/tahmin/ufuklar");
    document.getElementById("ust").textContent =
      `${tarih(u.bilgi_gunu)} kapanışına göre (son kapanış ${sayi(u.son_kapanis, 2)} TL). Hafta sonu ve resmî tatiller işlem günü sayılmaz.`;
    document.getElementById("kartlar").replaceChildren(...u.ufuklar.map(kart));
    if (gecmis && gecmis.length > 2) document.getElementById("yelpaze").replaceChildren(yelpaze(u, gecmis, genislik("yelpaze")),
      el("div", { class: "lejant" }, el("span", { class: "lg" }, "gerçekleşen kapanış"),
         el("span", { class: "l80" }, "%80 aralık"), el("span", { class: "l95" }, "%95 aralık")));
    const hk = document.getElementById("hareket");
    if (u.hareketli_olasiligi_pct != null) {
      const p = u.hareketli_olasiligi_pct;
      hk.className = p >= 50 ? "not uyari-kutu" : "not";
      hk.textContent = (p >= 50 ? "Hareketli gün uyarısı: " : "") +
        `Bir sonraki işlem gününde fiyatın %${sayi(u.esik_pct, 0)}'den fazla değişme olasılığı ${yuzde(p, 0)}. ` +
        "Bu bir yatırım tavsiyesi değil, istatistiksel bir risk aralığıdır.";
    }
  } catch (e) {
    document.getElementById("ust").textContent = "Tahmin şu an hesaplanamıyor.";
    document.getElementById("kartlar").replaceChildren(el("p", { class: "bos" }, `Güncel fiyat verisi ve yarışma sonucu gerekiyor (${e.message}).`));
  }
}

async function buYil() {
  const ozet = document.getElementById("buyil-ozet");
  try {
    const d = await al("/api/tahmin/bu-yil-yarisma");
    const g = d.gunler;
    if (!g.length) { ozet.textContent = `${d.yil} için henüz sınanmış gün yok.`; return g; }
    ozet.textContent = `${d.yil} yılında şimdiye kadar ${d.gun} işlem gününün ${d.aralik80_tuttu} tanesinde fiyat, bir gün önceden ` +
      `verilen %80'lik aralığın içinde kaldı (${yuzde(d.aralik80_pct)}). %95'lik aralık ${yuzde(d.aralik95_pct)} tuttu. Model: ${ad(d.model)}.`;
    const W = genislik("buyil"), H = 250, L = 66, R = 8, T = 12, B = 28;
    const ymin = Math.min(...g.map(r => r.lo95)), ymax = Math.max(...g.map(r => r.hi95));
    const x = i => L + (i / Math.max(g.length - 1, 1)) * (W - L - R);
    const y = v => T + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - T - B);
    const bant = (a, b) => g.map((r, i) => `${x(i)},${y(r[a])}`).join(" ") + " " +
      g.map((r, i) => [i, r]).reverse().map(([i, r]) => `${x(i)},${y(r[b])}`).join(" ");
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ozet.textContent });
    s.append(svg("polygon", { points: bant("hi95", "lo95"), fill: "var(--bant95)" }));
    s.append(svg("polygon", { points: bant("hi80", "lo80"), fill: "var(--bant80)" }));
    s.append(svg("polyline", { points: g.map((r, i) => `${x(i)},${y(r.gercek)}`).join(" "), fill: "none", stroke: "var(--murekkep)", "stroke-width": 1.5 }));
    g.forEach((r, i) => { if (r.gercek < r.lo80 || r.gercek > r.hi80) s.append(svg("circle", { cx: x(i), cy: y(r.gercek), r: 2.4, fill: "var(--uyari)" })); });
    s.append(svg("line", { x1: L, x2: L, y1: T, y2: H - B, stroke: "var(--cizgi)" }));
    s.append(svg("text", { x: L - 8, y: T + 4, "text-anchor": "end" }, `${sayi(ymax)} TL`));
    s.append(svg("text", { x: L - 8, y: H - B, "text-anchor": "end" }, `${sayi(ymin)} TL`));
    s.append(svg("text", { x: L, y: H - 6 }, kisa(g[0].tarih)));
    s.append(svg("text", { x: W - R, y: H - 6, "text-anchor": "end" }, kisa(g[g.length - 1].tarih)));
    document.getElementById("buyil").replaceChildren(s, el("div", { class: "lejant" },
      el("span", { class: "lg" }, "gerçekleşen kapanış"), el("span", { class: "l80" }, "%80 aralık"),
      el("span", { class: "l95" }, "%95 aralık"), el("span", { class: "ld" }, "%80 aralığın dışında kalan gün")));
    return g;
  } catch (e) {
    ozet.textContent = `Bu yılın tahminleri hesaplanamadı (${e.message}).`;
    return [];
  }
}

function yilGrafik(rows, W) {
  const H = 220, L = 34, B = 28, T = 22;
  const bw = (W - L) / rows.length;
  const y = v => T + (1 - v / 100) * (H - T - B);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Yıllara göre %80 aralığın tutma oranı" });
  s.append(svg("line", { x1: L, x2: W, y1: y(0), y2: y(0), stroke: "var(--cizgi)" }));
  s.append(svg("text", { x: L - 6, y: y(80) + 4, "text-anchor": "end" }, "%80"));
  rows.forEach((r, i) => {
    const x0 = L + i * bw + bw * 0.18, w = bw * 0.64, fin = r.yil >= 2023;
    s.append(svg("rect", { x: x0, y: y(r.tutma80), width: w, height: y(0) - y(r.tutma80), fill: fin ? "var(--final)" : "var(--gercek)" }));
    if (W > 600) s.append(svg("text", { x: x0 + w / 2, y: y(r.tutma80) + 18, "text-anchor": "middle", class: "ici" }, sayi(r.tutma80, 0)));
    s.append(svg("text", { x: x0 + w / 2, y: H - 8, "text-anchor": "middle" }, W > 600 ? String(r.yil) : String(r.yil).slice(2)));
  });
  s.append(svg("line", { x1: L, x2: W, y1: y(80), y2: y(80), stroke: "var(--murekkep)", "stroke-width": 1.5, "stroke-dasharray": "5 5" }));
  return s;
}

async function yarisma() {
  try {
    const d = await al("/api/tahmin/yarisma");
    const u1 = d.ufuklar[0];
    document.getElementById("guven-ozet").textContent =
      "Modeller 2012–2022 arasında seçildi, sonra hiç görmedikleri 2023–2026 döneminde sınandı. Aşağıdaki oranlar o dönemin " +
      "sonuçları: aralık, söylenen olasılıkla ne kadar sık tuttu ve hiçbir şey öğrenmeyen sabit aralığa göre ne kadar daha iyi.";
    document.getElementById("guven").replaceChildren(...d.ufuklar.map(u => {
      const a = (u.anlamlilik || {}).final;
      return el("div", { class: "blok" },
        el("p", { class: "baslik" }, u.ufuk_gun === 1 ? "Bir sonraki gün" : `${u.ufuk_gun} gün sonra`),
        el("p", { class: "buyuk" }, yuzde(u.final.tutma95)),
        el("p", { class: "alt" }, `%95 dediği aralığın tutma oranı (${sayi(u.final.gun)} gün)`),
        el("p", { class: "satir" }, `%80 aralık: ${yuzde(u.final.tutma80)} tuttu`),
        a ? el("p", { class: "satir" }, `Sabit aralığa göre ${yuzde(a.iyilesme_pct)} daha iyi (${pdeger(a.p_degeri)})`) : "",
        el("p", { class: "satir" }, `Model: ${ad(u.kazanan)}`));
    }));
    document.getElementById("yarisma-ozet").textContent =
      `${d.model_sayisi} model, 2012–2022 arasında aylık pencerelerle yarıştı. Puan, aralığın hem genişliğini hem de ıskaladığı ` +
      "günleri cezalandırır; düşük puan daha dar ve daha isabetli aralık demek. Her ufukta, aralıkları hedefe yakın tutan modeller " +
      "arasında puanı en düşük olan kazandı (●).";
    const modeller = u1.modeller.map(r => r.model);
    const bul = (u, m) => u.modeller.find(r => r.model === m);
    const tbl = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Model"),
      ...d.ufuklar.map(u => el("th", {}, u.ufuk_gun === 1 ? "1 gün" : `${u.ufuk_gun} gün`)))));
    const gov = el("tbody");
    modeller.forEach(m => gov.append(el("tr", {}, el("td", {}, ad(m)), ...d.ufuklar.map(u => {
      const r = bul(u, m), cls = u.kazanan === m ? "kazanan" : (r.uygun ? "" : "disarida");
      return el("td", { class: cls }, el("span", { class: "puan" }, sayi(r.puan, 2)),
                el("span", { class: "oran" }, `%80: ${sayi(r.tutma80, 1)}  %95: ${sayi(r.tutma95, 1)}`));
    }))));
    tbl.append(gov);
    document.getElementById("yarisma").replaceChildren(tbl, el("p", { class: "not" },
      `● kazanan. Soluk hücreler: aralıkları hedefe yeterince yakın değil (%80 aralık 80 ± ${sayi(d.hedef_sapma["80"], 0)}, ` +
      `%95 aralık 95 ± ${sayi(d.hedef_sapma["95"], 0)} puan); puanı düşük olsa bile seçilemez.`));
    document.getElementById("yillar-ozet").textContent =
      `Bir sonraki gün için %80 aralığın her yıl tuttuğu günlerin oranı (${ad(u1.kazanan)}). Mavi çubuklar hiç görülmemiş 2023–2026, ` +
      "yeşiller seçim dönemi. 2018 ve 2021 Türk lirası krizleri, 2020 pandemi, 2023 fiziki prim kırılması yılı.";
    document.getElementById("yillar").replaceChildren(yilGrafik(u1.yillar, genislik("yillar")),
      el("div", { class: "lejant" }, el("span", { class: "lgr" }, "seçim dönemi 2012–2022"), el("span", { class: "lf" }, "hiç görülmemiş 2023–2026")));
  } catch (e) {
    document.getElementById("guven-ozet").textContent = `Yarışma sonucu okunamadı (${e.message}).`;
  }
}

async function hepsi() {
  const gecmis = await buYil();
  await Promise.all([tahmin(gecmis), yarisma()]);
}
hepsi();
let sonGenislik = window.innerWidth, zamanlayici = null;
window.addEventListener("resize", () => {
  clearTimeout(zamanlayici);
  zamanlayici = setTimeout(() => { if (Math.abs(window.innerWidth - sonGenislik) > 40) { sonGenislik = window.innerWidth; hepsi(); } }, 250);
});
