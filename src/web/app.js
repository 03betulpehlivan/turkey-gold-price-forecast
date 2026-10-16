const TL = new Intl.NumberFormat("tr-TR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const MODEL_ADI = { "RW": "Yarın = bugün", "Bosluk": "Boşluk", "Bosluk-Kayan": "Boşluk (kayan)",
                    "Prim-Capa": "Prim çapası", "Mean-Return": "Ortalama getiri" };
const PROTOKOL = { "LIVE_PHYS_V1": "Fiziki sabah", "LIVE_V1": "Teorik, GC=F", "LIVE_V2": "Teorik, GLD" };
const NS = "http://www.w3.org/2000/svg";

async function al(yol) {
  const r = await fetch(yol);
  if (!r.ok) throw new Error((await r.json()).detail || r.status);
  return r.json();
}
function el(ad, nit = {}, ...cocuk) {
  const e = document.createElement(ad);
  Object.entries(nit).forEach(([k, v]) => e.setAttribute(k, v));
  cocuk.forEach(c => e.append(c));
  return e;
}
function svg(ad, nit) {
  const e = document.createElementNS(NS, ad);
  Object.entries(nit).forEach(([k, v]) => e.setAttribute(k, v));
  return e;
}
function tarih(s) { return s ? new Date(s + "T12:00:00").toLocaleDateString("tr-TR", { day: "numeric", month: "long", year: "numeric" }) : "—"; }

function aralikSeridi(kutu, veri, baslik) {
  kutu.replaceChildren();
  if (!veri) { kutu.append(el("p", { class: "bos" }, `${baslik}: henüz tahmin yok.`)); return; }
  const f = veri.forecasts;
  const uclar = f.flatMap(x => [x.lo95, x.hi95, x.point, x.realized]).filter(v => v != null);
  const min = Math.min(...uclar), max = Math.max(...uclar), pay = (max - min) * 0.06 || 1;
  const x = v => ((v - (min - pay)) / ((max + pay) - (min - pay))) * 1000;
  kutu.append(el("p", { class: "alt" }, `${baslik}: ${tarih(veri.asof_date)} bilgisiyle`));
  const cizgi = (v, renk, k) => svg("line", { x1: x(v), x2: x(v), y1: k, y2: 28 - k, stroke: renk, "stroke-width": 3,
                                             "vector-effect": "non-scaling-stroke" });
  f.forEach(r => {
    const s = svg("svg", { class: "bant", viewBox: "0 0 1000 28", preserveAspectRatio: "none", role: "img",
      "aria-label": `${MODEL_ADI[r.model] || r.model}: ${TL.format(r.point)} TL` + (r.lo80 != null ? `, %80 aralık ${TL.format(r.lo80)} – ${TL.format(r.hi80)}` : "") });
    if (r.lo95 != null) s.append(svg("rect", { x: x(r.lo95), y: 6, width: x(r.hi95) - x(r.lo95), height: 16, fill: "var(--bant95)" }));
    if (r.lo80 != null) s.append(svg("rect", { x: x(r.lo80), y: 6, width: x(r.hi80) - x(r.lo80), height: 16, fill: "var(--bant80)" }));
    s.append(cizgi(r.point, "var(--murekkep)", 2));
    if (r.realized != null) s.append(cizgi(r.realized, "var(--gercek)", 0));
    const hata = r.abs_err != null ? ` (hata ${TL.format(r.abs_err)})` : "";
    kutu.append(el("div", { class: "satir" }, el("span", { class: "ad" }, MODEL_ADI[r.model] || r.model),
                   el("span", { class: "deger", title: `Tahmin${hata}` }, TL.format(r.point)), s));
  });
  kutu.append(el("div", { class: "eksen" }, el("div"), el("div"),
                  el("div", {}, el("span", {}, TL.format(min)), el("span", {}, TL.format(max)))));
  if (f[0].realized != null) {
    const iyi = f.filter(r => r.abs_err != null).sort((a, b) => a.abs_err - b.abs_err)[0];
    kutu.append(el("p", { class: "alt" }, `Gerçekleşen: ${TL.format(f[0].realized)} TL. En yakın tahmin ${MODEL_ADI[iyi.model] || iyi.model}, ` +
      `fark ${TL.format(iyi.abs_err)} TL. Tek günlük sonuç; betimleyicidir.`));
  }
}

function cizgiGrafik(kutu, fiyat, tahmin) {
  kutu.replaceChildren();
  const p = fiyat.slice(-120);
  if (p.length < 2) { kutu.append(el("p", { class: "bos" }, "Fiyat serisi veritabanında yok (önce yükleme yapın).")); return; }
  const G = 1000, Y = 300, P = 6;
  const t = tahmin.filter(f => f.asof_date >= p[0].date);
  const degerler = p.map(d => d.value).concat(t.map(f => f.point));
  const min = Math.min(...degerler), max = Math.max(...degerler);
  const t0 = new Date(p[0].date), t1 = new Date(p[p.length - 1].date);
  const son = t.length ? new Date(t[0].asof_date) : t1, bitis = son > t1 ? son : t1;
  const x = s => P + ((new Date(s) - t0) / Math.max(bitis - t0, 1)) * (G - 2 * P);
  const y = v => Y - P - ((v - min) / (max - min || 1)) * (Y - 2 * P);
  const s = svg("svg", { class: "grafik", viewBox: `0 0 ${G} ${Y}`, role: "img",
    "aria-label": `Fiziki kapanış ${tarih(p[0].date)} – ${tarih(p[p.length - 1].date)}, ${TL.format(min)} ile ${TL.format(max)} TL arası` });
  [min, (min + max) / 2, max].forEach(v => s.append(svg("line", { x1: 0, x2: G, y1: y(v), y2: y(v),
    stroke: "var(--cizgi)", "stroke-width": 1, "stroke-dasharray": "4 6" })));
  s.append(svg("polyline", { points: p.map(d => `${x(d.date)},${y(d.value)}`).join(" "), fill: "none",
                             stroke: "var(--murekkep)", "stroke-width": 2 }));
  t.forEach(f => s.append(svg("circle", { cx: x(f.asof_date), cy: y(f.point), r: 5, fill: "var(--pirinc)" })));
  kutu.append(el("div", { class: "grafik-ust" }, el("span", {}, `En yüksek ${TL.format(max)} TL`),
                 el("span", {}, `Son kapanış ${TL.format(p[p.length - 1].value)} TL`)),
              s,
              el("div", { class: "grafik-eksen" }, el("span", {}, `${tarih(p[0].date)} (en düşük ${TL.format(min)} TL)`),
                 el("span", {}, tarih(bitis.toISOString().slice(0, 10)))),
              el("p", { class: "alt" }, "Çizgi: gerçekleşen kapanış. Noktalar: Boşluk modelinin sabah tahminleri."));
}

const yuzde = v => v == null ? "—" : (v < 0 ? "−%" : "%") + Math.abs(v).toFixed(1).replace(".", ",");
function karneTablosu(kutu, karneler) {
  kutu.replaceChildren();
  const t = el("table", {}, el("thead", {}, el("tr", {}, ...["Protokol", "Model", "Puanlanan", "Ort. hata (TL)",
    "Yarın = bugün'e göre", "%80 aralıkta", "%95 aralıkta"].map(h => el("th", { scope: "col" }, h)))));
  const g = el("tbody");
  let bos = true;
  karneler.forEach(k => k.models.forEach(m => {
    bos = false;
    const kap = v => v == null ? "—" : `${Math.round(v * m.n)}/${m.n}`;
    g.append(el("tr", {}, el("td", {}, PROTOKOL[k.protocol_id]), el("td", {}, MODEL_ADI[m.model] || m.model),
      el("td", {}, String(m.n)), el("td", {}, TL.format(m.mae)), el("td", {}, m.model === "RW" ? "referans" : yuzde(m.vs_rw_pct)),
      el("td", {}, kap(m.cov80)), el("td", {}, kap(m.cov95))));
  }));
  t.append(g);
  kutu.append(bos ? el("p", { class: "bos" }, "Henüz puanlanmış tahmin yok. İlk puanlama, gerçekleşen fiyat yayımlanınca yapılır.") : t);
}

const HUKUM = { "DAHA_IYI": "daha iyi", "DAHA_KOTU": "daha kötü", "PRATIKTE_ESDEGER": "pratikte eşdeğer",
                "AYIRT_EDILEMEDI": "ayırt edilemedi" };
const SINAV = {
  FINAL_EXAM_V1: { ad: "Teorik fiyat, 2026", satir: [
    ["point.verdict", "Nokta tahmini, yarın = bugün'e göre"], ["interval.verdict.all_passed", "Aralık ölçütlerinin tümü", "evet_hayir"]] },
  FINAL_EXAM_PHYS_V1: { ad: "Fiziki fiyat, 2023–2026", satir: [["primary.verdict", "Boşluk modeli, yarın = bugün'e göre"]] },
  BACK_EXAM_ONS_V1: { ad: "Geriye dönük, 2001–2004", satir: [
    ["hypothesis_supported", "Ons bilgisi hipotezi", "destek"],
    ["primary_walk_forward.Ons_vs_RW.verdict", "Ons modeli, yarın = bugün'e göre", null, "primary_walk_forward.Ons_vs_RW.improvement_pct"],
    ["primary_walk_forward.Ons_vs_Egilim.verdict", "Ons modeli, yalnız eğilime göre", null, "primary_walk_forward.Ons_vs_Egilim.improvement_pct"],
    ["secondary_frozen_dev.Donuk_vs_RW.verdict", "İkincil: dondurulmuş katsayılar, yarın = bugün'e göre", "ikincil"],
    ["secondary_frozen_dev.Donuk_vs_Egilim.verdict", "İkincil: dondurulmuş katsayılar, yalnız eğilime göre", "ikincil"]] }
};
const yol = (o, y) => y.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
function sinavListesi(kutu, sinavlar) {
  kutu.replaceChildren();
  if (!sinavlar.length) { kutu.append(el("p", { class: "bos" }, "Sınav sonucu yok.")); return; }
  sinavlar.forEach(s => {
    const tanim = SINAV[s.protocol_id] || { ad: "", satir: [] };
    const liste = el("div");
    tanim.satir.forEach(([y, etiket, tur, pct]) => {
      const v = yol(s.result, y);
      if (v === undefined) return;
      let metin = HUKUM[v] || String(v), olumsuz = ["DAHA_KOTU", "AYIRT_EDILEMEDI"].includes(v);
      if (tur === "destek") { metin = v ? "desteklendi" : "desteklenmedi"; olumsuz = !v; }
      if (tur === "evet_hayir") { metin = v ? "sağlandı" : "sağlanmadı"; olumsuz = !v; }
      const ek = pct && yol(s.result, pct) != null ? ` (iyileşme ${yuzde(yol(s.result, pct))})` : "";
      const sinif = tur === "ikincil" ? "ikincil" : (olumsuz ? "olumsuz" : "");
      liste.append(el("div", { class: sinif }, `${etiket}: ${metin}${ek}`));
    });
    if (!liste.childElementCount) liste.append(el("div", { class: "bos" }, "Özet alanı bulunamadı."));
    kutu.append(el("div", { class: "sinav" }, el("div", {}, el("b", {}, tanim.ad || s.protocol_id), el("small", {}, s.protocol_id)), liste));
  });
}

const DURUM = { TAMAM: "sorunsuz", UYARI: "uyarı var", HATA: "hata var" };
const GOREV = { sabah: "Sabah görevi (12:30)", aksam: "Akşam görevi (22:30)" };
const DENETIM = { zaman: "Zamanlama", gorev: "Görev", gunluk: "Günlük", yedek: "Yedek", github: "GitHub",
                  tahmin: "Tahmin", butunluk: "Bütünlük" };
async function sistemDurumu() {
  const kutu = document.getElementById("sistem");
  kutu.replaceChildren();
  let s;
  try { s = await al("/api/system"); } catch (e) { kutu.append(el("p", { class: "bos" }, "Sistem durumu alınamadı.")); return; }
  kutu.className = `sistem ${s.overall}`;
  kutu.append(el("h2", {}, `Sistem: ${DURUM[s.overall] || s.overall}`));
  Object.entries(s.tasks).forEach(([ad, g]) => {
    const zaman = g.at ? new Date(g.at).toLocaleString("tr-TR", { dateStyle: "short", timeStyle: "short" }) : "kayıt yok";
    const blok = el("div", { class: "gorev" }, el("div", {}, `${GOREV[ad] || ad}: son çalışma ${zaman}, ${DURUM[g.overall] || g.overall}`));
    const sorun = g.checks.filter(c => c.status !== "TAMAM");
    if (sorun.length) blok.append(el("ul", {}, ...sorun.map(c => el("li", { class: c.status }, `${DENETIM[c.name] || c.name}: ${c.detail}`))));
    const tum = el("details", {}, el("summary", {}, "Tüm denetimler"),
                   el("ul", {}, ...g.checks.map(c => el("li", { class: c.status }, `${DENETIM[c.name] || c.name}: ${c.detail}`))));
    blok.append(tum);
    kutu.append(blok);
  });
}

async function yukle() {
  sistemDurumu();
  const durum = document.getElementById("durum");
  try {
    const saglik = await al("/api/health");
    if (saglik.status !== "ok") { durum.textContent = "Veritabanı boş: önce “python -m src.db.store load” çalıştırın."; return; }
    const motor = saglik.db.startsWith("mssql") ? "SQL Server" : (saglik.db.startsWith("sqlite") ? "SQLite" : "veritabanı");
    durum.textContent = `Veritabanı: ${motor}. Son güncelleme: ${new Date(saglik.last_load.at_utc).toLocaleString("tr-TR")}`;
    const sonu = p => al(`/api/forecasts/latest?protocol=${p}`).catch(() => null);
    const [fiz, v1, v2] = await Promise.all(["LIVE_PHYS_V1", "LIVE_V1", "LIVE_V2"].map(sonu));
    aralikSeridi(document.getElementById("fiziki"), fiz, "Bugünün tahmini");
    aralikSeridi(document.getElementById("teorik-v1"), v1, "Vadeli ons (GC=F)");
    aralikSeridi(document.getElementById("teorik-v2"), v2, "Altın fonu (GLD)");
    const tum = (await al("/api/forecasts?protocol=LIVE_PHYS_V1&limit=5000")).forecasts;
    const puanli = tum.filter(f => f.realized != null).map(f => f.asof_date).sort().pop();
    const dunKutu = document.getElementById("fiziki-dun");
    if (puanli) aralikSeridi(dunKutu, { asof_date: puanli, forecasts: tum.filter(f => f.asof_date === puanli) },
                             "Son puanlanan gün");
    else dunKutu.remove();
    const fiyat = await al("/api/prices?series=fiziki_kapanis");
    cizgiGrafik(document.getElementById("grafik"), fiyat.prices, tum.filter(f => f.model === "Bosluk"));
    const karneler = await Promise.all(["LIVE_PHYS_V1", "LIVE_V1", "LIVE_V2"].map(p => al(`/api/scorecard?protocol=${p}`)));
    karneTablosu(document.getElementById("karne"), karneler);
    sinavListesi(document.getElementById("sinavlar"), (await al("/api/exams")).exams);
  } catch (e) {
    durum.textContent = `Veriler alınamadı: ${e.message}. API çalışıyor mu?`;
  }
}
yukle();
