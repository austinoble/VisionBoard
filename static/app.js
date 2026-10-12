/*
 * Detection GUI frontend.
 *
 * Rendering model: OpenSeadragon shows the image as a deep-zoom tile pyramid (the browser
 * never loads the full multi-gigapixel PNG). Annotations are NOT baked into the image —
 * they are re-drawn every frame as vectors on a transparent <canvas> stacked on top of the
 * viewer, with stroke widths and font sizes fixed in *screen* pixels. That is what keeps
 * overlays razor-sharp at any zoom level while the imagery underneath pixelates naturally.
 *
 * Coordinates: the API serves all polygons in displayed-image pixel space. Each frame we
 * compute the affine image->screen transform once (OSD viewport has no rotation here) and
 * apply it to every visible polygon.
 */
"use strict";

/* ── tiny helpers ─────────────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
};
const debounce = (fn, ms) => {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ── global state ─────────────────────────────────────────────────────────── */
const S = {
  scenes: [],
  cur: null,            // {source, stem, width, height, has_stage1, has_stage2, has_gt}
  viewer: null,
  ann: { stage1: [], stage2: [], preds: [], gt: [], delete_regions: [] },
  eval: null,           // evaluation used to color/navigate the currently visible layer
  stage1Eval: null,     // class-agnostic Stage-1 evaluation, always loaded
  stage2Eval: null,     // class-aware Stage-2 evaluation, always loaded
  stage2DetectionEval: null, // class-agnostic Stage-2 evaluation, always loaded
  evalLayer: null,      // "stage1" when alone, otherwise "stage2"
  classOn: {},          // cls -> bool
  navIndex: {},         // cls -> current prediction index
  colors: {},           // cls -> css color
  selected: null,       // {kind:'pred'|'gt', id}
  pendingFocus: null,   // instance waiting for its scene tile viewer to open
  ui: {
    showStage1: true, showStage2: true, showGt: false,
    evalMode: false,
    suppressStage1: true, perClassConf: true,
    conf: 0.30, iou: 0.30,
  },
  annRequest: 0,
  tilePoll: null,       // interval handle while pyramid builds
  jobWatch: {},         // job_id -> interval handle
};

const EVAL_COLORS = { TP: "#2ecc71", FP: "#ff5252" };
// stable palette for the 8 model classes; anything else gets a hashed hue
const CLASS_COLORS = {
  radar: "#a855f7", missile: "#22d3ee", object: "#3b82f6",
};
const CLASS_ORDER = [
  "radar", "missile", "tank", "arty", "camo",
  "logistic", "smallvehicle", "vehicle", "object",
];
const TRAINED_CLASSES = CLASS_ORDER.slice(0, 8);
function orderedClasses(classes) {
  const rank = (c) => {
    const i = CLASS_ORDER.indexOf(c);
    return i < 0 ? CLASS_ORDER.length : i;
  };
  return [...classes].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

function activePredictions() {
  const out = [];
  if (S.ui.showStage1) {
    for (const p of S.ann.stage1) {
      if (!(S.ui.showStage2 && S.ui.suppressStage1 && p.suppressed_by_stage2)) out.push(p);
    }
  }
  if (S.ui.showStage2) out.push(...S.ann.stage2);
  return out;
}

function activePredictionSet() {
  if (S.ui.showStage2) return "stage2";
  if (S.ui.showStage1) return "stage1";
  return null;
}

function syncVisualEvaluation() {
  if (S.ui.showStage2 && S.stage2Eval) {
    S.eval = S.stage2Eval;
    S.evalLayer = "stage2";
  } else if (S.ui.showStage1 && S.stage1Eval) {
    S.eval = S.stage1Eval;
    S.evalLayer = "stage1";
  } else {
    S.eval = null;
    S.evalLayer = null;
  }
}

function navigationItems(cls) {
  const navLayer = S.evalLayer || activePredictionSet();
  const layer = navLayer === "stage1" ? S.ann.stage1
    : (navLayer === "stage2" ? S.ann.stage2 : []);
  const predictions = layer
    .filter((p) => !p.in_delete && p.cls === cls &&
                   !(p.layer === "stage1" && S.ui.showStage2 && S.ui.suppressStage1 &&
                     p.suppressed_by_stage2))
    .map((obj) => ({ kind: "pred", obj,
                     status: S.eval ? S.eval.pred_status[obj.id] : null }));
  if (!(S.ui.evalMode && S.eval)) return predictions;
  const gtItems = S.ann.gt
    .filter((g) => g.cls === cls && ["TP", "FN"].includes(S.eval.gt_status[g.id]))
    .map((obj) => ({ kind: "gt", obj, status: S.eval.gt_status[obj.id] }));
  const falsePositives = predictions.filter((x) => x.status === "FP");
  return gtItems.concat(falsePositives);
}

function classColor(cls) {
  if (!S.colors[cls]) {
    if (CLASS_COLORS[cls]) S.colors[cls] = CLASS_COLORS[cls];
    else S.colors[cls] = "#facc15";
  }
  return S.colors[cls];
}

function confidenceMode() {
  return S.ui.perClassConf && S.cur && S.cur.has_stage2_raw ? "per_class" : "global";
}

function updateConfidenceControls() {
  const toggle = $("per-class-conf");
  const slider = $("conf-thr");
  const rawAvailable = !S.cur || S.cur.has_stage2_raw;
  const perClassActive = S.ui.perClassConf && rawAvailable;
  toggle.disabled = !rawAvailable;
  slider.disabled = perClassActive && !S.ui.showStage1;
  $("conf-label").textContent = perClassActive && S.ui.showStage1
    ? "Stage-1 confidence" : "Confidence";
  let note;
  if (!rawAvailable) {
    note = "Raw Stage-2 scores unavailable for this scene; per-class mode is disabled.";
  } else if (perClassActive) {
    note = S.ui.showStage1
      ? "Stage 2 uses shipped per-class gates; this slider applies only to Stage 1."
      : "Stage 2 uses shipped per-class gates; the global slider is inactive.";
  } else {
    note = "Global raw confidence is used consistently for boxes, counts, arrows and metrics.";
  }
  $("confidence-mode-note").textContent = note;
}

/* ── dataset + ID lookup ─────────────────────────────────────────────────── */
async function loadScenes() {
  const data = await api("/api/scenes");
  S.scenes = data.scenes;
  S.ui.canUpload = !!(data.ui && data.ui.upload);
  S.ui.canInfer = !!(data.ui && data.ui.inference);
  $("btn-upload").classList.toggle("hidden", !S.ui.canUpload);
  $("btn-infer").classList.toggle("hidden", !S.ui.canInfer);
  $("btn-jobs").classList.toggle("hidden", !S.ui.canInfer);
}

async function searchInstance(mode, inputId) {
  const q = $(inputId).value.trim();
  const out = $("instance-results");
  if (!q) { out.classList.add("hidden"); out.innerHTML = ""; return; }
  const otherId = mode === "image" ? "instance-id-search" : "image-id-search";
  $(otherId).value = "";
  out.classList.remove("hidden");
  out.innerHTML = `<div class="instance-result">searching…</div>`;
  try {
    const data = await api(`/api/instances?q=${encodeURIComponent(q)}&mode=${mode}`);
    const exact = mode === "full" ? data.results.filter((r) =>
      r.instance_id && r.instance_id.toLowerCase() === q.toLowerCase()) : [];
    if (mode === "full" && exact.length === 1) { await goToInstance(exact[0]); return; }
    if (mode === "image" && data.results.length === 1) {
      await goToInstance(data.results[0]); return;
    }
    if (!data.results.length) {
      out.innerHTML = `<div class="instance-result">No matching instance ID</div>`;
      return;
    }
    out.innerHTML = data.results.map((r, i) => `
      <div class="instance-result" data-result="${i}">
        <div class="iid">${r.instance_id ? `${esc(r.instance_id)} · ${esc(r.cls)}`
          : `image ${esc(r.image_number)}`}</div>
        <div class="where">${esc(r.stem)}</div>
      </div>`).join("");
    out.querySelectorAll("[data-result]").forEach((el) => {
      el.onclick = () => goToInstance(data.results[Number(el.dataset.result)]);
    });
  } catch (e) {
    out.innerHTML = `<div class="instance-result">${esc(e.message)}</div>`;
  }
}

async function goToInstance(result) {
  $("instance-results").classList.add("hidden");
  await openScene(result.source, result.stem);
  if (!result.instance_id) return;
  const obj = S.ann.gt.find((g) => g.instance_id === result.instance_id);
  if (!obj) { alert(`Instance ${result.instance_id} was not found in ${result.stem}`); return; }
  S.ui.showGt = true;
  $("show-gt").checked = true;
  focusObject("gt", obj);
}

/* ── opening a scene ──────────────────────────────────────────────────────── */
async function openScene(source, stem) {
  clearInterval(S.tilePoll);
  S.cur = await api(`/api/scenes/${source}/${stem}`);
  S.ann = { stage1: [], stage2: [], preds: [], gt: [], delete_regions: [] };
  S.eval = null;
  S.stage1Eval = null;
  S.stage2Eval = null;
  S.stage2DetectionEval = null;
  S.evalLayer = null;
  S.selected = null;
  hideBoxInfo();
  $("scene-title").textContent =
    `${source}/${stem}  —  ${S.cur.width}×${S.cur.height}px`;
  $("btn-infer").disabled = !S.ui.canInfer;
  updateConfidenceControls();

  // default to TP/FP/FN coloring whenever a scene can be evaluated (until the user
  // toggles eval mode manually — then their choice wins for the session)
  if (!S.evalTouched) {
    const both = (S.cur.has_stage1 || S.cur.has_stage2) && S.cur.has_gt;
    S.ui.evalMode = both;
    $("eval-mode").checked = both;
    $("eval-legend").classList.toggle("hidden", !both);
    if (both && !S.ui.showGt) { S.ui.showGt = true; $("show-gt").checked = true; }
  }

  if (S.viewer) { S.viewer.destroy(); S.viewer = null; }
  $("viewer-placeholder").classList.remove("hidden");
  $("placeholder-msg").classList.add("hidden");
  $("tile-progress").classList.remove("hidden");

  // kick off / wait for the tile pyramid, then start the viewer
  const poll = async () => {
    let st;
    try { st = await api(`/api/scenes/${source}/${stem}/tile_status`); }
    catch (e) { clearInterval(S.tilePoll); alert("tile status failed: " + e.message); return; }
    if (S.cur.source !== source || S.cur.stem !== stem) { clearInterval(S.tilePoll); return; }
    $("tile-progress-pct").textContent = Math.round(st.progress * 100) + "%";
    $("tile-progress-fill").style.width = st.progress * 100 + "%";
    if (st.state === "ready") {
      clearInterval(S.tilePoll);
      $("viewer-placeholder").classList.add("hidden");
      startViewer(source, stem);
    } else if (st.state === "error") {
      clearInterval(S.tilePoll);
      $("tile-progress").innerHTML =
        `<div style="color:#ff5252">tile build failed: ${esc(st.error || "?")}</div>`;
    }
  };
  await poll();
  if (!S.viewer) S.tilePoll = setInterval(poll, 800);

  await reloadAnnotations();
}

function startViewer(source, stem) {
  S.viewer = OpenSeadragon({
    id: "viewer",
    tileSources: `/api/scenes/${source}/${stem}/image.dzi`,
    showNavigationControl: false,
    showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT",
    navigatorHeight: 120, navigatorWidth: 160,
    maxZoomPixelRatio: 24,          // let the user dive far past 1:1 to inspect pixels
    minZoomImageRatio: 0.7,
    imageSmoothingEnabled: true,
    animationTime: 0.4,
    gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
    crossOriginPolicy: false,
  });
  S.viewer.addHandler("open", () => {
    requestDraw();
    if (S.pendingFocus && S.pendingFocus.source === S.cur.source &&
        S.pendingFocus.stem === S.cur.stem) {
      const pending = S.pendingFocus;
      S.pendingFocus = null;
      focusObject(pending.kind, pending.obj);
    }
  });
  S.viewer.addHandler("update-viewport", requestDraw);
  S.viewer.addHandler("resize", () => { sizeOverlay(); requestDraw(); });
  S.viewer.addHandler("canvas-click", (ev) => {
    if (!ev.quick) return;
    const vp = S.viewer.viewport.pointFromPixel(ev.position);
    const img = S.viewer.viewport.viewportToImageCoordinates(vp);
    handlePick(img.x, img.y);
  });
  sizeOverlay();
}

/* ── annotations + evaluation ─────────────────────────────────────────────── */
async function reloadAnnotations() {
  if (!S.cur) return;
  const { source, stem } = S.cur;
  const requestId = ++S.annRequest;
  const q = new URLSearchParams({
    conf: String(S.ui.conf),
    confidence_mode: confidenceMode(),
    suppress_stage1: String(S.ui.suppressStage1),
  });
  const annotations = await api(`/api/scenes/${source}/${stem}/annotations?${q}`);
  if (requestId !== S.annRequest || S.cur.source !== source || S.cur.stem !== stem) return;
  S.ann = annotations;
  S.ann.delete_regions = S.ann.delete_regions || [];
  S.ann.stage1 = S.ann.stage1 || [];
  S.ann.stage2 = S.ann.stage2 || [];

  S.navIndex = {};
  rebuildClassList();
  await refreshEval();
  requestDraw();
}

const refreshEval = async () => {
  S.eval = null;
  S.stage1Eval = null;
  S.stage2Eval = null;
  S.stage2DetectionEval = null;
  S.evalLayer = null;
  $("stage1-detection-stats-section").classList.add("hidden");
  $("stage2-detection-stats-section").classList.add("hidden");
  $("classification-stats-section").classList.add("hidden");
  if (S.cur && S.cur.has_gt && (S.cur.has_stage1 || S.cur.has_stage2)) {
    const { source, stem } = S.cur;
    const common = `iou=${S.ui.iou}&conf=${S.ui.conf}&ignore_difficult=true`
      + `&confidence_mode=${confidenceMode()}`
      + `&suppress_stage1=${S.ui.suppressStage1}`;
    try {
      const stage1Promise = S.cur.has_stage1
        ? api(`/api/scenes/${source}/${stem}/eval?${common}&prediction_set=stage1&class_agnostic=true`)
        : Promise.resolve(null);
      const stage2ClassPromise = S.cur.has_stage2
        ? api(`/api/scenes/${source}/${stem}/eval?${common}&prediction_set=stage2&class_agnostic=false`)
        : Promise.resolve(null);
      const stage2DetectionPromise = S.cur.has_stage2
        ? api(`/api/scenes/${source}/${stem}/eval?${common}&prediction_set=stage2&class_agnostic=true`)
        : Promise.resolve(null);
      const [stage1, stage2Class, stage2Detection] = await Promise.all([
        stage1Promise, stage2ClassPromise, stage2DetectionPromise,
      ]);
      if (S.cur.source === source && S.cur.stem === stem) {
        S.stage1Eval = stage1;
        S.stage2Eval = stage2Class;
        S.stage2DetectionEval = stage2Detection;
        syncVisualEvaluation();
        renderStats();
        rebuildClassList();
        $("stage1-detection-stats-section").classList.toggle("hidden", !stage1);
        $("stage2-detection-stats-section").classList.toggle("hidden", !stage2Detection);
        $("classification-stats-section").classList.toggle("hidden", !stage2Class);
      }
    } catch (e) { console.warn("eval failed:", e.message); }
  }
  requestDraw();
};
const refreshEvalDebounced = debounce(refreshEval, 350);

function rebuildClassList() {
  const counts = {};
  for (const cls of CLASS_ORDER) counts[cls] = { p: 0, g: 0 };
  for (const p of [...S.ann.stage2, ...S.ann.stage1]) {
    if (p.in_delete) continue;
    (counts[p.cls] = counts[p.cls] || { p: 0, g: 0 }).p++;
  }
  for (const g of S.ann.gt) {
    (counts[g.cls] = counts[g.cls] || { p: 0, g: 0 }).g++;
  }
  const el = $("class-list");
  el.innerHTML = "";
  for (const cls of orderedClasses(Object.keys(counts))) {
    if (!(cls in S.classOn)) S.classOn[cls] = true;
    const available = navigationItems(cls);
    const current = available.length ? ((S.navIndex[cls] || 0) % available.length) + 1 : 0;
    const row = document.createElement("div");
    row.className = "class-row";
    row.innerHTML = `
      <input class="class-visible" type="checkbox" ${S.classOn[cls] ? "checked" : ""}>
      <span class="class-swatch" style="background:${classColor(cls)}"></span>
      <span class="class-name">${esc(cls)}</span>
      <span class="cnt">${counts[cls].p}p/${counts[cls].g}g</span>
      <span class="instance-nav">
        <button class="prev-instance" ${available.length ? "" : "disabled"} title="Previous ${esc(cls)}">‹</button>
        <span>${current}/${available.length}</span>
        <button class="next-instance" ${available.length ? "" : "disabled"} title="Next ${esc(cls)}">›</button>
      </span>`;
    row.querySelector(".class-visible").onchange = (e) => {
      S.classOn[cls] = e.target.checked;
      requestDraw();
    };
    row.querySelector(".prev-instance").onclick = () => focusPrediction(cls, -1);
    row.querySelector(".next-instance").onclick = () => focusPrediction(cls, 1);
    el.appendChild(row);
  }
}

function focusPrediction(cls, direction) {
  const items = navigationItems(cls);
  if (!items.length || !S.viewer || !S.viewer.isOpen()) return;
  const old = S.navIndex[cls] == null ? (direction > 0 ? -1 : 0) : S.navIndex[cls];
  const index = (old + direction + items.length) % items.length;
  S.navIndex[cls] = index;
  S.classOn[cls] = true;
  const item = items[index];
  const obj = item.obj;
  focusObject(item.kind, obj, true);
  rebuildClassList();
}

function focusObject(kind, obj, detailed = false) {
  S.selected = { kind, id: obj.id };
  showBoxInfo({ kind, obj }, detailed);
  if (!S.viewer || !S.viewer.isOpen()) {
    S.pendingFocus = { source: S.cur.source, stem: S.cur.stem, kind, obj };
    requestDraw();
    return;
  }
  const xs = obj.poly.map((p) => p[0]), ys = obj.poly.map((p) => p[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const pad = Math.max(24, Math.max(maxx - minx, maxy - miny) * 2.5);
  const rect = S.viewer.viewport.imageToViewportRectangle(
    minx - pad, miny - pad, maxx - minx + 2 * pad, maxy - miny + 2 * pad);
  S.viewer.viewport.fitBounds(rect, false);
  requestDraw();
}

function renderStats() {
  const fmt = (v) => (v == null ? "–" : v.toFixed(2));
  const table = (st, showClasses) => {
    const classes = showClasses
      ? orderedClasses(new Set([
          ...TRAINED_CLASSES,
          ...Object.keys(st).filter((k) => k !== "_overall"),
        ])) : [];
    let html = `<table><tr><th>${showClasses ? "class" : "mode"}</th><th class="tp-c">TP</th>
    <th class="fp-c">FP</th><th class="fn-c">FN</th><th>P</th><th>R</th><th>F1</th></tr>`;
    for (const c of classes) {
      const s = st[c] || { tp: 0, fp: 0, fn: 0, precision: 0, recall: 0, f1: 0 };
      html += `<tr><td>${esc(c)}</td><td class="tp-c">${s.tp}</td>
        <td class="fp-c">${s.fp}</td><td class="fn-c">${s.fn}</td>
        <td>${fmt(s.precision)}</td><td>${fmt(s.recall)}</td><td>${fmt(s.f1)}</td></tr>`;
    }
    const o = st._overall || { tp: 0, fp: 0, fn: 0 };
    html += `<tr class="overall"><td>${showClasses ? "all" : "objects"}</td>
      <td class="tp-c">${o.tp}</td><td class="fp-c">${o.fp}</td>
      <td class="fn-c">${o.fn}</td><td>${fmt(o.precision)}</td>
      <td>${fmt(o.recall)}</td><td>${fmt(o.f1)}</td></tr></table>`;
    return html;
  };
  if (S.stage1Eval)
    $("stage1-detection-stats-table").innerHTML = table(S.stage1Eval.stats, false);
  if (S.stage2DetectionEval)
    $("stage2-detection-stats-table").innerHTML = table(S.stage2DetectionEval.stats, false);
  if (S.stage2Eval)
    $("classification-stats-table").innerHTML = table(S.stage2Eval.stats, true);
}

/* ── overlay drawing ──────────────────────────────────────────────────────── */
let drawQueued = false;
function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}

function sizeOverlay() {
  const cv = $("overlay");
  const r = $("viewer-wrap").getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  cv.style.width = r.width + "px";
  cv.style.height = r.height + "px";
}

function draw() {
  const cv = $("overlay");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!S.viewer || !S.viewer.isOpen()) { $("zoom-level").textContent = "–"; return; }

  const dpr = window.devicePixelRatio || 1;
  // image->screen affine (no rotation): screen = origin + img * scale
  const vp = S.viewer.viewport;
  const p0 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0));
  const p1 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(1, 0));
  const scale = (p1.x - p0.x) * dpr, ox = p0.x * dpr, oy = p0.y * dpr;
  $("zoom-level").textContent = (scale / dpr).toFixed(scale / dpr < 1 ? 2 : 1) + "×";

  const W = cv.width, H = cv.height;
  const vis = (poly) => {
    let minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
    for (const [x, y] of poly) {
      const sx = ox + x * scale, sy = oy + y * scale;
      if (sx < minx) minx = sx; if (sx > maxx) maxx = sx;
      if (sy < miny) miny = sy; if (sy > maxy) maxy = sy;
    }
    return maxx >= 0 && maxy >= 0 && minx <= W && miny <= H
      ? { minx, miny, maxx, maxy } : null;
  };
  const trace = (poly) => {
    ctx.beginPath();
    ctx.moveTo(ox + poly[0][0] * scale, oy + poly[0][1] * scale);
    for (let i = 1; i < poly.length; i++)
      ctx.lineTo(ox + poly[i][0] * scale, oy + poly[i][1] * scale);
    ctx.closePath();
  };
  const label = (text, color, bb) => {
    const fs = 11 * dpr;
    ctx.font = `600 ${fs}px ui-monospace, monospace`;
    const w = ctx.measureText(text).width + 8 * dpr;
    const x = Math.max(0, bb.minx), y = Math.max(fs + 4 * dpr, bb.miny);
    ctx.fillStyle = "rgba(10,13,16,.82)";
    ctx.fillRect(x, y - fs - 4 * dpr, w, fs + 6 * dpr);
    ctx.fillStyle = color;
    ctx.fillText(text, x + 4 * dpr, y - 3 * dpr);
  };
  const evalOn = S.ui.evalMode && S.eval;
  const lw = 2 * dpr;                      // constant screen-space stroke → never pixelates

  // ── delete regions: always opaque black, never interactive ──
  for (const r of S.ann.delete_regions) {
    if (!vis(r.poly)) continue;
    trace(r.poly);
    ctx.fillStyle = "#000000";
    ctx.fill();
  }

  // ── GT: dotted class color; selection becomes a thick solid class-color line ──
  if (S.ui.showGt) {
    for (const g of S.ann.gt) {
      if (!S.classOn[g.cls]) continue;
      const bb = vis(g.poly);
      if (!bb) continue;
      const color = classColor(g.cls);
      let width = 2 * dpr;
      const selected = S.selected && S.selected.kind === "gt" && S.selected.id === g.id;
      if (selected) width = 5 * dpr;
      trace(g.poly);
      ctx.lineCap = "round";
      ctx.setLineDash(selected ? [] : [2 * dpr, 6 * dpr]);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.lineCap = "butt";
    }
  }

  // ── Stage 1 + Stage 2 predictions ──
  for (const p of activePredictions()) {
    if (p.in_delete || !S.classOn[p.cls]) continue;
    const bb = vis(p.poly);
    if (!bb) continue;
    let color = p.layer === "stage1" ? "#3b82f6" : classColor(p.cls);
    if (evalOn && p.layer === S.evalLayer) {
      const stt = S.eval.pred_status[p.id];
      if (stt === "below_conf") continue;
      color = EVAL_COLORS[stt] || color;
    }
    trace(p.poly);
    const selected = S.selected && S.selected.kind === "pred" && S.selected.id === p.id;
    ctx.lineWidth = selected ? 5 * dpr : (p.layer === "stage1" ? 2 * dpr : lw);
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  // The arrow-selected instance must remain visible when several boxes overlap.
  // Redraw its real polygon last so a TP/another prediction cannot cover an FP.
  if (S.selected) {
    let obj = null, color = "#ffffff";
    if (S.selected.kind === "pred") {
      obj = activePredictions().find((p) => p.id === S.selected.id);
      if (obj && (obj.in_delete || !S.classOn[obj.cls] ||
          (evalOn && obj.layer === S.evalLayer && S.eval.pred_status[obj.id] === "below_conf"))) obj = null;
      if (obj) {
        color = obj.layer === "stage1" ? "#3b82f6" : classColor(obj.cls);
        if (evalOn && obj.layer === S.evalLayer)
          color = EVAL_COLORS[S.eval.pred_status[obj.id]] || color;
      }
    } else if (S.ui.showGt) {
      obj = S.ann.gt.find((g) => g.id === S.selected.id);
      if (obj && !S.classOn[obj.cls]) obj = null;
      if (obj) color = classColor(obj.cls);
    }
    if (obj && vis(obj.poly)) {
      trace(obj.poly);
      ctx.setLineDash([]);
      ctx.lineCap = "round";
      ctx.lineWidth = 5 * dpr;
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.lineCap = "butt";
    }
  }
}

/* ── box picking (click) ──────────────────────────────────────────────────── */
function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}
function polyArea(poly) {
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++)
    a += (poly[j][0] + poly[i][0]) * (poly[j][1] - poly[i][1]);
  return Math.abs(a / 2);
}

function handlePick(ix, iy) {
  const hits = [];
  if (S.ui.showStage1 || S.ui.showStage2)
    for (const p of activePredictions()) {
      if (p.in_delete || !S.classOn[p.cls]) continue;
      if (S.ui.evalMode && S.eval && p.layer === S.evalLayer &&
          S.eval.pred_status[p.id] === "below_conf") continue;
      if (pointInPoly(ix, iy, p.poly)) hits.push({ kind: "pred", obj: p, area: polyArea(p.poly) });
    }
  if (S.ui.showGt)
    for (const g of S.ann.gt) {
      if (!S.classOn[g.cls]) continue;
      if (pointInPoly(ix, iy, g.poly)) hits.push({ kind: "gt", obj: g, area: polyArea(g.poly) });
    }
  if (!hits.length) { S.selected = null; hideBoxInfo(); requestDraw(); return; }
  hits.sort((a, b) => a.area - b.area);          // smallest box wins (nested objects)
  const h = hits[0];
  S.selected = { kind: h.kind, id: h.obj.id };
  showBoxInfo(h);
  requestDraw();
}

function showBoxInfo(h, detailed = false) {
  const o = h.obj;
  let lines = [];
  let detailedLines = [];
  if (h.kind === "pred") {
    lines.push(`${o.layer === "stage1" ? "STAGE 1" : "STAGE 2"} PREDICTION`,
               `class: ${o.cls}${o.raw_cls ? `  (source: ${o.raw_cls})` : ""}`,
               `score: ${o.score}${o.raw != null ? `  (raw fused: ${o.raw})` : ""}`);
    if (detailed && o.layer === "stage2") {
      if (o.top_classes && o.top_classes.length) {
        lines.push("", "TOP 3 CLASSES");
        o.top_classes.forEach((entry, index) => {
          lines.push(`${index + 1}. ${entry.cls}: ${Number(entry.score).toFixed(4)}`);
        });
      } else {
        lines.push("", "top-3 scores: unavailable");
      }
    }
    if (S.eval && o.layer === S.evalLayer) {
      const stt = S.eval.pred_status[o.id];
      lines.push(`eval:  ${stt}`);
      if (S.eval.matches[o.id] != null) {
        const gtId = S.eval.matches[o.id];
        const matched = S.ann.gt.find((g) => String(g.id) === String(gtId));
        lines.push(`matched GT #${gtId}`);
        if (matched && matched.instance_id) lines.push(`instance ID: ${matched.instance_id}`);
      }
      if (stt === "FP") lines.push("instance ID: unavailable (unmatched prediction)");
    }
  } else {
    const d = o.details || {};
    lines.push("GROUND TRUTH",
               `instance ID: ${o.instance_id || "unavailable"}`,
               `labels_simple class: ${o.cls}`);
    if (detailed) {
      if (d.coordinates) detailedLines.push(`coordinates: ${d.coordinates.join(" ")}`);
      const detailOrder = [
        "super_category", "fine_label", "Normal", "Snow", "Dark", "Quality",
        "Functionality", "Position", "P_Camo", "P_Cut", "Remarks", "User",
        "Is_Empty(ML)", "SOC", "Camo Instance", "Background Matching", "difficult",
      ];
      for (const key of detailOrder)
        detailedLines.push(`${key}: ${d[key] ?? "unavailable"}`);
    }
    if (S.eval) lines.push(`eval:  ${S.eval.gt_status[o.id]}`);
  }
  if (detailed) {
    const [x, y] = o.poly[0];
    lines.push(`corner: (${Math.round(x)}, ${Math.round(y)}) px`);
  }
  const el = $("box-info");
  el.innerHTML = `<span class="close" onclick="this.parentElement.classList.add('hidden')">✕</span>`
    + `<div class="box-summary">${esc(lines.join("\n"))}</div>`
    + (detailedLines.length
      ? `<details class="gt-detail-fold"><summary>▾ labels_detailed</summary>`
        + `<div class="gt-detail-body">${esc(detailedLines.join("\n"))}</div></details>`
      : "");
  el.classList.remove("hidden");
}
function hideBoxInfo() { $("box-info").classList.add("hidden"); }

/* ── modals ───────────────────────────────────────────────────────────────── */
function openModal(title, bodyHTML, actions) {
  $("modal-title").textContent = title;
  $("modal-body").innerHTML = bodyHTML;
  const act = $("modal-actions");
  act.innerHTML = "";
  for (const a of actions) {
    const b = document.createElement("button");
    b.textContent = a.label;
    if (a.primary) b.className = "primary";
    b.onclick = a.onclick;
    act.appendChild(b);
  }
  $("modal-backdrop").classList.remove("hidden");
}
function closeModal() {
  $("modal-backdrop").classList.add("hidden");
  clearInterval(S.jobsModalPoll);
}

/* upload */
function uploadDialog() {
  openModal("Upload image",
    `<label>Image file (png / jpg / tif)
       <input type="file" id="up-file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"></label>
     <div class="hint">The image must already be in the pixel space your model expects
     (e.g. the YOLO+DINO ensemble needs 2× super-resolved imagery). The GUI never resizes
     your file — it is displayed and inferred at its native dimensions.</div>
     <div id="up-status" class="hint"></div>`,
    [
      { label: "Cancel", onclick: closeModal },
      { label: "Upload", primary: true, onclick: doUpload },
    ]);
}
async function doUpload() {
  const f = $("up-file").files[0];
  if (!f) return;
  $("up-status").textContent = `uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    closeModal();
    await loadScenes(false);
    openScene(res.source, res.stem);
  } catch (e) {
    $("up-status").textContent = "✗ " + e.message;
  }
}

/* run inference */
async function inferDialog() {
  if (!S.cur) return;
  const { adapters } = await api("/api/adapters");
  if (!adapters.length) { alert("no model adapters registered"); return; }
  const opts = adapters.map((a) =>
    `<option value="${esc(a.name)}">${esc(a.display_name)}</option>`).join("");
  openModal(`Run inference — ${S.cur.stem}`,
    `<label>Model <select id="inf-adapter">${opts}</select></label>
     <div class="adapter-desc" id="inf-desc"></div>
     <div id="inf-opts"></div>
     <label>GPU device <input type="text" id="inf-device" value="0"></label>
     <div class="hint">Jobs run one at a time on the server; watch progress under “Jobs”.
     When the job finishes, this scene's prediction overlay refreshes automatically.</div>`,
    [
      { label: "Cancel", onclick: closeModal },
      { label: "Start job", primary: true, onclick: () => startInfer(adapters) },
    ]);
  const renderAd = () => {
    const a = adapters.find((x) => x.name === $("inf-adapter").value);
    $("inf-desc").textContent = a.description;
    $("inf-opts").innerHTML = (a.options || []).map((o) => o.type === "bool"
      ? `<label><input type="checkbox" data-opt="${esc(o.name)}" ${o.default ? "checked" : ""}>
           ${esc(o.label)}</label><div class="hint">${esc(o.help || "")}</div>`
      : `<label>${esc(o.label)}<input type="text" data-opt="${esc(o.name)}"
           value="${esc(o.default ?? "")}"></label>`).join("");
  };
  $("inf-adapter").onchange = renderAd;
  renderAd();
}
async function startInfer() {
  const extra = {};
  document.querySelectorAll("#inf-opts [data-opt]").forEach((el) => {
    extra[el.dataset.opt] = el.type === "checkbox" ? el.checked : el.value;
  });
  try {
    const job = await api("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: S.cur.source, stem: S.cur.stem,
        adapter: $("inf-adapter").value, device: $("inf-device").value, extra,
      }),
    });
    watchJob(job.id);
    jobsDialog();
  } catch (e) { alert("failed to queue job: " + e.message); }
}

/* background watcher: refresh overlays when a job for the open scene completes */
function watchJob(jobId) {
  if (S.jobWatch[jobId]) return;
  S.jobWatch[jobId] = setInterval(async () => {
    let j;
    try { j = await api(`/api/jobs/${jobId}`); } catch { return; }
    if (["done", "failed", "error"].includes(j.state)) {
      clearInterval(S.jobWatch[jobId]);
      delete S.jobWatch[jobId];
      await loadScenes();
      if (j.state === "done" && S.cur &&
          S.cur.source === j.source && S.cur.stem === j.stem) {
        S.cur.has_pred = true;
        S.cur.has_stage2 = true;
        await reloadAnnotations();
      }
    }
  }, 3000);
}

/* datasets: point the GUI at prepared image / pred / GT folders */
async function datasetsDialog() {
  const { sources } = await api("/api/sources");
  const rows = sources.map((s) => `
    <div class="job-row"><div class="jr-head">
      <b>${esc(s.title)}</b><span class="hint">${esc(s.name)}</span>
      ${s.builtin ? `<span class="hint" style="margin-left:auto">built-in</span>`
                  : `<button data-rm="${esc(s.name)}" style="margin-left:auto;padding:1px 8px;font-size:11px">remove</button>`}
    </div>
    <div class="hint">img: ${esc(s.images)}${s.preds ? ` · pred: ${esc(s.preds)}` : ""}${s.gt ? ` · gt: ${esc(s.gt)}` : ""}
      · scales p×${s.pred_scale} / g×${s.gt_scale}</div></div>`).join("");
  openModal("Datasets",
    `${rows}
     <h4 style="margin:14px 0 2px">Add a dataset</h4>
     <div class="hint">Point at folders on the server. Pred/GT files must be DOTA txt named
       <b>&lt;image stem&gt;.txt</b>. Scales multiply stored coords into the displayed image's
       pixel space (e.g. GT annotated on native imagery shown at 2× SR → GT scale 2.0).</div>
     <label>Name (id) <input type="text" id="ds-name" placeholder="my_run_01"></label>
     <label>Title (sidebar heading) <input type="text" id="ds-title" placeholder="My run 01"></label>
     <label>Images folder <input type="text" id="ds-images" placeholder="/data/…/images_2x"></label>
     <label>Image glob <input type="text" id="ds-glob" value="*.png"></label>
     <label>Strip suffix (image stem → file stem) <input type="text" id="ds-strip" placeholder="_SR2x"></label>
     <label>Predictions folder <input type="text" id="ds-preds" placeholder="/data/…/outputs"></label>
     <label>Ground-truth folder <input type="text" id="ds-gt" placeholder="/data/…/labels"></label>
     <label>Delete-regions folder (optional) <input type="text" id="ds-delete" placeholder="/data/…/labels_delete"></label>
     <label>Pred scale <input type="text" id="ds-pscale" value="1.0"></label>
     <label>GT scale <input type="text" id="ds-gscale" value="1.0"></label>
     <div id="ds-status" class="hint"></div>`,
    [
      { label: "Close", onclick: closeModal },
      { label: "Add dataset", primary: true, onclick: addDataset },
    ]);
  document.querySelectorAll("[data-rm]").forEach((b) => {
    b.onclick = async () => {
      if (!confirm(`Unregister dataset '${b.dataset.rm}'? (no files are deleted)`)) return;
      await api(`/api/sources/${b.dataset.rm}`, { method: "DELETE" });
      closeModal();
      await loadScenes();
      datasetsDialog();
    };
  });
}
async function addDataset() {
  const val = (id) => $(id).value.trim();
  try {
    await api("/api/sources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: val("ds-name"), title: val("ds-title"), images: val("ds-images"),
        image_glob: val("ds-glob"), strip_suffix: val("ds-strip"),
        preds: val("ds-preds"), gt: val("ds-gt"), delete: val("ds-delete"),
        pred_scale: val("ds-pscale"), gt_scale: val("ds-gscale"),
      }),
    });
    closeModal();
    await loadScenes();
  } catch (e) {
    $("ds-status").textContent = "✗ " + e.message;
  }
}

/* jobs list */
async function jobsDialog() {
  openModal("Inference jobs", `<div id="jobs-body">loading…</div>`,
    [{ label: "Close", onclick: closeModal }]);
  const render = async () => {
    let data;
    try { data = await api("/api/jobs"); } catch { return; }
    const openLogs = new Set(
      [...document.querySelectorAll(".job-log:not(.hidden)")].map((e) => e.dataset.job));
    const body = $("jobs-body");
    if (!body) return;
    if (!data.jobs.length) { body.innerHTML = `<div class="hint">no jobs yet</div>`; return; }
    body.innerHTML = data.jobs.map((j) => {
      const dur = j.finished && j.started ? ` · ${Math.round(j.finished - j.started)}s` : "";
      return `<div class="job-row">
        <div class="jr-head">
          <span class="job-state ${esc(j.state)}">${esc(j.state)}</span>
          <span>${esc(j.stem)}</span><span class="hint">${esc(j.adapter)}${dur}</span>
          <button data-log="${esc(j.id)}" style="margin-left:auto;padding:1px 8px;font-size:11px">log</button>
        </div>
        ${j.error ? `<div class="hint" style="color:#ff8080">${esc(j.error)}</div>` : ""}
        <div class="job-log hidden" data-job="${esc(j.id)}"></div>
      </div>`;
    }).join("");
    body.querySelectorAll("[data-log]").forEach((b) => {
      b.onclick = async () => {
        const el = body.querySelector(`.job-log[data-job="${b.dataset.log}"]`);
        el.classList.toggle("hidden");
        if (!el.classList.contains("hidden")) {
          const j = await api(`/api/jobs/${b.dataset.log}`);
          el.textContent = j.log_tail || "(no output yet)";
          el.scrollTop = el.scrollHeight;
        }
      };
    });
    for (const id of openLogs) {
      const el = body.querySelector(`.job-log[data-job="${id}"]`);
      if (el) {
        el.classList.remove("hidden");
        api(`/api/jobs/${id}`).then((j) => {
          el.textContent = j.log_tail || "(no output yet)";
          el.scrollTop = el.scrollHeight;
        });
      }
    }
  };
  await render();
  clearInterval(S.jobsModalPoll);
  S.jobsModalPoll = setInterval(render, 2500);
}

/* ── wire up static controls ─────────────────────────────────────────────── */
function bind() {
  const runImageSearch = () => searchInstance("image", "image-id-search");
  const runInstanceSearch = () => searchInstance("full", "instance-id-search");
  $("btn-image-id-search").onclick = runImageSearch;
  $("btn-instance-id-search").onclick = runInstanceSearch;
  $("image-id-search").onkeydown = (e) => { if (e.key === "Enter") runImageSearch(); };
  $("instance-id-search").onkeydown = (e) => { if (e.key === "Enter") runInstanceSearch(); };
  for (const id of ["image-id-search", "instance-id-search"]) {
    $(id).oninput = (e) => {
      if (!e.target.value.trim()) $("instance-results").classList.add("hidden");
    };
  }
  $("btn-upload").onclick = uploadDialog;
  $("btn-infer").onclick = inferDialog;
  $("btn-jobs").onclick = jobsDialog;
  $("modal-backdrop").onclick = (e) => { if (e.target.id === "modal-backdrop") closeModal(); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

  const reloadSafely = () => reloadAnnotations().catch((e) => {
    console.warn("annotation reload failed:", e.message);
    alert(e.message);
  });
  const reloadAnnotationsDebounced = debounce(reloadSafely, 350);
  const stageChanged = () => {
    updateConfidenceControls();
    syncVisualEvaluation();
    rebuildClassList();
    requestDraw();
  };
  $("show-stage1").onchange = (e) => { S.ui.showStage1 = e.target.checked; stageChanged(); };
  $("show-stage2").onchange = (e) => { S.ui.showStage2 = e.target.checked; stageChanged(); };
  $("show-gt").onchange = (e) => { S.ui.showGt = e.target.checked; requestDraw(); };
  $("eval-mode").onchange = (e) => {
    S.evalTouched = true;
    S.ui.evalMode = e.target.checked;
    if (S.ui.evalMode) rebuildClassList();
    $("eval-legend").classList.toggle("hidden", !S.ui.evalMode);
    if (S.ui.evalMode && !$("show-gt").checked) { $("show-gt").checked = true; S.ui.showGt = true; }
    syncVisualEvaluation();
    rebuildClassList();
    requestDraw();
  };
  $("conf-thr").oninput = (e) => {
    S.ui.conf = parseFloat(e.target.value);
    $("conf-val").textContent = S.ui.conf.toFixed(2);
    reloadAnnotationsDebounced();
  };
  $("per-class-conf").onchange = (e) => {
    S.ui.perClassConf = e.target.checked;
    updateConfidenceControls();
    reloadSafely();
  };
  $("suppress-stage1").onchange = (e) => {
    S.ui.suppressStage1 = e.target.checked;
    reloadSafely();
  };
  $("iou-thr").oninput = (e) => {
    S.ui.iou = parseFloat(e.target.value);
    $("iou-val").textContent = S.ui.iou.toFixed(2);
    refreshEvalDebounced();
  };
  $("cls-all").onclick = () => { for (const c in S.classOn) S.classOn[c] = true; rebuildClassList(); requestDraw(); };
  $("cls-none").onclick = () => { for (const c in S.classOn) S.classOn[c] = false; rebuildClassList(); requestDraw(); };

  $("zoom-in").onclick = () => S.viewer && S.viewer.viewport.zoomBy(1.5).applyConstraints();
  $("zoom-out").onclick = () => S.viewer && S.viewer.viewport.zoomBy(1 / 1.5).applyConstraints();
  $("zoom-home").onclick = () => S.viewer && S.viewer.viewport.goHome();

  new ResizeObserver(() => { sizeOverlay(); requestDraw(); }).observe($("viewer-wrap"));
}

bind();
loadScenes(false).catch((e) => alert("failed to load scenes: " + e.message));
