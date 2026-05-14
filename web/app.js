/* elevator-rl dashboard — vanilla JS + Canvas.
 *
 * Loads /traces/manifest.json, lists episodes grouped by seed, replays three
 * policies side by side on a shared playback clock with a scrub bar. No
 * framework, no backend.
 */
(() => {
  "use strict";

  const TRACE_DIR = "traces";
  const POLICIES = ["scan", "nearest", "random"];
  const POLICY_LABELS = { scan: "SCAN", nearest: "Nearest-Car", random: "Random" };
  const COLORS = {
    bg: "#1c2029",
    floor: "#262b36",
    rail: "#323845",
    car: "#4ea0ff",
    carDir: "#62d18b",
    carRev: "#ff9a4e",
    carService: "#ffd45e",
    hallUp: "#62d18b",
    hallDown: "#ff7e7e",
    text: "#e7e9ec",
    muted: "#8a92a0",
  };

  /** Mutable state. */
  const state = {
    manifest: null,
    seedsAvailable: [], // unique sorted seeds across all policies
    seed: null,
    traces: {}, // { policy: trace }
    step: 0,
    totalSteps: 0,
    playing: false,
    speed: 2,
    lastTick: 0,
    highlights: [],
  };

  const dom = {
    seedPicker: document.getElementById("seed-picker"),
    scrub: document.getElementById("scrub"),
    stepLabel: document.getElementById("step-label"),
    playPause: document.getElementById("play-pause"),
    speed: document.getElementById("speed"),
    metricsLine: document.getElementById("metrics-line"),
    highlightList: document.getElementById("highlight-list"),
    lanes: {},
  };
  document.querySelectorAll(".lane").forEach((el) => {
    const p = el.dataset.policy;
    dom.lanes[p] = {
      root: el,
      summary: el.querySelector(".lane-summary"),
      canvas: el.querySelector("canvas"),
      ctx: el.querySelector("canvas").getContext("2d"),
    };
  });

  // ------------------------------------------------------------------ data

  async function loadManifest() {
    const res = await fetch(`${TRACE_DIR}/manifest.json`);
    if (!res.ok) throw new Error(`manifest.json HTTP ${res.status}`);
    return res.json();
  }

  async function loadTrace(file) {
    const res = await fetch(`${TRACE_DIR}/${file}`);
    if (!res.ok) throw new Error(`${file} HTTP ${res.status}`);
    return res.json();
  }

  function collectSeeds(manifest) {
    const seen = new Set();
    for (const ep of manifest.episodes) seen.add(ep.seed);
    return [...seen].sort((a, b) => a - b);
  }

  function findFile(manifest, policy, seed) {
    return manifest.episodes.find((e) => e.policy === policy && e.seed === seed);
  }

  // ----------------------------------------------------------------- render

  function fmt(n, digits = 1) {
    if (n == null || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits);
  }

  /** Draw one building. */
  function drawBuilding(ctx, w, h, trace, step) {
    const F = trace.num_floors;
    const N = trace.num_elevators;
    const s = trace.steps[Math.min(step, trace.steps.length - 1)];

    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    const padTop = 14;
    const padBottom = 18;
    const padLeft = 36;
    const padRight = 12;
    const usableH = h - padTop - padBottom;
    const usableW = w - padLeft - padRight;
    const floorHeight = usableH / F;
    const colWidth = usableW / (N + 1); // last col for hall + waiting

    // floor rows
    ctx.fillStyle = COLORS.muted;
    ctx.font = "11px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    for (let f = 0; f < F; f++) {
      const yTop = padTop + (F - 1 - f) * floorHeight;
      const yMid = yTop + floorHeight / 2;
      // floor line
      ctx.strokeStyle = COLORS.floor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, yTop + floorHeight);
      ctx.lineTo(w - padRight, yTop + floorHeight);
      ctx.stroke();
      // floor label
      ctx.fillStyle = COLORS.muted;
      ctx.textAlign = "right";
      ctx.fillText(String(f), padLeft - 6, yMid);
    }

    // elevator shafts (rails)
    for (let i = 0; i < N; i++) {
      const cx = padLeft + colWidth * i + colWidth / 2;
      ctx.strokeStyle = COLORS.rail;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, padTop);
      ctx.lineTo(cx, padTop + usableH);
      ctx.stroke();
    }

    // elevator cars
    for (let i = 0; i < N; i++) {
      const pos = s.elevator_positions[i]; // float position
      const dir = s.elevator_directions[i];
      const load = s.elevator_loads[i];
      const servicing = s.elevator_servicing[i];

      const cx = padLeft + colWidth * i + colWidth / 2;
      const cy = padTop + (F - 1 - pos) * floorHeight + floorHeight / 2;

      const boxW = Math.min(colWidth * 0.6, 28);
      const boxH = Math.min(floorHeight * 0.6, 22);

      let fill = COLORS.car;
      if (servicing) fill = COLORS.carService;
      else if (dir > 0) fill = COLORS.carDir;
      else if (dir < 0) fill = COLORS.carRev;

      ctx.fillStyle = fill;
      ctx.fillRect(cx - boxW / 2, cy - boxH / 2, boxW, boxH);

      // load text
      ctx.fillStyle = "#0e1014";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(load), cx, cy);

      // direction arrow above box
      if (!servicing && dir !== 0) {
        ctx.fillStyle = fill;
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(dir > 0 ? "↑" : "↓", cx, cy - boxH / 2 - 7);
      }
    }

    // hall calls + waiting count on the right of each floor
    const hallX = padLeft + colWidth * N + 6;
    for (let f = 0; f < F; f++) {
      const yTop = padTop + (F - 1 - f) * floorHeight;
      const yMid = yTop + floorHeight / 2;
      const up = s.hall_calls_up[f];
      const dn = s.hall_calls_down[f];
      const wait = s.waiting_per_floor[f];

      // up triangle
      if (up) {
        ctx.fillStyle = COLORS.hallUp;
        ctx.beginPath();
        ctx.moveTo(hallX, yMid - 1);
        ctx.lineTo(hallX + 7, yMid - 1);
        ctx.lineTo(hallX + 3.5, yMid - 7);
        ctx.closePath();
        ctx.fill();
      }
      // down triangle
      if (dn) {
        ctx.fillStyle = COLORS.hallDown;
        ctx.beginPath();
        ctx.moveTo(hallX, yMid + 1);
        ctx.lineTo(hallX + 7, yMid + 1);
        ctx.lineTo(hallX + 3.5, yMid + 7);
        ctx.closePath();
        ctx.fill();
      }
      // waiting count
      if (wait > 0) {
        ctx.fillStyle = COLORS.text;
        ctx.font = "11px ui-monospace, monospace";
        ctx.textAlign = "left";
        ctx.fillText(String(wait), hallX + 12, yMid);
      }
    }

    // footer: cumulative reward proxy + completions so far
    let cumComp = 0;
    for (let t = 0; t <= step && t < trace.steps.length; t++) {
      cumComp += trace.steps[t].completions_this_step;
    }
    ctx.fillStyle = COLORS.muted;
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`t=${s.t} done=${cumComp}`, padLeft, h - 4);
  }

  /** Update per-lane summary line: AWT so far, rank vs others. */
  function updateLaneSummaries(step) {
    // current cumulative wait approximation: not the true running AWT
    // (which is only finalized at episode end), but a useful relative ranking.
    // Use the final-episode AWT for comparison.
    const finals = {};
    for (const p of POLICIES) finals[p] = state.traces[p]?.metrics.avg_wait_s;
    const ranking = POLICIES.slice().sort((a, b) => finals[a] - finals[b]);

    for (const p of POLICIES) {
      const t = state.traces[p];
      if (!t) continue;
      const m = t.metrics;
      const ord = ranking.indexOf(p);
      let label = "";
      if (ord === 0) label = `<span class="winner">★ winner</span>`;
      else if (ord === POLICIES.length - 1) label = `<span class="loser">last</span>`;
      else label = `2nd`;
      dom.lanes[p].summary.innerHTML =
        `${label} · final AWT <b>${fmt(m.avg_wait_s)}s</b> · ` +
        `journey ${fmt(m.avg_journey_s)}s · completed ${fmt(m.completed, 0)}`;
    }
  }

  function updateMetricsLine() {
    if (!POLICIES.every((p) => state.traces[p])) {
      dom.metricsLine.textContent = "";
      return;
    }
    const summary = POLICIES.map((p) => {
      const m = state.traces[p].metrics;
      return `${POLICY_LABELS[p]} AWT ${fmt(m.avg_wait_s)}s`;
    }).join(" · ");
    dom.metricsLine.innerHTML = `<b>seed ${state.seed}</b> · ${summary}`;
  }

  function render() {
    for (const p of POLICIES) {
      const t = state.traces[p];
      if (!t) continue;
      const { ctx, canvas } = dom.lanes[p];
      drawBuilding(ctx, canvas.width, canvas.height, t, state.step);
    }
    dom.scrub.value = String(state.step);
    dom.stepLabel.textContent = `step ${state.step}`;
  }

  // ---------------------------------------------------------------- playback

  function tick(now) {
    if (!state.playing) return;
    if (!state.lastTick) state.lastTick = now;
    const dt = (now - state.lastTick) / 1000;
    state.lastTick = now;
    // advance by speed * dt steps per second (real wall time)
    // The sim is 1 step = 1 sim-second, so speed=1 ≈ real time, speed=10 = 10×.
    const advance = Math.max(1, Math.round(state.speed * dt * 10));
    state.step = Math.min(state.totalSteps - 1, state.step + advance);
    render();
    if (state.step >= state.totalSteps - 1) {
      stopPlayback();
      return;
    }
    requestAnimationFrame(tick);
  }

  function startPlayback() {
    if (state.step >= state.totalSteps - 1) state.step = 0;
    state.playing = true;
    state.lastTick = 0;
    dom.playPause.textContent = "❚❚ pause";
    requestAnimationFrame(tick);
  }
  function stopPlayback() {
    state.playing = false;
    dom.playPause.textContent = "▶ play";
  }
  function togglePlayback() {
    if (state.playing) stopPlayback();
    else startPlayback();
  }

  // ----------------------------------------------------------------- highlights

  /** Compute a few notable episodes/moments for the user to jump to. */
  function buildHighlights(manifest) {
    const out = [];
    // group episodes by seed
    const bySeed = {};
    for (const ep of manifest.episodes) {
      bySeed[ep.seed] ??= {};
      bySeed[ep.seed][ep.policy] = ep;
    }

    // Seeds where Nearest-Car catastrophically underperforms SCAN
    for (const [seed, byPolicy] of Object.entries(bySeed)) {
      const scan = byPolicy.scan?.metrics.avg_wait_s;
      const near = byPolicy.nearest?.metrics.avg_wait_s;
      if (!scan || !near) continue;
      if (near > scan * 2.5) {
        out.push({
          seed: Number(seed),
          step: Math.floor(0.4 * (manifest.config?.total_steps || 1200)),
          title: `Seed ${seed}: Nearest-Car stalls`,
          meta: `SCAN AWT ${fmt(scan)}s vs Nearest ${fmt(near)}s (${fmt(near / scan, 1)}× worse)`,
        });
      }
    }
    // Best SCAN episode (lowest AWT)
    const scans = manifest.episodes.filter((e) => e.policy === "scan");
    if (scans.length) {
      const bestScan = scans.reduce((a, b) =>
        a.metrics.avg_wait_s < b.metrics.avg_wait_s ? a : b
      );
      out.push({
        seed: bestScan.seed,
        step: 0,
        title: `SCAN's best run (seed ${bestScan.seed})`,
        meta: `AWT ${fmt(bestScan.metrics.avg_wait_s)}s, completed ${fmt(bestScan.metrics.completed, 0)}`,
      });
    }
    // Best Nearest-Car episode (lowest AWT)
    const nears = manifest.episodes.filter((e) => e.policy === "nearest");
    if (nears.length) {
      const bestNear = nears.reduce((a, b) =>
        a.metrics.avg_wait_s < b.metrics.avg_wait_s ? a : b
      );
      if (bestNear.metrics.avg_wait_s) {
        out.push({
          seed: bestNear.seed,
          step: 0,
          title: `Nearest-Car's best run (seed ${bestNear.seed})`,
          meta: `AWT ${fmt(bestNear.metrics.avg_wait_s)}s — it can win sometimes`,
        });
      }
    }
    return out.slice(0, 6);
  }

  function renderHighlights() {
    dom.highlightList.innerHTML = "";
    for (const h of state.highlights) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="h-title">${h.title}</span><span class="h-meta">${h.meta} · jump to step ${h.step}</span>`;
      li.addEventListener("click", () => {
        loadSeed(h.seed).then(() => {
          state.step = h.step;
          render();
        });
      });
      dom.highlightList.appendChild(li);
    }
  }

  // --------------------------------------------------------------- seed load

  async function loadSeed(seed) {
    state.seed = seed;
    state.step = 0;
    state.traces = {};
    const promises = POLICIES.map(async (p) => {
      const entry = findFile(state.manifest, p, seed);
      if (!entry) return;
      const t = await loadTrace(entry.file);
      state.traces[p] = t;
    });
    await Promise.all(promises);

    const t = state.traces[POLICIES[0]];
    if (t) {
      state.totalSteps = t.total_steps;
      dom.scrub.max = String(t.total_steps - 1);
    }

    dom.seedPicker.value = String(seed);
    updateLaneSummaries(state.step);
    updateMetricsLine();
    render();

    // reflect in URL for deep linking
    const url = new URL(window.location.href);
    url.searchParams.set("seed", String(seed));
    window.history.replaceState({}, "", url);
  }

  // ------------------------------------------------------------------- init

  async function init() {
    try {
      state.manifest = await loadManifest();
    } catch (err) {
      document.body.innerHTML = `<pre style="color:#ff7e7e">Failed to load traces. Did you run \`python -m scripts.emit_traces --out web/traces\`?\n\n${err.message}</pre>`;
      return;
    }
    state.seedsAvailable = collectSeeds(state.manifest);
    state.highlights = buildHighlights(state.manifest);
    renderHighlights();

    // populate seed dropdown
    for (const seed of state.seedsAvailable) {
      const opt = document.createElement("option");
      opt.value = String(seed);
      opt.textContent = `seed ${seed}`;
      dom.seedPicker.appendChild(opt);
    }

    // url params: ?seed=X&step=Y
    const params = new URLSearchParams(window.location.search);
    let seed = Number(params.get("seed"));
    if (!state.seedsAvailable.includes(seed)) seed = state.seedsAvailable[0];

    await loadSeed(seed);
    const step = Number(params.get("step"));
    if (Number.isFinite(step) && step > 0 && step < state.totalSteps) {
      state.step = step;
      render();
    }

    // wire controls
    dom.seedPicker.addEventListener("change", (e) => {
      stopPlayback();
      loadSeed(Number(e.target.value));
    });
    dom.playPause.addEventListener("click", togglePlayback);
    dom.scrub.addEventListener("input", (e) => {
      stopPlayback();
      state.step = Number(e.target.value);
      render();
    });
    dom.speed.addEventListener("change", (e) => {
      state.speed = Number(e.target.value);
    });

    // keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlayback();
      } else if (e.code === "ArrowLeft") {
        stopPlayback();
        state.step = Math.max(0, state.step - 30);
        render();
      } else if (e.code === "ArrowRight") {
        stopPlayback();
        state.step = Math.min(state.totalSteps - 1, state.step + 30);
        render();
      }
    });

    // smooth scroll for #about
    document.getElementById("about-link").addEventListener("click", (e) => {
      e.preventDefault();
      document.getElementById("about").scrollIntoView({ behavior: "smooth" });
    });
  }

  init().catch((err) => {
    console.error(err);
    document.body.insertAdjacentHTML(
      "beforeend",
      `<pre style="color:#ff7e7e;background:#15181f;padding:12px;border-radius:6px">init error: ${err.message}</pre>`
    );
  });
})();
