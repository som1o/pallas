const POLL_INTERVAL_ACTIVE_MS = 300;
const POLL_INTERVAL_IDLE_MS = 700;
const MODEL_REFRESH_INTERVAL_MS = 1800;
const API_TIMEOUT_MS = 3500;

const state = {
  tick: 0,
  countries: [],
  decisions: [],
  messages: [],
  leaderboard: [],
  battle: {
    active: false,
    elapsed_sec: 0,
    remaining_sec: 0,
    min_duration_sec: 60,
    max_duration_sec: 180,
    target_duration_sec: 180
  },
  distributed: { node_id: 0, total_nodes: 1, bind_host: "0.0.0.0", bind_port: 0 },
  diagnostics: {
    tick: 0,
    battle_active: false,
    distributed: {
      node_id: 0,
      total_nodes: 1,
      bind_host: "0.0.0.0",
      bind_port: 0,
      exchange_count: 0,
      packets_sent: 0,
      packets_received: 0,
      packets_dropped: 0,
      peer_count: 0
    }
  },
  readiness: { ready: false, missing_country_ids: [] },
  countrySlots: [],
  uploadedModels: {},
  competition: { finalists: [], eliminated: [], winner_country_name: "", winner_country_id: 0, winner_model: "" },
  map: { width: 0, height: 0, cells: [] },
  apiMeta: {
    api_version: 1,
    max_upload_bytes: 16 * 1024 * 1024,
    strategies: [],
    targeted_strategies: []
  },
  winnerCountryId: 0,
  winnerCountryName: "",
  modelLoadErrors: [],
  projectiles: [],
  explosions: [],
  smoke: [],
  embers: [],
  actionPulses: [],
  cameraShake: 0,
  ui: {
    refreshInFlight: false,
    uploadInFlight: false,
    pendingCountryUpload: null,
    lastAnimatedTick: -1,
    previousBattleActive: false,
    lastModelRefreshAt: 0,
    refreshFailures: 0,
    lastRefreshAt: 0,
    lastRefreshError: "",
    countryUploadSignature: "",
    modelPanelDirty: true,
    resultsOpenForTick: -1
  }
};

const visualConfig = {
  maxProjectiles: 96,
  maxExplosions: 120,
  maxSmoke: 220,
  maxEmbers: 260,
  pressureLevel: 1
};

const camera = {
  zoom: 1,
  minZoom: 0.7,
  maxZoom: 4,
  x: 0,
  y: 0,
  dragging: false,
  dragStartX: 0,
  dragStartY: 0,
  startX: 0,
  startY: 0,
  pointers: new Map(),
  pinchStartDistance: 0,
  pinchStartZoom: 1
};

const geometryCache = {
  key: "",
  centroids: new Map()
};

let lastFrame = performance.now();
let fpsAverage = 60;
let lowFpsStreak = 0;
let highFpsStreak = 0;
let refreshTimer = null;

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const statsEl = document.getElementById("stats");
const decisionsEl = document.getElementById("decisions");
const messagesEl = document.getElementById("messages");
const leaderboardEl = document.getElementById("leaderboard");
const distributedEl = document.getElementById("distributed");
const diagnosticsEl = document.getElementById("diagnostics");
const mapLegendEl = document.getElementById("mapLegend");
const strategicSummaryEl = document.getElementById("strategicSummary");
const countrySummaryEl = document.getElementById("countrySummary");
const uploadStatusEl = document.getElementById("uploadStatus");
const countryUploadGridEl = document.getElementById("countryUploadGrid");
const modelHierarchyEl = document.getElementById("modelHierarchy");
const modelReadinessEl = document.getElementById("modelReadiness");
const finalistsListEl = document.getElementById("finalistsList");
const eliminatedListEl = document.getElementById("eliminatedList");
const speedInput = document.getElementById("speedInput");
const speedValue = document.getElementById("speedValue");
const durationInput = document.getElementById("durationInput");
const resultsModalEl = document.getElementById("resultsModal");
const resultsSummaryEl = document.getElementById("resultsSummary");
const resultsLeaderboardEl = document.getElementById("resultsLeaderboard");
const resultsMessageEl = document.getElementById("resultsMessage");
const stepBtn = document.getElementById("stepBtn");
const startBtn = document.getElementById("startBtn");
const pauseBtn = document.getElementById("pauseBtn");
const tooltipEl = document.getElementById("countryTooltip");
const perfBadgeEl = document.getElementById("perfBadge");
const modelErrorsEl = document.getElementById("modelErrors");
const commandStatusEl = document.getElementById("commandStatus");
const overrideStrategyEl = document.getElementById("overrideStrategy");

const hiddenCountryFileInput = document.createElement("input");
hiddenCountryFileInput.type = "file";
hiddenCountryFileInput.accept = ".bin,application/octet-stream";
hiddenCountryFileInput.style.display = "none";
document.body.appendChild(hiddenCountryFileInput);

const COUNTRY_PALETTE = ["#db7a4a", "#55a18f", "#d3b862", "#5d7dd8", "#ad6ddd", "#85b65b", "#e58d5f", "#5ca8d6"];
let targetedStrategies = new Set([
  "attack",
  "negotiate",
  "transfer_weapons",
  "form_alliance",
  "betray",
  "cyber_operation",
  "sign_trade_agreement",
  "cancel_trade_agreement",
  "impose_embargo",
  "propose_defense_pact",
  "propose_non_aggression",
  "break_treaty",
  "request_intel",
  "deploy_units",
  "tactical_nuke",
  "strategic_nuke",
  "cyber_attack"
]);

const MAP_TAGS = {
  SEA: 1,
  STRATEGIC: 2,
  CHOKE_STRAIT: 4,
  CHOKE_CANAL: 8,
  MOUNTAIN_PASS: 16,
  RIVER_CROSSING: 32,
  PORT: 64
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function escapeText(value) {
  return value == null ? "" : String(value);
}

function formatCompactNumber(value) {
  return Math.round(asNumber(value, 0)).toLocaleString();
}

function milliToUnits(value) {
  return asNumber(value, 0) / 1000;
}

function milliToPercent(value) {
  return asNumber(value, 0) / 1000;
}

function countryUnits(country) {
  const units = country && country.units ? country.units : {};
  return {
    infantry: asNumber(units.infantry),
    armor: asNumber(units.armor),
    artillery: asNumber(units.artillery),
    airFighter: asNumber(units.air_fighter),
    airBomber: asNumber(units.air_bomber),
    navalSurface: asNumber(units.naval_surface),
    navalSubmarine: asNumber(units.naval_submarine)
  };
}

function formatUnitCount(value) {
  return Math.round(milliToUnits(value)).toLocaleString();
}

function relationshipCount(country) {
  return asArray(country.trade_partners).length +
    asArray(country.defense_pacts).length +
    asArray(country.non_aggression_pacts).length +
    asArray(country.trade_treaties).length;
}

function formatPower(country) {
  return Math.round(countryPower(country)).toLocaleString();
}

function resizeCanvas() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(480, Math.floor(rect.width * dpr));
  const height = Math.max(320, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    geometryCache.key = "";
  }
}

function countryColor(country) {
  if (!country) return "#778bad";
  if (country.color && /^#[0-9a-f]{6}$/i.test(country.color)) {
    return country.color;
  }
  return COUNTRY_PALETTE[(Math.max(0, asNumber(country.id, 1) - 1)) % COUNTRY_PALETTE.length];
}

function countryPower(country) {
  if (!country) return 0;
  const units = countryUnits(country);
  const weightedRaw =
    units.infantry +
    units.armor * 1.55 +
    units.artillery * 1.28 +
    units.airFighter * 1.38 +
    units.airBomber * 1.62 +
    units.navalSurface * 1.42 +
    units.navalSubmarine * 1.56;
  return weightedRaw / 1000;
}

function hexToRgb(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16)
  };
}

function showTextList(target, items) {
  target.innerHTML = "";
  const fragment = document.createDocumentFragment();
  const list = asArray(items);
  if (list.length === 0) {
    const li = document.createElement("li");
    li.textContent = "None";
    fragment.appendChild(li);
  } else {
    for (const item of list) {
      const li = document.createElement("li");
      li.textContent = escapeText(item);
      fragment.appendChild(li);
    }
  }
  target.appendChild(fragment);
}

async function api(path, method = "GET") {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    const response = await fetch(path, {
      method,
      signal: controller.signal,
      cache: "no-store"
    });
    const text = await response.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { raw: text };
      }
    }
    if (!response.ok) {
      throw new Error(payload.error || `API ${path} failed`);
    }
    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

function uploadModelToCountry(countryId, label, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload-model?country_id=${encodeURIComponent(countryId)}&label=${encodeURIComponent(label)}`);
    xhr.responseType = "text";

    xhr.upload.onprogress = (event) => {
      if (!onProgress || !event.lengthComputable) return;
      onProgress(clamp(event.loaded / event.total, 0, 1));
    };

    xhr.onload = () => {
      let payload = {};
      try {
        payload = JSON.parse(xhr.responseText || "{}");
      } catch {
        payload = { raw: xhr.responseText };
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
      } else {
        reject(new Error(payload.error || "upload failed"));
      }
    };

    xhr.onerror = () => reject(new Error("network upload failed"));
    xhr.send(file);
  });
}

function buildCountryMap() {
  return new Map(asArray(state.countries).map((country) => [asNumber(country.id), country]));
}

function groupedUploadsByCountry(uploadedModels) {
  const grouped = new Map();
  for (const names of Object.values(uploadedModels || {})) {
    for (const name of asArray(names)) {
      const match = /_c(\d+)$/i.exec(escapeText(name));
      const countryId = match ? Number(match[1]) : 0;
      if (!grouped.has(countryId)) {
        grouped.set(countryId, []);
      }
      if (!grouped.get(countryId).includes(name)) {
        grouped.get(countryId).push(name);
      }
    }
  }
  return grouped;
}

function validateModelFile(file) {
  if (!file) return "No file selected.";
  if (!/\.bin$/i.test(file.name)) return "Only .bin model files are accepted.";
  if (file.size <= 0) return "Model file is empty.";
  const maxUploadBytes = Math.max(1, asNumber(state.apiMeta.max_upload_bytes, 16 * 1024 * 1024));
  if (file.size > maxUploadBytes) {
    return `Model file exceeds ${(maxUploadBytes / (1024 * 1024)).toFixed(1)}MB limit.`;
  }
  return "";
}

function setCardProgress(countryId, progress, label) {
  const card = countryUploadGridEl.querySelector(`.country-card[data-country-id='${countryId}']`);
  if (!card) return;
  const bar = card.querySelector(".upload-progress");
  const note = card.querySelector(".upload-note");
  if (bar) {
    bar.value = Math.round(clamp(progress, 0, 1) * 100);
  }
  if (note && label) {
    note.textContent = label;
  }
}

function setCountryButtonsDisabled(disabled) {
  const buttons = countryUploadGridEl.querySelectorAll(".country-upload-btn");
  for (const button of buttons) {
    button.disabled = disabled;
  }
}

async function handleCountryUpload(countryId, countryName, file) {
  if (state.ui.uploadInFlight) return;

  const validationError = validateModelFile(file);
  if (validationError) {
    uploadStatusEl.textContent = `Upload blocked for C${countryId}: ${validationError}`;
    setCardProgress(countryId, 0, validationError);
    return;
  }

  const label = `${file.name.replace(/\.bin$/i, "")}_c${countryId}`;
  state.ui.uploadInFlight = true;
  setCountryButtonsDisabled(true);
  uploadStatusEl.textContent = `Uploading ${file.name} to ${countryName}...`;
  setCardProgress(countryId, 0, "Validating and uploading...");

  try {
    const result = await uploadModelToCountry(countryId, label, file, (fraction) => {
      setCardProgress(countryId, fraction, `Uploading ${(fraction * 100).toFixed(0)}%`);
    });
    const applied = asArray(result.applied_models).length > 0
      ? result.applied_models.join(", ")
      : (result.applied_model || "country slot");
    uploadStatusEl.textContent = `Uploaded to ${countryName}. Applied: ${applied}`;
    setCardProgress(countryId, 1, "Upload complete.");
    state.ui.modelPanelDirty = true;
    await refreshAll(true);
  } catch (error) {
    const message = `Upload failed: ${error.message}`;
    uploadStatusEl.textContent = message;
    setCardProgress(countryId, 0, message);
  } finally {
    state.ui.uploadInFlight = false;
    setCountryButtonsDisabled(false);
  }
}

function screenToWorld(x, y) {
  const cx = canvas.width * 0.5;
  const cy = canvas.height * 0.5;
  return {
    x: (x - cx) / camera.zoom + cx - camera.x,
    y: (y - cy) / camera.zoom + cy - camera.y
  };
}

function keepCameraInBounds() {
  const maxPan = 0.4 * Math.max(canvas.width, canvas.height);
  camera.x = clamp(camera.x, -maxPan, maxPan);
  camera.y = clamp(camera.y, -maxPan, maxPan);
}

function setZoomAtPoint(nextZoom, px, py) {
  const clamped = clamp(nextZoom, camera.minZoom, camera.maxZoom);
  const before = screenToWorld(px, py);
  camera.zoom = clamped;
  const after = screenToWorld(px, py);
  camera.x += after.x - before.x;
  camera.y += after.y - before.y;
  keepCameraInBounds();
}

function countryCentroids() {
  const map = state.map;
  const key = [map.width, map.height, canvas.width, canvas.height, asArray(map.cells).length].join("|");
  if (geometryCache.key === key) {
    return geometryCache.centroids;
  }

  const centroids = new Map();
  if (!map || map.width <= 0 || map.height <= 0) {
    geometryCache.key = key;
    geometryCache.centroids = centroids;
    return centroids;
  }

  const cells = asArray(map.cells);
  const cellW = canvas.width / map.width;
  const cellH = canvas.height / map.height;
  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const id = cells[y * map.width + x];
      if (!centroids.has(id)) centroids.set(id, { sx: 0, sy: 0, n: 0 });
      const entry = centroids.get(id);
      entry.sx += (x + 0.5) * cellW;
      entry.sy += (y + 0.5) * cellH;
      entry.n += 1;
    }
  }

  for (const entry of centroids.values()) {
    entry.x = entry.sx / entry.n;
    entry.y = entry.sy / entry.n;
  }

  geometryCache.key = key;
  geometryCache.centroids = centroids;
  return centroids;
}

function actionPulseColor(strategy) {
  switch (strategy) {
    case "attack":
      return "#88d6ff";
    case "defend":
      return "#76e4b8";
    case "negotiate":
      return "#f2d28b";
    case "transfer_weapons":
      return "#d9b0ff";
    case "form_alliance":
      return "#8fd7c8";
    case "betray":
      return "#ffb6b6";
    case "cyber_operation":
      return "#9ac6ff";
    case "sign_trade_agreement":
      return "#ffd36f";
    case "cancel_trade_agreement":
      return "#f1a76c";
    case "impose_embargo":
      return "#ff8d8d";
    case "invest_in_resource_extraction":
      return "#b8d97f";
    case "reduce_military_upkeep":
      return "#c4e6ff";
    case "suppress_dissent":
      return "#ffb07a";
    case "hold_elections":
      return "#8fe0d1";
    case "coup_attempt":
      return "#ff6f6f";
    case "focus_economy":
      return "#f5c689";
    case "develop_technology":
      return "#a8d4ff";
    case "surrender":
      return "#9da7b5";
    default:
      return "#b6d9f0";
  }
}

function spawnActionPulse(countryId, strategy, magnitude = 1) {
  const safeCountryId = asNumber(countryId, 0);
  if (!safeCountryId) return;
  if (state.actionPulses.length >= 360) {
    state.actionPulses.splice(0, state.actionPulses.length - 359);
  }
  state.actionPulses.push({
    countryId: safeCountryId,
    color: actionPulseColor(strategy),
    radiusBoost: clamp(magnitude, 0.65, 1.5),
    age: 0,
    ttl: 0.5 + Math.random() * 0.35
  });
}

function addProjectile(actorId, targetId, intensity) {
  if (state.projectiles.length >= visualConfig.maxProjectiles) state.projectiles.shift();
  state.projectiles.push({
    actorId,
    targetId,
    intensity: clamp(intensity, 0.25, 2.4),
    t: 0,
    speed: 0.65 + Math.random() * 0.75,
    arc: 24 + Math.random() * 80,
    wobble: Math.random() * Math.PI * 2,
    nearSelf: asNumber(actorId) === asNumber(targetId)
  });
}

function addExplosion(x, y, intensity) {
  if (state.explosions.length >= visualConfig.maxExplosions) state.explosions.shift();
  state.explosions.push({ x, y, intensity: clamp(intensity, 0.3, 2.7), ttl: 1.0, age: 0 });
  state.cameraShake = clamp(state.cameraShake + intensity * 4.6, 0, 18);

  const smokeCount = Math.round(4 + intensity * 6 * visualConfig.pressureLevel);
  for (let i = 0; i < smokeCount; i += 1) {
    if (state.smoke.length >= visualConfig.maxSmoke) state.smoke.shift();
    state.smoke.push({
      x: x + (Math.random() - 0.5) * 8,
      y: y + (Math.random() - 0.5) * 8,
      vx: (Math.random() - 0.5) * 16,
      vy: -16 - Math.random() * 24,
      r: 3 + Math.random() * 10,
      ttl: 1.1 + Math.random() * 1.2,
      age: 0
    });
  }

  const emberCount = Math.round(6 + intensity * 7 * visualConfig.pressureLevel);
  for (let i = 0; i < emberCount; i += 1) {
    if (state.embers.length >= visualConfig.maxEmbers) state.embers.shift();
    const angle = Math.random() * Math.PI * 2;
    const speed = 25 + Math.random() * 70;
    state.embers.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 12,
      ttl: 0.45 + Math.random() * 0.5,
      age: 0,
      r: 1 + Math.random() * 2.2
    });
  }
}

function animateDecisionsForTick() {
  if (state.tick === state.ui.lastAnimatedTick) return;
  state.ui.lastAnimatedTick = state.tick;

  const countriesById = buildCountryMap();
  for (const decision of asArray(state.decisions)) {
    const actorId = asNumber(decision.actor_country_id, 0);
    const targetId = asNumber(decision.target_country_id, 0);
    const strategy = escapeText(decision.strategy);

    if (actorId > 0) {
      spawnActionPulse(actorId, strategy, strategy === "attack" ? 1.15 : 1);
    }
    if (targetId > 0 && targetId !== actorId) {
      spawnActionPulse(targetId, strategy, 0.9);
    }

    if (strategy !== "attack") continue;
    const actor = countriesById.get(actorId);
    const target = countriesById.get(targetId);
    if (!actor || !target) continue;

    const actorPower = Math.max(1, countryPower(actor));
    const targetPower = Math.max(1, countryPower(target));
    const intensity = clamp(actorPower / targetPower, 0.35, 2.3);
    const burstCount = actorId === targetId ? 2 : 1;
    for (let i = 0; i < burstCount; i += 1) {
      addProjectile(actorId, targetId, intensity);
    }
  }
}

function renderStats() {
  const totalPop = asArray(state.countries).reduce((sum, country) => sum + asNumber(country.population), 0);
  const avgSupply = state.countries.length === 0
    ? 0
    : asArray(state.countries).reduce((sum, country) => sum + milliToPercent(country.supply_level), 0) / state.countries.length;
  const avgMorale = state.countries.length === 0
    ? 0
    : asArray(state.countries).reduce((sum, country) => sum + milliToPercent(country.civilian_morale), 0) / state.countries.length;
  const top = state.leaderboard[0] || null;
  statsEl.innerHTML = [
    `<div><b>Tick:</b> ${state.tick}</div>`,
    `<div><b>Countries:</b> ${state.countries.length}</div>`,
    `<div><b>Total Population:</b> ${totalPop.toLocaleString()}</div>`,
    `<div><b>Top Model:</b> ${top ? escapeText(top.model || top.team) : "n/a"}</div>`,
    `<div><b>Winner Country:</b> ${escapeText(state.winnerCountryName || state.competition.winner_country_name || "n/a")}${state.winnerCountryId ? ` (C${state.winnerCountryId})` : ""}</div>`,
    `<div><b>Battle:</b> ${state.battle.active ? "Running" : "Stopped"}</div>`,
    `<div><b>Elapsed:</b> ${asNumber(state.battle.elapsed_sec)}s | <b>Remaining:</b> ${asNumber(state.battle.remaining_sec)}s</div>`,
    `<div><b>Avg Supply:</b> ${avgSupply.toFixed(1)} | <b>Avg Morale:</b> ${avgMorale.toFixed(1)}</div>`
  ].join("");
  pauseBtn.disabled = !state.battle.active;
}

function renderStrategicSummary() {
  const countries = asArray(state.countries);
  const tradeLinks = countries.reduce((sum, country) => sum + asArray(country.trade_partners).length, 0) / 2;
  const defensePacts = countries.reduce((sum, country) => sum + asArray(country.defense_pacts).length, 0) / 2;
  const nonAggression = countries.reduce((sum, country) => sum + asArray(country.non_aggression_pacts).length, 0) / 2;
  const highestEscalation = countries.reduce((maxValue, country) => Math.max(maxValue, milliToPercent(country.escalation_level)), 0);
  const secondStrikeCount = countries.filter((country) => Boolean(country.second_strike_capable)).length;
  const topPower = countries.slice().sort((left, right) => countryPower(right) - countryPower(left))[0];

  strategicSummaryEl.innerHTML = [
    `<div><b>Trade Links:</b> ${formatCompactNumber(tradeLinks)} | <b>Defense Pacts:</b> ${formatCompactNumber(defensePacts)}</div>`,
    `<div><b>Non-Aggression Pacts:</b> ${formatCompactNumber(nonAggression)} | <b>Second Strike States:</b> ${secondStrikeCount}</div>`,
    `<div><b>Peak Escalation:</b> ${highestEscalation.toFixed(1)} | <b>Strongest Country:</b> ${topPower ? `${escapeText(topPower.name)} (${formatPower(topPower)})` : "n/a"}</div>`
  ].join("");
}

function renderDistributed() {
  distributedEl.innerHTML = [
    `<div><b>Node:</b> ${asNumber(state.distributed.node_id)} / ${Math.max(1, asNumber(state.distributed.total_nodes, 1))}</div>`,
    `<div><b>Bind:</b> ${escapeText(state.distributed.bind_host)}:${asNumber(state.distributed.bind_port)}</div>`,
    `<div><b>Mode:</b> ${asNumber(state.distributed.total_nodes, 1) > 1 ? "Clustered" : "Single-node cluster"}</div>`
  ].join("");
}

function renderDiagnostics() {
  const d = state.diagnostics.distributed || {};
  diagnosticsEl.innerHTML = [
    `<div><b>Battle Active:</b> ${state.diagnostics.battle_active ? "yes" : "no"} | <b>Tick:</b> ${asNumber(state.diagnostics.tick, state.tick)}</div>`,
    `<div><b>Peers:</b> ${asNumber(d.peer_count)} | <b>Exchanges:</b> ${asNumber(d.exchange_count)}</div>`,
    `<div><b>Packets:</b> sent ${asNumber(d.packets_sent)}, recv ${asNumber(d.packets_received)}, drop ${asNumber(d.packets_dropped)}</div>`,
    `<div><b>Sync:</b> ${state.ui.lastRefreshError ? "degraded" : "stable"} | <b>Last Refresh:</b> ${state.ui.lastRefreshAt ? new Date(state.ui.lastRefreshAt).toLocaleTimeString() : "n/a"}</div>`
  ].join("");
}

function renderMapLegend() {
  const tags = asArray(state.map.tags);
  const seaZones = asArray(state.map.sea_zones);
  let seaCells = 0;
  let strategicCells = 0;
  let chokepointCells = 0;
  let passOrCrossingCells = 0;
  let portCells = 0;

  for (const tag of tags) {
    const value = asNumber(tag, 0);
    if ((value & 1) !== 0) seaCells += 1;
    if ((value & 2) !== 0) strategicCells += 1;
    if ((value & (4 | 8)) !== 0) chokepointCells += 1;
    if ((value & (16 | 32)) !== 0) passOrCrossingCells += 1;
    if ((value & 64) !== 0) portCells += 1;
  }

  const zoneCount = new Set(seaZones.filter((zone) => asNumber(zone, 0) > 0)).size;
  const uploadLimitMb = (Math.max(1, asNumber(state.apiMeta.max_upload_bytes, 16 * 1024 * 1024)) / (1024 * 1024)).toFixed(1);
  mapLegendEl.innerHTML = [
    `<div><b>Sea Cells:</b> ${seaCells} | <b>Sea Zones:</b> ${zoneCount}</div>`,
    `<div><b>Strategic:</b> ${strategicCells} | <b>Chokepoints:</b> ${chokepointCells}</div>`,
    `<div><b>Pass/Crossing:</b> ${passOrCrossingCells} | <b>Ports:</b> ${portCells}</div>`,
    `<div><b>API:</b> v${asNumber(state.apiMeta.api_version, 1)} | <b>Upload Limit:</b> ${uploadLimitMb}MB</div>`
  ].join("");
}

function renderCountrySummary() {
  countrySummaryEl.innerHTML = "";
  const fragment = document.createDocumentFragment();
  const ranked = asArray(state.countries)
    .slice()
    .sort((left, right) => {
      const territoryDelta = asNumber(right.territory_cells) - asNumber(left.territory_cells);
      if (territoryDelta !== 0) return territoryDelta;
      return countryPower(right) - countryPower(left);
    })
    .slice(0, 6);

  for (const country of ranked) {
    const li = document.createElement("li");
    li.textContent = `${escapeText(country.name)} | power ${formatPower(country)} | supply ${milliToPercent(country.supply_level).toFixed(1)} | trade ${asArray(country.trade_partners).length} | treaties ${relationshipCount(country)} | escalation ${milliToPercent(country.escalation_level).toFixed(1)}`;
    fragment.appendChild(li);
  }

  if (fragment.childNodes.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No country state available";
    fragment.appendChild(li);
  }

  countrySummaryEl.appendChild(fragment);
}

function renderLeaderboard() {
  leaderboardEl.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const row of asArray(state.leaderboard)) {
    const li = document.createElement("li");
    li.textContent = `#${asNumber(row.rank, 0)} ${escapeText(row.model || row.team)} (${formatCompactNumber(row.score)})`;
    fragment.appendChild(li);
  }
  if (fragment.childNodes.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No scores yet";
    fragment.appendChild(li);
  }
  leaderboardEl.appendChild(fragment);
}

function formatDecision(decision) {
  const actor = asNumber(decision.actor_country_id, 0);
  const target = asNumber(decision.target_country_id, 0);
  const strategy = escapeText(decision.strategy || "unknown");
  const model = escapeText(decision.model || decision.team || "model");
  const termsType = escapeText(decision.terms?.type);
  const termsDetails = escapeText(decision.terms?.details);
  const forceCommitment = asNumber(decision.force_commitment, 0);
  const targetLabel = target > 0 ? ` C${target}` : "";
  const terms = termsType ? ` (${termsType}${termsDetails ? `: ${termsDetails}` : ""})` : "";
  const secondary = decision.has_secondary_action
    ? ` | secondary ${escapeText(decision.secondary_action?.strategy || "defend")}${asNumber(decision.secondary_action?.target_country_id, 0) > 0 ? ` C${asNumber(decision.secondary_action.target_country_id, 0)}` : ""}`
    : "";
  return `${model} C${actor} -> ${strategy}${targetLabel}${terms} | commitment ${forceCommitment.toFixed(2)}${secondary}`;
}

function renderDecisions() {
  decisionsEl.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const decision of asArray(state.decisions)) {
    const li = document.createElement("li");
    li.textContent = formatDecision(decision);
    fragment.appendChild(li);
  }
  if (fragment.childNodes.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No decisions this tick";
    fragment.appendChild(li);
  }
  decisionsEl.appendChild(fragment);
}

function renderMessages() {
  messagesEl.innerHTML = "";
  const fragment = document.createDocumentFragment();
  const recent = asArray(state.messages).slice(-16);
  for (const msg of recent) {
    const li = document.createElement("li");
    li.textContent = `${escapeText(msg.from)} -> ${escapeText(msg.to)} [${escapeText(msg.channel)}] ${escapeText(msg.content)}`;
    fragment.appendChild(li);
  }
  if (fragment.childNodes.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No messages yet";
    fragment.appendChild(li);
  }
  messagesEl.appendChild(fragment);
}

function renderModelReadiness() {
  const readiness = state.readiness || { ready: false, missing_country_ids: [] };
  const missing = asArray(readiness.missing_country_ids);
  if (readiness.ready) {
    modelReadinessEl.innerHTML = "<div><b>Status:</b> Ready</div><div>All countries have loaded neural model binaries.</div>";
  } else {
    const labels = missing.map((id) => {
      const country = state.countries.find((entry) => asNumber(entry.id) === asNumber(id));
      return country ? `C${id} ${country.name}` : `C${id}`;
    });
    modelReadinessEl.innerHTML = `<div><b>Status:</b> Blocked</div><div><b>Missing:</b> ${labels.join(", ") || "n/a"}</div>`;
  }
  stepBtn.disabled = !readiness.ready;
  startBtn.disabled = !readiness.ready;
}

function renderCountryUploadGrid() {
  const signature = JSON.stringify({
    countries: asArray(state.countries).map((country) => ({ id: country.id, name: country.name })),
    slots: asArray(state.countrySlots),
    uploaded: state.uploadedModels
  });

  if (!state.ui.modelPanelDirty && signature === state.ui.countryUploadSignature) {
    return;
  }

  state.ui.countryUploadSignature = signature;
  state.ui.modelPanelDirty = false;

  const slotByCountry = new Map(asArray(state.countrySlots).map((slot) => [asNumber(slot.country_id), slot]));
  countryUploadGridEl.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const country of asArray(state.countries)) {
    const slot = slotByCountry.get(asNumber(country.id));
    const card = document.createElement("div");
    card.className = "country-card";
    card.dataset.countryId = String(country.id);

    const head = document.createElement("div");
    head.className = "country-card-head";

    const title = document.createElement("div");
    title.className = "country-title";
    title.textContent = `C${country.id} ${country.name}`;

    const button = document.createElement("button");
    button.className = "country-upload-btn";
    button.dataset.countryId = String(country.id);
    button.dataset.countryName = country.name;
    button.textContent = "Select File";
    button.disabled = state.ui.uploadInFlight;

    const progress = document.createElement("progress");
    progress.className = "upload-progress";
    progress.max = 100;
    progress.value = 0;

    const note = document.createElement("div");
    note.className = "upload-note";
    note.textContent = slot?.selected_model
      ? `Loaded: ${slot.selected_model}${slot.loaded ? "" : " (not ready)"}`
      : "Drop .bin file here or click Select File.";

    head.appendChild(title);
    head.appendChild(button);
    card.appendChild(head);
    card.appendChild(progress);
    card.appendChild(note);

    card.addEventListener("dragover", (event) => {
      event.preventDefault();
      card.classList.add("drag-hover");
    });
    card.addEventListener("dragleave", () => card.classList.remove("drag-hover"));
    card.addEventListener("drop", async (event) => {
      event.preventDefault();
      card.classList.remove("drag-hover");
      const file = event.dataTransfer?.files?.[0];
      await handleCountryUpload(country.id, country.name, file);
    });

    fragment.appendChild(card);
  }
  countryUploadGridEl.appendChild(fragment);
}

function renderModelHierarchy() {
  modelHierarchyEl.innerHTML = "";
  const countryMap = buildCountryMap();
  const uploadsByCountry = groupedUploadsByCountry(state.uploadedModels);
  const slotByCountry = new Map(asArray(state.countrySlots).map((slot) => [asNumber(slot.country_id), slot]));

  const section = document.createElement("section");
  section.className = "hierarchy-team";
  const title = document.createElement("h3");
  title.textContent = "Independent Countries";
  section.appendChild(title);

  const ids = new Set(asArray(state.countries).map((country) => asNumber(country.id)));
  for (const id of uploadsByCountry.keys()) {
    if (id > 0) ids.add(id);
  }

  const sortedIds = Array.from(ids).sort((a, b) => a - b);
  for (const countryId of sortedIds) {
    const country = countryMap.get(countryId);
    const block = document.createElement("div");
    block.className = "hierarchy-country";

    const head = document.createElement("div");
    head.className = "hierarchy-country-title";
    head.textContent = country ? `C${country.id} ${country.name}` : `C${countryId}`;

    const list = document.createElement("ul");
    list.className = "hierarchy-models";
    const uploads = (uploadsByCountry.get(countryId) || []).slice().reverse();
    if (uploads.length > 0) {
      for (const item of uploads) {
        const li = document.createElement("li");
        li.textContent = escapeText(item);
        list.appendChild(li);
      }
    } else {
      const slot = slotByCountry.get(countryId);
      const li = document.createElement("li");
      li.textContent = slot?.slot_model ? `${slot.slot_model} (${slot.loaded ? "loaded" : "empty"})` : "No models";
      list.appendChild(li);
    }

    block.appendChild(head);
    block.appendChild(list);
    section.appendChild(block);
  }

  modelHierarchyEl.appendChild(section);
}

function renderModelPanels() {
  renderModelReadiness();
  renderCountryUploadGrid();
  renderModelHierarchy();
  showTextList(finalistsListEl, state.competition.finalists);
  showTextList(eliminatedListEl, state.competition.eliminated);
  showTextList(modelErrorsEl, state.modelLoadErrors);
}

function renderResultsModal() {
  const top = state.leaderboard[0] || {};
  const winner = escapeText(top.model || top.team || state.competition.winner_model || "n/a");
  const winnerScore = formatCompactNumber(top.score || 0);
  const winningCountry = escapeText(state.winnerCountryName || state.competition.winner_country_name || "n/a");
  const winnerCountryId = asNumber(state.winnerCountryId || state.competition.winner_country_id, 0);

  resultsSummaryEl.innerHTML = [
    `<div><b>Winner:</b> ${winner}</div>`,
    `<div><b>Winner Country:</b> ${winningCountry}${winnerCountryId ? ` (C${winnerCountryId})` : ""}</div>`,
    `<div><b>Final Score:</b> ${winnerScore}</div>`,
    `<div><b>Ticks:</b> ${state.tick}</div>`,
    `<div><b>Duration:</b> ${asNumber(state.battle.elapsed_sec)}s</div>`
  ].join("");

  resultsLeaderboardEl.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const row of asArray(state.leaderboard)) {
    const li = document.createElement("li");
    li.textContent = `#${asNumber(row.rank)} ${escapeText(row.model || row.team)} (${formatCompactNumber(row.score)})`;
    fragment.appendChild(li);
  }
  resultsLeaderboardEl.appendChild(fragment);
  resultsMessageEl.textContent = `Battle ended: country ${winningCountry} won; winning model: ${winner}.`;
  resultsModalEl.classList.remove("hidden");
}

function renderOverview() {
  renderStats();
  renderStrategicSummary();
  renderDistributed();
  renderDiagnostics();
  renderMapLegend();
  renderLeaderboard();
  renderCountrySummary();
  renderMessages();
  renderDecisions();
}

function renderAll() {
  try {
    renderOverview();
  } catch (error) {
    commandStatusEl.textContent = `Overview render failed: ${error.message}`;
  }

  try {
    renderModelPanels();
  } catch (error) {
    uploadStatusEl.textContent = `Models render failed: ${error.message}`;
  }
}

function drawGridMap() {
  const map = state.map;
  if (!map || map.width <= 0 || map.height <= 0) return;

  const cells = asArray(map.cells);
  const tags = asArray(map.tags);
  const seaZones = asArray(map.sea_zones);
  const cellW = canvas.width / map.width;
  const cellH = canvas.height / map.height;
  const countriesById = buildCountryMap();

  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const index = y * map.width + x;
      const id = cells[index];
      const country = countriesById.get(id);
      const base = countryColor(country);
      const rgb = hexToRgb(base);
      const tagValue = asNumber(tags[index], 0);
      const isSea = (tagValue & MAP_TAGS.SEA) !== 0;
      const seaZone = asNumber(seaZones[index], 0);
      const n = Math.sin((x * 0.27 + y * 0.41 + id * 0.77 + state.tick * 0.02) * 2.4) * 0.08;

      if (isSea) {
        const tint = seaZone <= 1 ? "rgba(38, 116, 157, 0.88)" : (seaZone === 2 ? "rgba(28, 98, 147, 0.9)" : "rgba(20, 78, 130, 0.9)");
        ctx.fillStyle = tint;
      } else {
        const light = 0.85 + n;
        ctx.fillStyle = `rgb(${Math.round(rgb.r * light)},${Math.round(rgb.g * light)},${Math.round(rgb.b * light)})`;
      }
      ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);

      if ((tagValue & MAP_TAGS.STRATEGIC) !== 0) {
        ctx.fillStyle = "rgba(246, 209, 120, 0.15)";
        ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  ctx.strokeStyle = "rgba(14,22,35,0.2)";
  ctx.lineWidth = 0.9;
  for (let x = 0; x <= map.width; x += 1) {
    const px = x * cellW;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= map.height; y += 1) {
    const py = y * cellH;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(canvas.width, py);
    ctx.stroke();
  }

  // Draw map tag markers after base terrain so players can see sea lanes and chokepoints.
  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const index = y * map.width + x;
      const tagValue = asNumber(tags[index], 0);
      if (tagValue === 0) continue;

      const cx = (x + 0.5) * cellW;
      const cy = (y + 0.5) * cellH;

      if ((tagValue & (MAP_TAGS.CHOKE_STRAIT | MAP_TAGS.CHOKE_CANAL)) !== 0) {
        ctx.fillStyle = "rgba(255, 224, 160, 0.92)";
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(1.2, Math.min(cellW, cellH) * 0.12), 0, Math.PI * 2);
        ctx.fill();
      } else if ((tagValue & MAP_TAGS.PORT) !== 0) {
        ctx.fillStyle = "rgba(118, 229, 199, 0.85)";
        ctx.fillRect(cx - cellW * 0.12, cy - cellH * 0.12, cellW * 0.24, cellH * 0.24);
      } else if ((tagValue & (MAP_TAGS.MOUNTAIN_PASS | MAP_TAGS.RIVER_CROSSING)) !== 0) {
        ctx.fillStyle = "rgba(219, 239, 255, 0.8)";
        ctx.fillRect(cx - cellW * 0.08, cy - cellH * 0.08, cellW * 0.16, cellH * 0.16);
      }
    }
  }

  const centers = countryCentroids();
  const maxPower = Math.max(1, ...asArray(state.countries).map((country) => countryPower(country)));
  for (const country of asArray(state.countries)) {
    const center = centers.get(asNumber(country.id));
    if (!center) continue;
    const ratio = clamp(countryPower(country) / maxPower, 0.06, 1);
    const radius = lerp(8, 38, Math.sqrt(ratio));
    const alpha = lerp(0.08, 0.22, ratio);
    const glow = ctx.createRadialGradient(center.x, center.y, 1, center.x, center.y, radius);
    glow.addColorStop(0, `rgba(170, 224, 255, ${alpha})`);
    glow.addColorStop(1, "rgba(170, 224, 255, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawActionPulses(dt) {
  if (!state.actionPulses.length) return;
  const centers = countryCentroids();
  const next = [];
  for (const pulse of state.actionPulses) {
    const center = centers.get(pulse.countryId);
    if (!center) continue;

    pulse.age += dt;
    const life = clamp(1 - pulse.age / Math.max(0.001, pulse.ttl), 0, 1);
    const growth = 1 - life;
    const radius = (8 + growth * 24) * pulse.radiusBoost;
    const alpha = 0.12 + life * 0.2;
    const rgb = hexToRgb(pulse.color);

    ctx.strokeStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha.toFixed(3)})`;
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = `rgba(220, 240, 255, ${(life * 0.16).toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 1.4 + life * 2.2, 0, Math.PI * 2);
    ctx.fill();

    if (pulse.age < pulse.ttl) next.push(pulse);
  }
  state.actionPulses = next;
}

function drawProjectilesAndExplosions(dt) {
  const centers = countryCentroids();
  const activeProjectiles = [];

  for (const projectile of state.projectiles) {
    const from = centers.get(projectile.actorId);
    let to = centers.get(projectile.targetId);
    if (!from || !to) continue;

    if (projectile.nearSelf || Math.hypot(from.x - to.x, from.y - to.y) < 8) {
      const angle = projectile.wobble + projectile.t * Math.PI * 2;
      to = {
        x: from.x + Math.cos(angle) * 24,
        y: from.y + Math.sin(angle) * 16
      };
    }

    projectile.t += dt * projectile.speed;
    const t = projectile.t;
    if (t >= 1) {
      addExplosion(to.x, to.y, projectile.intensity);
      continue;
    }

    const midX = (from.x + to.x) * 0.5 + Math.sin(projectile.wobble + t * 6.8) * 6;
    const midY = (from.y + to.y) * 0.5 - projectile.arc;
    const x = (1 - t) * (1 - t) * from.x + 2 * (1 - t) * t * midX + t * t * to.x;
    const y = (1 - t) * (1 - t) * from.y + 2 * (1 - t) * t * midY + t * t * to.y;

    ctx.strokeStyle = `rgba(108, 196, 255, ${0.28 + (1 - t) * 0.38})`;
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.quadraticCurveTo(midX, midY, x, y);
    ctx.stroke();

    ctx.fillStyle = `rgba(198, 236, 255, ${0.5 + (1 - t) * 0.38})`;
    ctx.beginPath();
    ctx.arc(x, y, 1.5 + projectile.intensity * 1.7, 0, Math.PI * 2);
    ctx.fill();

    activeProjectiles.push(projectile);
  }
  state.projectiles = activeProjectiles;

  const activeExplosions = [];
  for (const blast of state.explosions) {
    blast.age += dt;
    blast.ttl -= dt;
    const life = clamp(blast.ttl / 1.0, 0, 1);
    const radius = (1 - life) * (16 + blast.intensity * 25);
    const glow = ctx.createRadialGradient(blast.x, blast.y, 1, blast.x, blast.y, Math.max(2, radius));
    glow.addColorStop(0, `rgba(228, 248, 255, ${0.8 * life})`);
    glow.addColorStop(0.35, `rgba(117, 207, 255, ${0.72 * life})`);
    glow.addColorStop(1, "rgba(20, 64, 92, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(blast.x, blast.y, Math.max(2, radius), 0, Math.PI * 2);
    ctx.fill();
    if (blast.ttl > 0) activeExplosions.push(blast);
  }
  state.explosions = activeExplosions;

  const smoke = [];
  for (const puff of state.smoke) {
    puff.age += dt;
    puff.ttl -= dt;
    puff.x += puff.vx * dt;
    puff.y += puff.vy * dt;
    puff.vx *= 0.97;
    puff.vy *= 0.99;
    puff.r += dt * 7;

    const alpha = clamp(puff.ttl / 2.0, 0, 1) * 0.4;
    ctx.fillStyle = `rgba(34, 48, 66, ${alpha})`;
    ctx.beginPath();
    ctx.arc(puff.x, puff.y, puff.r, 0, Math.PI * 2);
    ctx.fill();
    if (puff.ttl > 0) smoke.push(puff);
  }
  state.smoke = smoke;

  const embers = [];
  for (const ember of state.embers) {
    ember.age += dt;
    ember.ttl -= dt;
    ember.vy += 45 * dt;
    ember.x += ember.vx * dt;
    ember.y += ember.vy * dt;
    const alpha = clamp(ember.ttl / 1.0, 0, 1);
    ctx.fillStyle = `rgba(123, 214, 255, ${alpha})`;
    ctx.beginPath();
    ctx.arc(ember.x, ember.y, ember.r, 0, Math.PI * 2);
    ctx.fill();
    if (ember.ttl > 0) embers.push(ember);
  }
  state.embers = embers;
}

function drawAtmosphere() {
  const phase = (state.tick % 260) / 260;
  const drift = 0.1 + Math.sin(phase * Math.PI * 2) * 0.06;
  const sky = ctx.createLinearGradient(0, 0, 0, canvas.height);
  sky.addColorStop(0, `rgba(70, 114, 145, ${0.09 + drift})`);
  sky.addColorStop(1, "rgba(5, 11, 18, 0.32)");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawScene(dt) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const shake = state.cameraShake > 0.01 ? state.cameraShake : 0;
  ctx.save();

  if (shake > 0) {
    ctx.translate((Math.random() * 2 - 1) * shake, (Math.random() * 2 - 1) * shake);
    state.cameraShake = Math.max(0, state.cameraShake - dt * 24);
  }

  const cx = canvas.width * 0.5;
  const cy = canvas.height * 0.5;
  ctx.translate(cx, cy);
  ctx.scale(camera.zoom, camera.zoom);
  ctx.translate(-cx + camera.x, -cy + camera.y);

  drawGridMap();
  drawActionPulses(dt);
  drawProjectilesAndExplosions(dt);
  drawAtmosphere();

  ctx.restore();
}

function updateParticleBudget() {
  if (fpsAverage < 35) {
    lowFpsStreak += 1;
    highFpsStreak = 0;
  } else if (fpsAverage > 52) {
    highFpsStreak += 1;
    lowFpsStreak = 0;
  } else {
    lowFpsStreak = Math.max(0, lowFpsStreak - 1);
    highFpsStreak = Math.max(0, highFpsStreak - 1);
  }

  if (lowFpsStreak > 24 && visualConfig.pressureLevel > 0.6) {
    visualConfig.pressureLevel = clamp(visualConfig.pressureLevel - 0.1, 0.5, 1.1);
    visualConfig.maxProjectiles = Math.round(96 * visualConfig.pressureLevel);
    visualConfig.maxExplosions = Math.round(120 * visualConfig.pressureLevel);
    visualConfig.maxSmoke = Math.round(220 * visualConfig.pressureLevel);
    visualConfig.maxEmbers = Math.round(260 * visualConfig.pressureLevel);
    lowFpsStreak = 0;
  }

  if (highFpsStreak > 80 && visualConfig.pressureLevel < 1) {
    visualConfig.pressureLevel = clamp(visualConfig.pressureLevel + 0.05, 0.5, 1);
    visualConfig.maxProjectiles = Math.round(96 * visualConfig.pressureLevel);
    visualConfig.maxExplosions = Math.round(120 * visualConfig.pressureLevel);
    visualConfig.maxSmoke = Math.round(220 * visualConfig.pressureLevel);
    visualConfig.maxEmbers = Math.round(260 * visualConfig.pressureLevel);
    highFpsStreak = 0;
  }
}

function frame(time) {
  resizeCanvas();
  const dt = Math.min(0.1, (time - lastFrame) / 1000);
  lastFrame = time;

  const fpsNow = 1 / Math.max(0.0001, dt);
  fpsAverage = fpsAverage * 0.9 + fpsNow * 0.1;
  perfBadgeEl.textContent = state.ui.lastRefreshError
    ? `FPS: ${Math.round(fpsAverage)} | Sync degraded`
    : `FPS: ${Math.round(fpsAverage)} | Live ${state.ui.refreshInFlight ? "syncing" : "steady"}`;

  updateParticleBudget();
  drawScene(dt);
  requestAnimationFrame(frame);
}

function applyStatePayload(data) {
  state.tick = asNumber(data.tick, state.tick);
  state.countries = asArray(data.countries);
  state.decisions = asArray(data.decisions);
  state.messages = asArray(data.messages);
  state.battle = {
    ...state.battle,
    ...(data.battle || {})
  };
  state.distributed = {
    ...state.distributed,
    ...(data.distributed || {})
  };
  state.map = data.map || { width: 0, height: 0, cells: [] };
  state.competition = {
    ...state.competition,
    ...(data.competition || {})
  };
  state.modelLoadErrors = asArray(data.model_load_errors);
  geometryCache.key = "";
  animateDecisionsForTick();
}

function applyLeaderboardPayload(data) {
  state.leaderboard = asArray(data.leaderboard);
  state.winnerCountryId = asNumber(data.winner_country_id, 0);
  state.winnerCountryName = escapeText(data.winner_country_name || "");
}

function applyDiagnosticsPayload(data) {
  state.diagnostics = {
    ...state.diagnostics,
    ...(data || {}),
    distributed: {
      ...state.diagnostics.distributed,
      ...((data && data.distributed) || {})
    }
  };
}

function applyModelsPayload(data) {
  state.uploadedModels = data.uploaded_models || {};
  state.readiness = data.readiness || { ready: false, missing_country_ids: [] };
  state.countrySlots = asArray(data.country_slots);
  state.modelLoadErrors = asArray(data.model_load_errors).length > 0 ? asArray(data.model_load_errors) : state.modelLoadErrors;
  state.ui.modelPanelDirty = true;
}

function populateStrategyOptions(strategies) {
  const items = asArray(strategies).filter((item) => typeof item === "string" && item.length > 0);
  if (!items.length || !overrideStrategyEl) {
    return;
  }

  const selected = overrideStrategyEl.value;
  overrideStrategyEl.innerHTML = "";
  for (const strategy of items) {
    const option = document.createElement("option");
    option.value = strategy;
    option.textContent = strategy;
    overrideStrategyEl.appendChild(option);
  }

  if (items.includes(selected)) {
    overrideStrategyEl.value = selected;
  } else if (items.includes("defend")) {
    overrideStrategyEl.value = "defend";
  } else {
    overrideStrategyEl.value = items[0];
  }
}

function applyMetaPayload(data) {
  const maxUploadBytes = Math.max(1, asNumber(data.max_upload_bytes, state.apiMeta.max_upload_bytes));
  const strategies = asArray(data.strategies);
  const targeted = asArray(data.targeted_strategies);

  state.apiMeta = {
    api_version: asNumber(data.api_version, 1),
    max_upload_bytes: maxUploadBytes,
    strategies,
    targeted_strategies: targeted
  };

  if (targeted.length > 0) {
    targetedStrategies = new Set(targeted);
  }
  if (strategies.length > 0) {
    populateStrategyOptions(strategies);
  }
}

async function refreshMeta() {
  try {
    const payload = await api("/api/meta");
    applyMetaPayload(payload);
  } catch (error) {
    state.ui.lastRefreshError = error.message;
  }
}

async function refreshModelsIfDue(force = false) {
  const now = Date.now();
  if (!force && now - state.ui.lastModelRefreshAt < MODEL_REFRESH_INTERVAL_MS) {
    return;
  }
  try {
    const payload = await api("/api/models");
    applyModelsPayload(payload);
    state.ui.lastModelRefreshAt = now;
  } catch (error) {
    state.ui.lastRefreshError = error.message;
  }
}

async function refreshAll(forceModels = false) {
  if (state.ui.refreshInFlight) return;
  state.ui.refreshInFlight = true;

  try {
    const [stateResult, leaderboardResult, diagnosticsResult] = await Promise.allSettled([
      api("/api/state"),
      api("/api/leaderboard"),
      api("/api/diagnostics")
    ]);

    let hadSuccess = false;
    let refreshError = "";

    if (stateResult.status === "fulfilled") {
      applyStatePayload(stateResult.value);
      hadSuccess = true;
    } else {
      refreshError = stateResult.reason?.message || "state refresh failed";
    }

    if (leaderboardResult.status === "fulfilled") {
      applyLeaderboardPayload(leaderboardResult.value);
      hadSuccess = true;
    } else if (!refreshError) {
      refreshError = leaderboardResult.reason?.message || "leaderboard refresh failed";
    }

    if (diagnosticsResult.status === "fulfilled") {
      applyDiagnosticsPayload(diagnosticsResult.value);
      hadSuccess = true;
    } else if (!refreshError) {
      refreshError = diagnosticsResult.reason?.message || "diagnostics refresh failed";
    }

    if (forceModels || state.apiMeta.strategies.length === 0) {
      await refreshMeta();
    }

    await refreshModelsIfDue(forceModels);

    if (hadSuccess) {
      state.ui.refreshFailures = 0;
      state.ui.lastRefreshAt = Date.now();
      state.ui.lastRefreshError = refreshError;
      renderAll();

      const battleJustEnded = state.ui.previousBattleActive && !state.battle.active;
      if (battleJustEnded && state.ui.resultsOpenForTick !== state.tick) {
        state.ui.resultsOpenForTick = state.tick;
        renderResultsModal();
      }
      state.ui.previousBattleActive = state.battle.active;
    } else {
      state.ui.refreshFailures += 1;
      state.ui.lastRefreshError = refreshError || "all refreshes failed";
    }
  } finally {
    state.ui.refreshInFlight = false;
    scheduleRefresh();
  }
}

function scheduleRefresh() {
  clearTimeout(refreshTimer);
  const interval = state.battle.active ? POLL_INTERVAL_ACTIVE_MS : POLL_INTERVAL_IDLE_MS;
  refreshTimer = setTimeout(() => {
    refreshAll(false).catch((error) => {
      state.ui.lastRefreshError = error.message;
      scheduleRefresh();
    });
  }, interval);
}

function strategyNeedsTarget(strategy) {
  return targetedStrategies.has(strategy);
}

async function applyManualOverride() {
  const actor = Number(document.getElementById("overrideActor").value || "0");
  const target = Number(document.getElementById("overrideTarget").value || "0");
  const strategy = document.getElementById("overrideStrategy").value;
  const termsType = document.getElementById("overrideTermsType").value.trim();
  const termsDetails = document.getElementById("overrideTermsDetails").value.trim();

  if (!Number.isFinite(actor) || actor <= 0) {
    commandStatusEl.textContent = "Manual override blocked: actor country id is required.";
    return;
  }

  if (strategyNeedsTarget(strategy) && (!Number.isFinite(target) || target <= 0)) {
    commandStatusEl.textContent = "Manual override blocked: selected strategy requires target country id.";
    return;
  }

  const query = new URLSearchParams({
    actor_country_id: String(actor),
    target_country_id: String(Math.max(0, target)),
    strategy,
    terms_type: termsType,
    terms_details: termsDetails
  });

  try {
    await api(`/api/control/override?${query.toString()}`, "POST");
    commandStatusEl.textContent = `Manual override applied: C${actor} -> ${strategy}`;
    await refreshAll(true);
  } catch (error) {
    commandStatusEl.textContent = `Manual override failed: ${error.message}`;
  }
}

function findCountryAtCanvas(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const dprX = canvas.width / Math.max(1, rect.width);
  const dprY = canvas.height / Math.max(1, rect.height);
  const sx = (clientX - rect.left) * dprX;
  const sy = (clientY - rect.top) * dprY;
  const world = screenToWorld(sx, sy);
  const map = state.map;
  if (!map || map.width <= 0 || map.height <= 0) return null;

  const cellX = Math.floor((world.x / canvas.width) * map.width);
  const cellY = Math.floor((world.y / canvas.height) * map.height);
  if (cellX < 0 || cellY < 0 || cellX >= map.width || cellY >= map.height) return null;

  const id = asArray(map.cells)[cellY * map.width + cellX];
  return state.countries.find((country) => asNumber(country.id) === asNumber(id)) || null;
}

function updateTooltip(event) {
  const country = findCountryAtCanvas(event.clientX, event.clientY);
  if (!country) {
    tooltipEl.classList.add("hidden");
    return;
  }

  tooltipEl.innerHTML = [
    `<div><b>${escapeText(country.name)}</b> (C${country.id})</div>`,
    `<div>Team: ${escapeText(country.team || "n/a")}</div>`,
    `<div>Ground: ${formatUnitCount(countryUnits(country).infantry)} inf / ${formatUnitCount(countryUnits(country).armor)} arm / ${formatUnitCount(countryUnits(country).artillery)} art</div>`,
    `<div>Air/Naval: ${formatUnitCount(countryUnits(country).airFighter)} fighters / ${formatUnitCount(countryUnits(country).airBomber)} bombers / ${formatUnitCount(countryUnits(country).navalSurface + countryUnits(country).navalSubmarine)} naval</div>`,
    `<div>Supply: ${milliToPercent(country.supply_level).toFixed(1)} / ${milliToPercent(country.supply_capacity).toFixed(1)} | Morale: ${milliToPercent(country.civilian_morale).toFixed(1)}</div>`,
    `<div>Economy: ${milliToPercent(country.economic_stability).toFixed(1)} | Reputation: ${milliToPercent(country.reputation).toFixed(1)} | Escalation: ${milliToPercent(country.escalation_level).toFixed(1)}</div>`,
    `<div>Trade: ${asArray(country.trade_partners).length} | Defense: ${asArray(country.defense_pacts).length} | NAPs: ${asArray(country.non_aggression_pacts).length}</div>`,
    `<div>Power: ${formatPower(country)} | Territory: ${formatCompactNumber(country.territory_cells)}${country.second_strike_capable ? " | second strike" : ""}</div>`
  ].join("");
  tooltipEl.style.left = `${event.clientX + 14}px`;
  tooltipEl.style.top = `${event.clientY + 14}px`;
  tooltipEl.classList.remove("hidden");
}

function handlePointerDown(event) {
  camera.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  canvas.setPointerCapture(event.pointerId);
  if (camera.pointers.size === 1) {
    camera.dragging = true;
    camera.dragStartX = event.clientX;
    camera.dragStartY = event.clientY;
    camera.startX = camera.x;
    camera.startY = camera.y;
  }
  if (camera.pointers.size === 2) {
    const values = Array.from(camera.pointers.values());
    const dx = values[0].x - values[1].x;
    const dy = values[0].y - values[1].y;
    camera.pinchStartDistance = Math.hypot(dx, dy);
    camera.pinchStartZoom = camera.zoom;
  }
}

function handlePointerMove(event) {
  if (!camera.pointers.has(event.pointerId)) {
    updateTooltip(event);
    return;
  }

  camera.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (camera.pointers.size === 1 && camera.dragging) {
    const rect = canvas.getBoundingClientRect();
    const dprX = canvas.width / Math.max(1, rect.width);
    const dprY = canvas.height / Math.max(1, rect.height);
    const dx = (event.clientX - camera.dragStartX) * dprX;
    const dy = (event.clientY - camera.dragStartY) * dprY;
    camera.x = camera.startX + dx / camera.zoom;
    camera.y = camera.startY + dy / camera.zoom;
    keepCameraInBounds();
  }

  if (camera.pointers.size === 2) {
    const values = Array.from(camera.pointers.values());
    const dx = values[0].x - values[1].x;
    const dy = values[0].y - values[1].y;
    const distance = Math.max(1, Math.hypot(dx, dy));
    const ratio = distance / Math.max(1, camera.pinchStartDistance);
    const rect = canvas.getBoundingClientRect();
    const midpointX = ((values[0].x + values[1].x) / 2 - rect.left) * (canvas.width / Math.max(1, rect.width));
    const midpointY = ((values[0].y + values[1].y) / 2 - rect.top) * (canvas.height / Math.max(1, rect.height));
    setZoomAtPoint(camera.pinchStartZoom * ratio, midpointX, midpointY);
  }

  updateTooltip(event);
}

function handlePointerUp(event) {
  camera.pointers.delete(event.pointerId);
  if (camera.pointers.size === 0) {
    camera.dragging = false;
  }
}

function setupMapInteractions() {
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
    const y = (event.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));
    const delta = event.deltaY < 0 ? 1.12 : 0.9;
    setZoomAtPoint(camera.zoom * delta, x, y);
  }, { passive: false });

  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", handlePointerUp);
  canvas.addEventListener("pointercancel", handlePointerUp);
  canvas.addEventListener("pointerleave", () => tooltipEl.classList.add("hidden"));
}

function setupSidebar() {
  const sidebar = document.getElementById("sidebar");
  const collapseBtn = document.getElementById("collapseSidebarBtn");
  const expandBtn = document.getElementById("expandSidebarBtn");

  collapseBtn.addEventListener("click", () => {
    sidebar.classList.add("collapsed");
    expandBtn.classList.remove("hidden");
  });

  expandBtn.addEventListener("click", () => {
    sidebar.classList.remove("collapsed");
    expandBtn.classList.add("hidden");
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(btn.dataset.tab);
      if (panel) panel.classList.add("active");
    });
  });
}

async function runControl(path, statusEl, successText, forceModels = false) {
  try {
    await api(path, "POST");
    if (statusEl && successText) {
      statusEl.textContent = successText;
    }
    await refreshAll(forceModels);
  } catch (error) {
    if (statusEl) {
      statusEl.textContent = `${successText || "Action"} failed: ${error.message}`;
    }
  }
}

function resetEffects() {
  state.projectiles = [];
  state.explosions = [];
  state.smoke = [];
  state.embers = [];
  state.actionPulses = [];
  state.cameraShake = 0;
  state.ui.lastAnimatedTick = -1;
}

function bindEvents() {
  document.getElementById("stepBtn").onclick = () => runControl("/api/control/step", uploadStatusEl, "Step complete", true);
  document.getElementById("startBtn").onclick = () => runControl("/api/control/start", uploadStatusEl, "Battle started", true);
  document.getElementById("pauseBtn").onclick = () => runControl("/api/control/pause", uploadStatusEl, "Battle paused", false);
  document.getElementById("endBtn").onclick = () => runControl("/api/control/end", uploadStatusEl, "Battle ended", true);

  document.getElementById("resetBtn").onclick = async () => {
    resetEffects();
    await runControl("/api/control/reset", uploadStatusEl, "Battle reset", true);
  };

  speedInput.oninput = async () => {
    speedValue.textContent = speedInput.value;
    try {
      await api(`/api/control/speed?ticks_per_second=${speedInput.value}`, "POST");
    } catch (error) {
      uploadStatusEl.textContent = `Speed update failed: ${error.message}`;
    }
  };

  durationInput.onchange = async () => {
    const minSec = Math.max(60, asNumber(state.battle.min_duration_sec, 60));
    const value = Math.max(minSec, asNumber(durationInput.value, minSec));
    durationInput.value = String(value);
    try {
      await api(`/api/control/duration?seconds=${value}`, "POST");
      await refreshAll(false);
    } catch (error) {
      uploadStatusEl.textContent = `Duration update failed: ${error.message}`;
    }
  };

  document.getElementById("closeResultsBtn").onclick = () => {
    resultsModalEl.classList.add("hidden");
  };

  document.getElementById("overrideApplyBtn").onclick = applyManualOverride;
  document.getElementById("zoomInBtn").onclick = () => setZoomAtPoint(camera.zoom * 1.15, canvas.width / 2, canvas.height / 2);
  document.getElementById("zoomOutBtn").onclick = () => setZoomAtPoint(camera.zoom * 0.87, canvas.width / 2, canvas.height / 2);
  document.getElementById("resetViewBtn").onclick = () => {
    camera.zoom = 1;
    camera.x = 0;
    camera.y = 0;
  };

  countryUploadGridEl.addEventListener("click", (event) => {
    const button = event.target.closest(".country-upload-btn");
    if (!button || state.ui.uploadInFlight) return;
    state.ui.pendingCountryUpload = {
      countryId: asNumber(button.dataset.countryId, 0),
      countryName: button.dataset.countryName || `C${button.dataset.countryId}`
    };
    hiddenCountryFileInput.value = "";
    hiddenCountryFileInput.click();
  });

  hiddenCountryFileInput.addEventListener("change", async () => {
    if (!state.ui.pendingCountryUpload || state.ui.uploadInFlight) return;
    const file = hiddenCountryFileInput.files && hiddenCountryFileInput.files[0];
    if (!file) {
      state.ui.pendingCountryUpload = null;
      return;
    }
    const { countryId, countryName } = state.ui.pendingCountryUpload;
    state.ui.pendingCountryUpload = null;
    await handleCountryUpload(countryId, countryName, file);
  });
}

window.addEventListener("resize", resizeCanvas);

setupSidebar();
setupMapInteractions();
bindEvents();
resizeCanvas();
renderAll();
refreshMeta().then(() => refreshAll(true)).catch((error) => {
  state.ui.lastRefreshError = error.message;
  scheduleRefresh();
});
requestAnimationFrame(frame);
