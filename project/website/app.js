/* ============================================================
   DemandAI — app.js
   Loads JSON artifacts (or falls back to demo data) and
   renders all charts + interactive elements.
   ============================================================ */

'use strict';

// ── DEMO DATA (used when JSON files aren't ready yet) ─────────────────────────
const DEMO_META = {
  overall_mae:           9.62,
  overall_rmse:          15.38,
  overall_r2:            0.9611,
  total_samples:         1185,
  num_entities:          15,
  entities: [
    "Store_A - Clothing","Store_A - Electronics","Store_A - Furniture",
    "Store_A - Groceries","Store_A - Toys",
    "Store_B - Clothing","Store_B - Electronics","Store_B - Furniture",
    "Store_B - Groceries","Store_B - Toys",
    "Store_C - Clothing","Store_C - Electronics","Store_C - Furniture",
    "Store_C - Groceries","Store_C - Toys",
  ],
  per_entity_mae: {
    "Store_A - Clothing":8.94,"Store_A - Electronics":6.71,
    "Store_A - Furniture":4.82,"Store_A - Groceries":20.27,"Store_A - Toys":7.28,
    "Store_B - Clothing":11.17,"Store_B - Electronics":6.55,
    "Store_B - Furniture":2.62,"Store_B - Groceries":20.87,"Store_B - Toys":6.94,
    "Store_C - Clothing":11.28,"Store_C - Electronics":5.36,
    "Store_C - Furniture":3.12,"Store_C - Groceries":20.09,"Store_C - Toys":8.34,
  },
  holding_cost_per_unit:  3.0,
  stockout_cost_per_unit: 22.5,
  avg_demand:             65.0,
  demand_min:             25.0,
  demand_max:             120.0,
};

function makeDemoForecast(entity) {
  // Simulate ~79 days of test data with realistic demand signal
  const n = 79;
  const dates = [], actual = [], predicted = [];
  let base = 55 + Math.random() * 30;

  for (let i = 0; i < n; i++) {
    const d = new Date(2024, 0, 1 + i);
    dates.push(d.toISOString().slice(0, 10));
    const trend   = Math.sin(i / 14 * Math.PI) * 12;
    const noise   = (Math.random() - .5) * 10;
    const act     = Math.round(Math.max(20, base + trend + noise));
    const predNoise = (Math.random() - .5) * 14;
    const pred    = Math.round(Math.max(20, act + predNoise));
    actual.push(act);
    predicted.push(pred);
    base += (Math.random() - .5) * 2;
  }
  return { store: entity.split(' - ')[0], product: entity.split(' - ')[1], dates, actual, predicted };
}

const DEMO_HISTORY = {
  loss:    Array.from({length:50}, (_,i) => parseFloat((0.012 * Math.exp(-i * .07) + 0.001 + Math.random()*.0003).toFixed(6))),
  val_loss:Array.from({length:50}, (_,i) => parseFloat((0.009 * Math.exp(-i * .06) + 0.0009 + Math.random()*.0003).toFixed(6))),
};

// ── STATE ──────────────────────────────────────────────────────────────────────
let meta         = null;
let forecasts    = {};
let historyData  = null;
let forecastChart = null;
let historyChart  = null;

// ── INIT ───────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const [m, f, h] = await Promise.all([
      fetchJSON('model_metadata.json'),
      fetchJSON('forecasts.json'),
      fetchJSON('training_history.json'),
    ]);
    meta        = m;
    forecasts   = f;
    historyData = h;
    setStatus('Model loaded', true);
  } catch (_) {
    // Fallback to demo data
    meta = DEMO_META;
    historyData = DEMO_HISTORY;
    for (const e of DEMO_META.entities) forecasts[e] = makeDemoForecast(e);
    setStatus('Demo mode — copy JSON files to enable real data', false);
  }

  renderKPIs();
  buildEntitySelectors();
  buildForecastChart();
  buildHistoryChart();
  buildPerfList();
  renderCostPanel();
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  return r.json();
}

function setStatus(msg, ok) {
  const chip = document.getElementById('statusChip');
  const dot  = chip.querySelector('.status-dot');
  const txt  = document.getElementById('statusText');
  txt.textContent = msg;
  dot.style.background   = ok ? 'var(--emerald)' : 'var(--amber)';
  dot.style.boxShadow    = ok ? '0 0 6px var(--emerald)' : '0 0 6px var(--amber)';
}

// ── KPI CARDS ─────────────────────────────────────────────────────────────────
function renderKPIs() {
  animCount('valMAE',     meta.overall_mae,     1, '');
  animCount('valRMSE',    meta.overall_rmse,    1, '');
  animCount('valR2',      meta.overall_r2,      4, '');
  animCount('valSamples', meta.total_samples,   0, '');
}

function animCount(id, target, decimals, suffix) {
  const el = document.getElementById(id);
  const start = 0, dur = 1400;
  let startTime = null;

  function step(ts) {
    if (!startTime) startTime = ts;
    const prog = Math.min((ts - startTime) / dur, 1);
    const ease = 1 - Math.pow(1 - prog, 3); // cubic ease-out
    const val  = start + (target - start) * ease;
    el.textContent = val.toFixed(decimals) + suffix;
    if (prog < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── ENTITY SELECTORS ──────────────────────────────────────────────────────────
function buildEntitySelectors() {
  const stores   = [...new Set(meta.entities.map(e => e.split(' - ')[0]))].sort();
  const products = [...new Set(meta.entities.map(e => e.split(' - ')[1]))].sort();

  const storeEl   = document.getElementById('storeSelect');
  const productEl = document.getElementById('productSelect');

  stores.forEach(s => {
    const o = document.createElement('option'); o.value = s; o.textContent = s;
    storeEl.appendChild(o);
  });
  products.forEach(p => {
    const o = document.createElement('option'); o.value = p; o.textContent = p;
    productEl.appendChild(o);
  });

  storeEl.addEventListener('change',   () => updateForecastChart());
  productEl.addEventListener('change', () => updateForecastChart());
}

function selectedEntity() {
  const s = document.getElementById('storeSelect').value;
  const p = document.getElementById('productSelect').value;
  return `${s} - ${p}`;
}

// ── FORECAST CHART ────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  responsive: true, maintainAspectRatio: true,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(5,8,16,.9)',
      borderColor: 'rgba(255,255,255,.1)',
      borderWidth: 1,
      padding: { x: 12, y: 10 },
      titleColor: '#94a3b8',
      bodyColor: '#f1f5f9',
      titleFont: { family: 'Inter', size: 11, weight: '600' },
      bodyFont:  { family: 'Inter', size: 13, weight: '700' },
      callbacks: {
        title: items => items[0].label,
        label: item => ` ${item.dataset.label}: ${Math.round(item.raw)} units`,
      },
    },
  },
  scales: {
    x: {
      ticks: { color: 'rgba(255,255,255,.35)', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 10 },
      grid:  { color: 'rgba(255,255,255,.05)', drawTicks: false },
      border:{ color: 'transparent' },
    },
    y: {
      ticks: { color: 'rgba(255,255,255,.35)', font: { family: 'JetBrains Mono', size: 10 } },
      grid:  { color: 'rgba(255,255,255,.05)', drawTicks: false },
      border:{ color: 'transparent', dash: [4,4] },
    },
  },
  animation: { duration: 600, easing: 'easeInOutQuart' },
};

function buildForecastChart() {
  const ctx = document.getElementById('forecastChart').getContext('2d');

  const entity = selectedEntity();
  const data   = forecasts[entity] || Object.values(forecasts)[0];

  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [
        {
          label: 'Actual',
          data: data.actual,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,.08)',
          borderWidth: 2.5,
          pointRadius: 0, pointHoverRadius: 5,
          pointHoverBackgroundColor: '#6366f1',
          fill: true,
          tension: .35,
        },
        {
          label: 'Predicted',
          data: data.predicted,
          borderColor: '#06b6d4',
          borderWidth: 2,
          borderDash: [6, 3],
          pointRadius: 0, pointHoverRadius: 5,
          pointHoverBackgroundColor: '#06b6d4',
          fill: false,
          tension: .35,
        },
      ],
    },
    options: CHART_DEFAULTS,
  });
}

function updateForecastChart() {
  const entity = selectedEntity();
  const data   = forecasts[entity] || Object.values(forecasts)[0];
  forecastChart.data.labels            = data.dates;
  forecastChart.data.datasets[0].data  = data.actual;
  forecastChart.data.datasets[1].data  = data.predicted;
  forecastChart.update('active');
}

// ── TRAINING HISTORY CHART ────────────────────────────────────────────────────
function buildHistoryChart() {
  const ctx    = document.getElementById('historyChart').getContext('2d');
  const epochs = historyData.loss.map((_,i) => `Epoch ${i+1}`);

  historyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Train Loss',
          data: historyData.loss,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,.07)',
          borderWidth: 2.5,
          pointRadius: 0, pointHoverRadius: 4,
          fill: true, tension: .4,
        },
        {
          label: 'Val Loss',
          data: historyData.val_loss,
          borderColor: '#06b6d4',
          borderWidth: 2, borderDash: [6, 3],
          pointRadius: 0, pointHoverRadius: 4,
          fill: false, tension: .4,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: {
            title: items => items[0].label,
            label: item => ` ${item.dataset.label}: ${item.raw.toFixed(5)}`,
          },
        },
      },
    },
  });
}

// ── TAB TOGGLE ────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-selected', b === btn);
    });
    document.getElementById('paneForecast').classList.toggle('hidden', tab !== 'forecast');
    document.getElementById('paneHistory') .classList.toggle('hidden', tab !== 'history');

    // Resize chart after becoming visible
    setTimeout(() => {
      (tab === 'forecast' ? forecastChart : historyChart).resize();
    }, 50);
  });
});

// ── PER-ENTITY PERFORMANCE LIST ───────────────────────────────────────────────
function buildPerfList() {
  const list    = document.getElementById('perfList');
  const maeMap  = meta.per_entity_mae;
  const sorted  = Object.entries(maeMap).sort((a,b) => a[1] - b[1]);
  const maxMAE  = Math.max(...sorted.map(e => e[1]));

  sorted.forEach(([entity, mae]) => {
    const pct = ((mae / maxMAE) * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'perf-row';
    row.innerHTML = `
      <span class="perf-entity" title="${entity}">${entity.replace('Store_','').replace(' - ',' · ')}</span>
      <div class="perf-bar-wrap">
        <div class="perf-bar" style="width:0%" data-pct="${pct}"></div>
      </div>
      <span class="perf-mae">${mae}</span>
    `;
    list.appendChild(row);
  });

  // Animate bars
  requestAnimationFrame(() => {
    list.querySelectorAll('.perf-bar').forEach(bar => {
      setTimeout(() => { bar.style.width = bar.dataset.pct + '%'; }, 200);
    });
  });
}

// ── COST SAVINGS PANEL ────────────────────────────────────────────────────────
function renderCostPanel() {
  const hc   = meta.holding_cost_per_unit  || 3.0;
  const sc   = meta.stockout_cost_per_unit || 22.5;
  const mae  = meta.overall_mae;
  const avg  = meta.avg_demand;

  // Estimated savings: naive vs AI ordering
  // Naive safety stock ≈ 2×MAE; AI-based ≈ 0.5×MAE
  const naiveOverstock  = 2   * mae;
  const aiOverstock     = 0.5 * mae;
  const holdingSaved    = ((naiveOverstock - aiOverstock) * hc).toFixed(2);

  // Stockout events reduced (rough model)
  const stockoutSaved = (mae * 0.4 * sc).toFixed(2);

  const bandLow  = Math.max(0, avg - mae).toFixed(1);
  const bandHigh = (avg + mae).toFixed(1);

  document.getElementById('costHolding').textContent   = `₹${holdingSaved}`;
  document.getElementById('costStockout').textContent  = `₹${stockoutSaved}`;
  document.getElementById('costBand').textContent      = `${bandLow} – ${bandHigh} units/day`;
  document.getElementById('holdConst').textContent     = hc;
  document.getElementById('stockConst').textContent    = sc;
}

// ── NAVBAR SCROLL SHADOW ──────────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  nav.style.boxShadow = window.scrollY > 10
    ? '0 4px 32px rgba(0,0,0,.45)'
    : '';
}, { passive: true });

// ── START ─────────────────────────────────────────────────────────────────────
init();
