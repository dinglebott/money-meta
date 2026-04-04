const API_BASE = 'https://tree-trader.up.railway.app';

const btn       = document.getElementById('fetch-btn');
const statusEl  = document.getElementById('status-text');
const candleEl  = document.getElementById('candle-card');
const predEl    = document.getElementById('pred-container');

function setStatus(msg, type = '') {
    statusEl.textContent = msg;
    statusEl.className = 'status-text' + (type ? ' ' + type : '');
}

function rsiClass(v) {
    if (v >= 70) return 'rsi-overbought';
    if (v <= 30) return 'rsi-oversold';
    return 'rsi-neutral';
}

function renderCandle(c) {
    const dt = new Date(c.timestamp); // start of candle
    var endDt = dt
    endDt = dt.setHours(dt.getHours() + 4) // end of candle

    const dtStr = dt.toLocaleString('en-SG', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
        hour12: false
    });
    if (endDt.getDate() == dt.getDate()) {
        var endDtStr = dt.toLocaleString("en-SG", {
            hour: "2-digit", minute: "2-digit",
            hour12: false
        })
    } else {
        var endDtStr = dt.toLocaleString("en-SG", {
            month: "short", day: "numeric",
            hour: "2-digit", minute: "2-digit",
            hour12: false
        })
    }
    const rsiCls = rsiClass(c.rsi);

    candleEl.innerHTML = `
    <div class="candle-grid">
        <div class="candle-item">
        <div class="candle-label">Open</div>
        <div class="candle-val">${c.open.toFixed(5)}</div>
        </div>
        <div class="candle-item">
        <div class="candle-label">High</div>
        <div class="candle-val">${c.high.toFixed(5)}</div>
        </div>
        <div class="candle-item">
        <div class="candle-label">Low</div>
        <div class="candle-val">${c.low.toFixed(5)}</div>
        </div>
        <div class="candle-item">
        <div class="candle-label">Close</div>
        <div class="candle-val">${c.close.toFixed(5)}</div>
        </div>
    </div>
    <div class="candle-meta">
        <div class="rsi-row">
        <span class="rsi-label">RSI-14</span>
        <span class="rsi-val ${rsiCls}">${c.rsi.toFixed(1)}</span>
        </div>
        <span class="timestamp">${dtStr} - ${endDtStr} SGT</span>
    </div>
    `;
}

function renderProbs(probs) {
    const dirs = ['down', 'flat', 'up'];
    return dirs.map((dir, idx) => {
    const pct = (probs[String(idx)] * 100).toFixed(1);
    return `
        <div class="prob-row">
        <span class="prob-dir ${dir}">${dir}</span>
        <div class="bar-track">
            <div class="bar-fill ${dir}" style="width: ${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
        </div>
    `;
    }).join('');
}

function renderPredictions(p) {
    predEl.innerHTML = `
    <div class="model-block">
        <div class="model-header">
        <span class="model-name">XGBoost</span>
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="model-ver">v${p.xgbModelVersion}</span>
            <span class="pred-pill ${p.xgbPred}">${["DOWN", "FLAT", "UP"][Number(p.xgbPred)]}</span>
        </div>
        </div>
        <div class="prob-rows">${renderProbs(p.xgbProbs)}</div>
    </div>

    <div class="model-block">
        <div class="model-header">
        <span class="model-name">CNN-LSTM</span>
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="model-ver">v${p.nnModelVersion}</span>
            <span class="pred-pill ${p.nnPred}">${["DOWN", "FLAT", "UP"][Number(p.nnPred)]}</span>
        </div>
        </div>
        <div class="prob-rows">${renderProbs(p.nnProbs)}</div>
    </div>
    `;
}

function setLoading(on) {
    btn.disabled = on;
    btn.innerHTML = on
    ? '<div class="spinner"></div><span>Fetching...</span>'
    : `<svg class="btn-icon" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
            <polygon points="2,1 13,7 2,13" fill="currentColor"/>
        </svg>Run inference`;
}

async function runInference() {
    setLoading(true);
    setStatus('Connecting to Railway...');

    try {
    const [candleRes, predRes] = await Promise.all([
        fetch(`${API_BASE}/candle`),
        fetch(`${API_BASE}/predict`)
    ]);

    if (!candleRes.ok) throw new Error(`/candle returned ${candleRes.status}`);
    if (!predRes.ok)   throw new Error(`/predict returned ${predRes.status}`);

    const candle = await candleRes.json();
    const pred   = await predRes.json();

    renderCandle(candle);
    renderPredictions(pred);

    const now = new Date().toLocaleTimeString('en-SG', { hour12: false });
    setStatus(`Last updated ${now}`, 'ok');

    } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, 'error');
    } finally {
    setLoading(false);
    }
}