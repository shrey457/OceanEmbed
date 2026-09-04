// OceanEmbed Interactive Logic
const daySlider = document.getElementById('day-slider');
const dayVal = document.getElementById('day-val');
const depthSlider = document.getElementById('depth-slider');
const depthVal = document.getElementById('depth-val');
const loadBtn = document.getElementById('load-btn');
const loading = document.getElementById('loading');

// DOM Metrics
const mTrue = document.getElementById('true-mean');
const mPred = document.getElementById('pred-mean');
const mErr = document.getElementById('abs-error');

let globalConfig = null;
let currentData = null; // Holds current day's true and pred cubes

// Common Layout Configuration for Plotly
const plotLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { t: 20, r: 20, b: 40, l: 40 },
    font: { family: 'Outfit', color: '#94a3b8' },
    xaxis: { title: 'Longitude', gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
    yaxis: { title: 'Latitude', gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
};

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        globalConfig = await res.json();
        
        // Setup slider bounds
        daySlider.max = globalConfig.num_days - 1;
        depthSlider.max = globalConfig.depths.length - 1;
        
    } catch (e) {
        console.error("Failed to load config", e);
    }
}

async function loadDayData(dayIdx) {
    loading.classList.add('active');
    try {
        const res = await fetch(`/api/predict/${dayIdx}`);
        currentData = await res.json();
        
        // Enable depth slider if disabled
        depthSlider.disabled = false;
        renderDepth(depthSlider.value);
        
    } catch (e) {
        console.error("Failed to load prediction", e);
    }
    loading.classList.remove('active');
}

function computeMean(array2D) {
    let sum = 0, count = 0;
    for(let r=0; r<array2D.length; r++) {
        for(let c=0; c<array2D[r].length; c++) {
            if(array2D[r][c] !== null) {
                sum += array2D[r][c];
                count++;
            }
        }
    }
    return count > 0 ? (sum / count) : 0;
}

function computeErrorMatrix(true2D, pred2D) {
    let err2D = [];
    let absSum = 0, count = 0;
    
    for(let r=0; r<true2D.length; r++) {
        let row = [];
        for(let c=0; c<true2D[r].length; c++) {
            if(true2D[r][c] !== null && pred2D[r][c] !== null) {
                let err = pred2D[r][c] - true2D[r][c];
                row.push(err);
                absSum += Math.abs(err);
                count++;
            } else {
                row.push(null);
            }
        }
        err2D.push(row);
    }
    return { err2D, absErrorMean: count > 0 ? (absSum / count) : 0 };
}

function renderDepth(depthIdx) {
    if (!currentData || !globalConfig) return;
    
    const trueSlice = currentData.true[depthIdx];
    const predSlice = currentData.pred[depthIdx];
    const { err2D, absErrorMean } = computeErrorMatrix(trueSlice, predSlice);
    
    // Update Metrics
    mTrue.innerText = computeMean(trueSlice).toFixed(3) + ' °C';
    mPred.innerText = computeMean(predSlice).toFixed(3) + ' °C';
    mErr.innerText = absErrorMean.toFixed(4) + ' °C';
    
    const depthStr = globalConfig.depths[depthIdx] + 'm';
    depthVal.innerText = depthStr;
    
    // Trace configurations
    const traceTrue = {
        z: trueSlice,
        x: globalConfig.lons,
        y: globalConfig.lats,
        type: 'heatmap',
        colorscale: 'Viridis',
        zmin: 10, zmax: 30, // Approximate oceanic limits
        colorbar: { title: '°C' }
    };
    
    const tracePred = {
        z: predSlice,
        x: globalConfig.lons,
        y: globalConfig.lats,
        type: 'heatmap',
        colorscale: 'Viridis',
        zmin: 10, zmax: 30,
        colorbar: { title: '°C' }
    };
    
    const traceErr = {
        z: err2D,
        x: globalConfig.lons,
        y: globalConfig.lats,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        zmin: -2, zmax: 2,
        colorbar: { title: 'Delta °C' }
    };

    Plotly.react('true-map', [traceTrue], plotLayout, {responsive: true});
    Plotly.react('pred-map', [tracePred], plotLayout, {responsive: true});
    Plotly.react('error-map', [traceErr], plotLayout, {responsive: true});
}

// Event Listeners
daySlider.addEventListener('input', (e) => {
    dayVal.innerText = `Day ${parseInt(e.target.value) + 1}`;
});

depthSlider.addEventListener('input', (e) => {
    depthVal.innerText = `${globalConfig.depths[e.target.value]}m`;
    renderDepth(e.target.value);
});

loadBtn.addEventListener('click', () => {
    loadDayData(daySlider.value);
});

// Initialize
fetchConfig().then(() => {
    // Automatically load day 0 on start
    loadDayData(0);
});
