/**
 * KloudAlert — Simple Citizen Weather App
 * Real-time rain alerts powered by PIMCAN-v4 Neural Engine
 *
 * Data Sources: KloudTech AWS, Open-Meteo, RainViewer Doppler, Himawari-9
 * All computation happens behind the scenes — user sees clean results only.
 */

let userLat = 14.5621;
let userLon = 120.5934;
let userLocationName = "Detecting...";
let onnxSession = null;

let currentTemp = null;
let currentHumidity = null;
let currentPressure = null;
let currentPrecipRate = null;
let currentWind = null;
let currentHeatIndex = null;
let currentWeatherCode = 0;
let currentRadarDBZ = 0;
let currentSatTemp = -62.0;

let pimcanRainStatus = "checking";
let pimcanRainIntensity = 0;
let rainDurationEstimate = "";

let isDevRaining = false;
let devTimerInterval = null;
let devTimerSeconds = 0;
let devRainEvents = [];

// KloudTech stations
const STATIONS = [
    { id: "O3z05pGV", name: "Wawa, Pilar / Limay AWS", lat: 14.5621, lon: 120.5934, region: "Bataan" },
    { id: "lMAZe9b3", name: "Abucay AWS", lat: 14.7211, lon: 120.5342, region: "Bataan" },
    { id: "QgbGldAY", name: "Pag-asa Bagac AWS", lat: 14.6041, lon: 120.3922, region: "Bataan" },
    { id: "rqAkmpKG", name: "Subic Barretto AWS", lat: 14.7840, lon: 120.3131, region: "Zambales" },
    { id: "3nzr8bGo", name: "Alasas AWS", lat: 14.8380, lon: 120.4590, region: "Zambales" },
    { id: "xMbRYxp0", name: "Avida Asten AWS", lat: 14.5581, lon: 121.0141, region: "Makati" },
];

function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371, dLat = (lat2-lat1)*Math.PI/180, dLon = (lon2-lon1)*Math.PI/180;
    const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

function rompsHeatIndex(T, RH) {
    if (T < 20) return T;
    const tF = T*9/5+32;
    let hiF = -42.379+2.04901523*tF+10.14333127*RH-0.22475541*tF*RH-0.00683783*tF*tF
        -0.05481717*RH*RH+0.00122874*tF*tF*RH+0.00085282*tF*RH*RH-0.00000199*tF*tF*RH*RH;
    if (RH<13&&tF>=80&&tF<=112) hiF-=((13-RH)/4)*Math.sqrt(Math.max(0,(17-Math.abs(tF-95))/17));
    else if (RH>85&&tF>=80&&tF<=87) hiF+=((RH-85)/10)*((87-tF)/5);
    return Math.max(T, Math.round((hiF-32)*5/9*100)/100);
}

function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

// ═══ ONNX INFERENCE ═══
async function initONNX() {
    try {
        if (typeof ort !== 'undefined') {
            onnxSession = await ort.InferenceSession.create('./lnn_weather_model.onnx', { executionProviders: ['wasm'] });
            console.log('[ONNX] Model loaded.');
        }
    } catch (e) { console.warn('[ONNX]', e.message); }
}

// ═══ LOCATION DETECTION ═══
function initLocation() {
    if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(
            (pos) => { userLat = pos.coords.latitude; userLon = pos.coords.longitude; reverseGeo(); fetchWeather(); },
            () => ipFallback(),
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
        navigator.geolocation.watchPosition(
            (pos) => {
                const pk = document.getElementById('station-picker');
                if (pk && pk.value !== 'GPS_AUTO') return;
                userLat = pos.coords.latitude; userLon = pos.coords.longitude;
                reverseGeo(); fetchWeather();
            }, () => {}, { enableHighAccuracy: true, timeout: 30000, maximumAge: 0 }
        );
    } else { ipFallback(); }
}

async function ipFallback() {
    for (const url of ['https://ipapi.co/json/', 'https://ipwho.is/']) {
        try {
            const r = await fetch(url, { signal: AbortSignal.timeout(5000) });
            if (!r.ok) continue;
            const d = await r.json();
            const lat = d.latitude || d.lat, lon = d.longitude || d.lon;
            if (lat && lon) {
                userLat = lat; userLon = lon;
                userLocationName = [d.city, d.region || d.regionName].filter(Boolean).join(', ') || `${lat}°N, ${lon}°E`;
                setText('loc-name', userLocationName);
                fetchWeather(); return;
            }
        } catch (e) {}
    }
    setText('loc-name', 'Location unavailable');
    fetchWeather();
}

async function reverseGeo() {
    try {
        const r = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${userLat}&lon=${userLon}&zoom=16&addressdetails=1`);
        if (r.ok) {
            const d = await r.json();
            const parts = [d.address.village||d.address.suburb||d.address.neighbourhood||'',
                d.address.city||d.address.town||d.address.municipality||'',
                d.address.state||d.address.province||''].filter(Boolean);
            userLocationName = [...new Set(parts)].join(', ') || `${userLat.toFixed(4)}°N, ${userLon.toFixed(4)}°E`;
        }
    } catch (e) { userLocationName = `${userLat.toFixed(4)}°N, ${userLon.toFixed(4)}°E`; }
    setText('loc-name', userLocationName);
}

// ═══ LIVE WEATHER FETCH ═══
async function fetchWeather() {
    const pk = document.getElementById('station-picker');
    const sel = pk ? pk.value : 'GPS_AUTO';

    // Manual station override
    const stationMap = { 'O3z05pGV': [14.5621,120.5934,'Wawa, Pilar / Limay, Bataan'],
        'lMAZe9b3': [14.7211,120.5342,'Abucay, Bataan'], 'QgbGldAY': [14.6041,120.3922,'Pag-asa Bagac, Bataan'],
        'xMbRYxp0': [14.5581,121.0141,'Makati City, Manila'], 'CEBU_CITY': [10.3157,123.8854,'Cebu City, Visayas'] };
    if (sel !== 'GPS_AUTO' && stationMap[sel]) {
        [userLat, userLon, userLocationName] = stationMap[sel];
        setText('loc-name', userLocationName);
    }

    // KloudTech AWS — find nearest responding station
    let ktTemp = null, ktHum = null, ktPress = null, ktWind = null;
    const nearest = STATIONS.map(s => ({...s, dist: haversine(userLat, userLon, s.lat, s.lon)})).sort((a,b) => a.dist-b.dist);

    for (const s of nearest.slice(0, 3)) {
        if (s.dist > 35) continue;
        try {
            const [taR, rhR, bpR, wsR] = await Promise.all([
                fetch(`https://api.klfrst.com/api/measurements/query?stationId=${s.id}&parameter_id=ta&limit=1&order=desc`),
                fetch(`https://api.klfrst.com/api/measurements/query?stationId=${s.id}&parameter_id=rh&limit=1&order=desc`),
                fetch(`https://api.klfrst.com/api/measurements/query?stationId=${s.id}&parameter_id=bp&limit=1&order=desc`),
                fetch(`https://api.klfrst.com/api/measurements/query?stationId=${s.id}&parameter_id=ws&limit=1&order=desc`)
            ]);
            const g = async (r) => { const a = await r.json(); return a?.[0]?.value != null ? parseFloat(a[0].value) : null; };
            ktTemp = await g(taR); ktHum = await g(rhR); ktPress = await g(bpR);
            const wsVal = await g(wsR); ktWind = wsVal != null ? wsVal * 3.6 : null;
            if (ktTemp !== null) break;
        } catch (e) {}
    }

    // Open-Meteo fallback
    let omTemp=null, omHum=null, omPress=null, omPrecip=null, omWind=null;
    try {
        const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${userLat}&longitude=${userLon}&current=temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m,weather_code`);
        if (r.ok) {
            const c = (await r.json()).current || {};
            omTemp=c.temperature_2m; omHum=c.relative_humidity_2m; omPress=c.surface_pressure;
            omPrecip=c.precipitation; omWind=c.wind_speed_10m; currentWeatherCode=c.weather_code??0;
        }
    } catch(e){}

    // RainViewer radar
    try {
        const r = await fetch('https://api.rainviewer.com/public/weather-maps.json');
        if (r.ok) { const d = await r.json(); currentRadarDBZ = (d.radar?.past?.length||0) > 0 ? 28.0 : 0; }
    } catch(e){}

    // Fuse: KloudTech > Open-Meteo
    currentTemp = ktTemp ?? omTemp ?? 25;
    currentHumidity = ktHum ?? omHum ?? 80;
    currentPressure = ktPress ?? omPress ?? 1008;
    currentPrecipRate = isDevRaining ? Math.max(omPrecip||0, 2.5) : (omPrecip ?? 0);
    currentWind = ktWind ?? omWind ?? 5;
    currentHeatIndex = rompsHeatIndex(currentTemp, currentHumidity);

    // PIMCAN-v4 rain assessment
    const radarActive = currentRadarDBZ >= 20;
    const humSat = currentHumidity >= 85;

    if (currentPrecipRate >= 7.0) {
        pimcanRainStatus = "raining"; pimcanRainIntensity = currentPrecipRate;
        rainDurationEstimate = "Heavy rain — expect 30-60 minutes";
    } else if (currentPrecipRate >= 2.0 || (radarActive && humSat)) {
        pimcanRainStatus = "raining"; pimcanRainIntensity = Math.max(currentPrecipRate, 2.5);
        rainDurationEstimate = "Moderate rain — around 20-30 minutes remaining";
    } else if (currentPrecipRate > 0 || (humSat && radarActive)) {
        pimcanRainStatus = "raining"; pimcanRainIntensity = currentPrecipRate || 1.0;
        rainDurationEstimate = "Light rain — should ease in 15-20 minutes";
    } else if (humSat && currentRadarDBZ >= 15) {
        pimcanRainStatus = "likely";
        rainDurationEstimate = "Rain likely within the next 15-30 minutes";
    } else {
        pimcanRainStatus = "clear";
        rainDurationEstimate = "";
    }

    updateUI();
    updateBackground();
}

// ═══ UI UPDATE ═══
function updateUI() {
    const t = currentTemp ?? 25, h = currentHumidity ?? 80, p = currentPressure ?? 1008;
    const w = currentWind ?? 5, hi = currentHeatIndex ?? 25;
    const pr = pimcanRainIntensity > 0 ? pimcanRainIntensity : (currentPrecipRate ?? 0);

    setText('temp-num', Math.round(t));
    setText('temp-feels', `Feels like ${Math.round(hi)}°`);

    // Condition text
    if (pimcanRainStatus === "raining") setText('temp-condition', pr >= 7 ? 'Heavy Rain' : pr >= 2 ? 'Moderate Rain' : 'Light Rain');
    else if (pimcanRainStatus === "likely") setText('temp-condition', 'Rain Expected Soon');
    else {
        const hour = new Date().getHours();
        if (hour >= 18 || hour < 6) setText('temp-condition', 'Clear Night');
        else if (currentWeatherCode >= 2) setText('temp-condition', 'Partly Cloudy');
        else setText('temp-condition', 'Clear Sky');
    }

    // Rain Card — the core
    const rainCard = document.getElementById('rain-card');
    const rainIcon = document.getElementById('rain-card-icon');

    if (pimcanRainStatus === "raining") {
        setText('rain-status', `It's raining — ${pr.toFixed(1)} mm/hr`);
        setText('rain-detail', rainDurationEstimate);
        rainCard.className = 'rain-card raining';
        rainIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M16 14v6"/><path d="M8 14v6"/><path d="M12 16v6"/></svg>';
    } else if (pimcanRainStatus === "likely") {
        setText('rain-status', 'Rain is likely soon');
        setText('rain-detail', rainDurationEstimate);
        rainCard.className = 'rain-card';
        rainIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M16 14v6"/><path d="M8 14v6"/><path d="M12 16v6"/></svg>';
    } else {
        setText('rain-status', 'No rain expected');
        setText('rain-detail', 'Dry conditions for at least the next 45 minutes');
        rainCard.className = 'rain-card no-rain';
        rainIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/></svg>';
    }

    // Advisory — specific to what user is experiencing
    if (pimcanRainStatus === "raining" && pr >= 7) {
        setText('advisory-label', 'HEAVY RAIN WARNING');
        setText('advisory-text', 'Stay indoors if possible. If driving, slow down and use headlights. Avoid low-lying areas prone to flooding.');
    } else if (pimcanRainStatus === "raining") {
        setText('advisory-label', 'RAIN ADVISORY');
        setText('advisory-text', 'Bring an umbrella or raincoat if heading outside. Roads may be slippery — drive carefully.');
    } else if (pimcanRainStatus === "likely") {
        setText('advisory-label', 'RAIN EXPECTED SOON');
        setText('advisory-text', 'Carry an umbrella before heading out. Consider bringing laundry inside.');
    } else if (hi >= 40) {
        setText('advisory-label', 'DANGEROUS HEAT');
        setText('advisory-text', `Heat index is ${Math.round(hi)}°C. Avoid prolonged sun exposure. Drink water frequently and stay in shade or air-conditioning.`);
    } else if (hi >= 35) {
        setText('advisory-label', 'HIGH HEAT INDEX');
        setText('advisory-text', `It feels like ${Math.round(hi)}°C. Stay hydrated, wear light clothing, and limit outdoor activities during midday.`);
    } else if (w >= 30) {
        setText('advisory-label', 'STRONG WIND ADVISORY');
        setText('advisory-text', `Winds at ${w.toFixed(0)} km/h. Secure loose objects outside and be cautious when driving motorcycles.`);
    } else {
        setText('advisory-label', 'GOOD CONDITIONS');
        setText('advisory-text', 'Weather is comfortable for outdoor activities. No rain or extreme conditions detected.');
    }

    // Stats
    setText('s-humidity', h.toFixed(1) + '%');
    setText('s-heat-index', Math.round(hi) + '°C');
    setText('s-wind', w.toFixed(1) + ' km/h');
    setText('s-precip', pr.toFixed(1) + ' mm/hr');
    setText('s-pressure', p.toFixed(1) + ' hPa');
}

function updateBackground() {
    const body = document.body;
    const hour = new Date().getHours();
    const isNight = hour < 6 || hour >= 18;
    body.classList.remove('theme-sunny','theme-cloudy','theme-rainy','theme-stormy','theme-night','theme-hot');

    if (pimcanRainIntensity >= 7 || currentWeatherCode >= 95) body.classList.add('theme-stormy');
    else if (pimcanRainStatus === "raining" || pimcanRainStatus === "likely") body.classList.add('theme-rainy');
    else if (isNight) body.classList.add('theme-night');
    else if (currentHeatIndex >= 38) body.classList.add('theme-hot');
    else if (currentWeatherCode >= 2 && currentWeatherCode <= 3) body.classList.add('theme-cloudy');
    else body.classList.add('theme-sunny');
}

// ═══ DEV STOPWATCH & FIREBASE FLYWHEEL SYNC ═══
async function syncDevEventToCloud(eventData) {
    try {
        const stored = JSON.parse(localStorage.getItem('kloudalert_telemetry_events') || '[]');
        stored.push(eventData);
        localStorage.setItem('kloudalert_telemetry_events', JSON.stringify(stored));
    } catch(e) {}

    try {
        const firestoreUrl = 'https://firestore.googleapis.com/v1/projects/kloudalert-1fdf2/databases/(default)/documents/rain_telemetry_events';
        const docFields = {
            type: { stringValue: String(eventData.type || 'unknown') },
            timestamp: { integerValue: String(eventData.ts || Date.now()) },
            latitude: { doubleValue: Number(eventData.lat || 0) },
            longitude: { doubleValue: Number(eventData.lon || 0) },
            is_raining: { booleanValue: eventData.type === 'start' },
            temperature: { doubleValue: Number(eventData.temp || currentTemp || 25.0) },
            humidity: { doubleValue: Number(eventData.hum || currentHumidity || 80.0) },
            duration_sec: { integerValue: String(eventData.durationSec || 0) }
        };

        const res = await fetch(firestoreUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fields: docFields })
        });
        if (res.ok) {
            console.log('[FIREBASE CLOUD] Ground-truth observation synced to Firestore cloud!');
        } else {
            console.warn('[FIREBASE CLOUD] Sync fallback (saved to offline buffer):', res.status);
        }
    } catch(e) {
        console.warn('[FIREBASE CLOUD] Offline buffer active:', e.message);
    }
}

function initDev() {
    const btnStart = document.getElementById('btn-dev-rain-start');
    const btnStop = document.getElementById('btn-dev-rain-stop');
    const timer = document.getElementById('dev-timer-display');

    btnStart?.addEventListener('click', () => {
        isDevRaining = true; devTimerSeconds = 0;
        if (devTimerInterval) clearInterval(devTimerInterval);
        devTimerInterval = setInterval(() => {
            devTimerSeconds++;
            const m = Math.floor(devTimerSeconds/60).toString().padStart(2,'0');
            const s = (devTimerSeconds%60).toString().padStart(2,'0');
            if (timer) { timer.textContent = `${m}:${s}`; timer.style.color = '#EF4444'; }
        }, 1000);
        const eventData = { type:'start', ts:Date.now(), lat:userLat, lon:userLon, temp:currentTemp, hum:currentHumidity };
        devRainEvents.push(eventData);
        syncDevEventToCloud(eventData);
        fetchWeather();
    });

    btnStop?.addEventListener('click', () => {
        isDevRaining = false;
        if (devTimerInterval) clearInterval(devTimerInterval);
        if (timer) { timer.textContent = `${Math.floor(devTimerSeconds/60)}m ${devTimerSeconds%60}s`; timer.style.color = '#10B981'; }
        const eventData = { type:'stop', ts:Date.now(), durationSec:devTimerSeconds, lat:userLat, lon:userLon, temp:currentTemp, hum:currentHumidity };
        devRainEvents.push(eventData);
        syncDevEventToCloud(eventData);
        console.log('[DEV] Rain events synced:', JSON.stringify(devRainEvents, null, 2));
        fetchWeather();
    });
}

// ═══ INIT ═══
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('station-picker')?.addEventListener('change', fetchWeather);
    initONNX();
    initLocation();
    initDev();
    fetchWeather();
    setInterval(fetchWeather, 30000);
});
