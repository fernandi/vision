// ── Weather mode (hidden: key M, or ?meteo) ──────────────────────────────────
// Today's or tomorrow's weather somewhere, illustrated by paintings from the
// index that show the same sky. Forecasts and place search come from Open-Meteo
// (no key needed); the name of a GPS position comes from BigDataCloud.
// Loaded after app.js, whose globals (results, showView, loadPage…) it uses.

const Weather = (() => {
    const PLACE_KEY = 'glane.weather.place';
    const FORECAST_TTL_MS = 15 * 60 * 1000;
    const header = $('weather-header');
    const pickerForm = $('weather-place-form');
    const placeInput = $('weather-place-input');
    const placeList = $('weather-place-list');

    let place = store.get(PLACE_KEY, null);   // { name, lat, lon }
    let day = 'today';                          // 'today' | 'tomorrow'
    let forecast = null;                        // { key, at, data }
    let saved = null;                           // results + URL to restore on leaving
    let lookupTimer = null;
    let lookupGen = 0;

    // WMO weather codes → a few skies a painting can show.
    function sky(code) {
        if (code <= 1) return 'clear';
        if (code === 2) return 'partly';
        if (code === 3) return 'overcast';
        if (code === 45 || code === 48) return 'fog';
        if (code >= 51 && code <= 57) return 'drizzle';
        if ((code >= 61 && code <= 67) || (code >= 80 && code <= 82)) return 'rain';
        if ((code >= 71 && code <= 77) || code === 85 || code === 86) return 'snow';
        if (code >= 95) return 'storm';
        return 'overcast';
    }

    const PROMPTS = {
        clear: 'painting of a sunny landscape under a clear blue sky, bright sunlight',
        partly: 'painting of a landscape with white clouds drifting across a blue sky',
        overcast: 'painting of a grey overcast sky over a quiet landscape, dull cloudy day',
        fog: 'painting of a misty foggy landscape, soft grey haze',
        drizzle: 'painting of a damp grey day with fine drizzle, wet ground',
        rain: 'painting of a rainy day, rain falling, wet streets and umbrellas',
        snow: 'painting of a snowy winter landscape, snow falling',
        storm: 'painting of a thunderstorm, lightning and dark stormy clouds',
    };
    const NIGHT_PROMPTS = {
        clear: 'painting of a clear night, moonlight and stars over a landscape',
        partly: 'painting of a cloudy night sky, moon behind clouds',
        overcast: 'painting of a dark cloudy night, dim light',
    };

    // What the forecast says for the chosen day, reduced to what a painting shows.
    function reading(data, which) {
        if (which === 'today') {
            const c = data.current;
            return {
                sky: sky(c.weather_code),
                night: c.is_day === 0,
                temp: c.temperature_2m,
                windy: c.wind_speed_10m >= 40,
                label: `${Math.round(c.temperature_2m)}°`,
            };
        }
        const d = data.daily;
        const max = d.temperature_2m_max[1];
        return {
            sky: sky(d.weather_code[1]),
            night: false,
            temp: max,
            windy: d.wind_speed_10m_max[1] >= 40,
            label: `${Math.round(d.temperature_2m_min[1])}° – ${Math.round(max)}°`,
        };
    }

    function promptFor(r) {
        let p = (r.night && NIGHT_PROMPTS[r.sky]) || PROMPTS[r.sky];
        if (r.night && !NIGHT_PROMPTS[r.sky]) p += ', at night';
        if (r.sky === 'clear' && !r.night && r.temp >= 27) p = 'painting of a hot summer day, blazing sun over a dry landscape';
        if (['clear', 'partly', 'overcast'].includes(r.sky) && r.temp <= 2) p += ', cold winter day, frost';
        if (r.windy) p += ', strong wind bending the trees';
        return p;
    }

    // "il fait beau et chaud" / "it will rain"…
    function phrase(r, which) {
        const tense = which === 'today' ? 'now' : 'later';
        if (r.night && ['clear', 'partly', 'overcast'].includes(r.sky)) {
            return t(`weather.night.${r.sky === 'clear' ? 'clear' : 'cloudy'}`);
        }
        let s = t(`weather.${tense}.${r.sky}`);
        if (r.sky === 'clear' || r.sky === 'overcast') {
            if (r.temp >= 27) s += t('weather.and.hot');
            else if (r.temp <= 5) s += t('weather.and.cold');
        }
        return s;
    }

    // A translated sentence with {place} and {weather} set as styled spans.
    function sentence(which, placeName, weatherText) {
        const node = el('p', 'weather-sentence');
        const template = t(`weather.sentence.${which}`, { place: '\u0001', weather: '\u0002' });
        for (const part of template.split(/(\u0001|\u0002)/)) {
            if (part === '\u0001') node.appendChild(el('span', 'weather-place', placeName));
            else if (part === '\u0002') node.appendChild(el('em', 'weather-words', weatherText));
            else if (part) node.appendChild(document.createTextNode(part));
        }
        return node;
    }

    async function getJSON(url) {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Request failed (${resp.status})`);
        return resp.json();
    }

    async function loadForecast() {
        const key = `${place.lat.toFixed(3)},${place.lon.toFixed(3)}`;
        if (forecast && forecast.key === key && Date.now() - forecast.at < FORECAST_TTL_MS) return forecast.data;
        const qs = new URLSearchParams({
            latitude: place.lat, longitude: place.lon, timezone: 'auto', forecast_days: '2',
            current: 'temperature_2m,weather_code,is_day,wind_speed_10m',
            daily: 'weather_code,temperature_2m_max,temperature_2m_min,wind_speed_10m_max',
        });
        const data = await getJSON(`https://api.open-meteo.com/v1/forecast?${qs}`);
        forecast = { key, at: Date.now(), data };
        return data;
    }

    function dateLabel(data, which) {
        const iso = data.daily.time[which === 'today' ? 0 : 1];
        const text = new Date(`${iso}T12:00:00`).toLocaleDateString(LANG, { weekday: 'long', day: 'numeric', month: 'long' });
        return text.charAt(0).toUpperCase() + text.slice(1);
    }

    // ── Rendering ────────────────────────────────────────────────────────────
    function showPicker(show) {
        pickerForm.hidden = !show;
        $('weather-change').hidden = show || !place;
        if (show) {
            placeInput.value = '';
            placeList.replaceChildren();
            placeInput.focus({ preventScroll: true });
        }
    }

    async function render() {
        const gen = renderGen;
        const body = $('weather-body');
        // Until the forecast is in, infinite scroll must not page through the previous search.
        results.hasMore = false;
        $('weather-day').hidden = !place;
        if (!place) {
            body.replaceChildren(el('p', 'weather-sentence', t('weather.where')));
            $('weather-meta').textContent = '';
            showPicker(true);
            mainGrid.reset();
            setStatus(statusEl, '');
            return;
        }
        showPicker(false);
        body.replaceChildren(el('p', 'weather-sentence is-loading', t('weather.loading', { place: place.name })));
        $('weather-meta').textContent = '';
        let data;
        try {
            data = await loadForecast();
        } catch (err) {
            console.error('weather', err);
            if (gen === renderGen) body.replaceChildren(el('p', 'weather-sentence', t('weather.unavailable')));
            return;
        }
        if (gen !== renderGen) return;
        const r = reading(data, day);
        body.replaceChildren(sentence(day, place.name, phrase(r, day)));
        $('weather-meta').textContent = `${dateLabel(data, day)} · ${r.label}${r.windy ? ` · ${t('weather.windy')}` : ''}`;

        results.request = {
            query: promptFor(r),
            page_size: PAGE_SIZE,
            diversity: 0.3,
            combination_mode: 'purified',
            techniques: ['painting'],
        };
        results.offset = 0;
        results.hasMore = true;
        results.items = [];
        mainGrid.skeleton(14);
        loadPage(gen);
    }

    function refresh() {
        showView('weather');   // new renderGen: drops anything still loading
        render();
    }

    function setPlace(next) {
        place = next;
        store.set(PLACE_KEY, place);
        refresh();
    }

    // ── Place search and GPS position ────────────────────────────────────────
    async function lookup(text) {
        const gen = ++lookupGen;
        if (text.trim().length < 2) { placeList.replaceChildren(); return; }
        try {
            const qs = new URLSearchParams({ name: text.trim(), count: '6', language: LANG, format: 'json' });
            const data = await getJSON(`https://geocoding-api.open-meteo.com/v1/search?${qs}`);
            if (gen !== lookupGen) return;
            const found = data.results || [];
            placeList.replaceChildren(...(found.length ? found.map(p => {
                const btn = el('button', 'weather-place-option');
                btn.type = 'button';
                btn.append(el('span', null, p.name), el('span', 'weather-place-region', [p.admin1, p.country].filter(Boolean).join(', ')));
                btn.addEventListener('click', () => setPlace({ name: p.name, lat: p.latitude, lon: p.longitude }));
                return btn;
            }) : [el('p', 'settings-help', t('weather.noPlace'))]));
        } catch (err) {
            console.error('geocoding', err);
        }
    }

    // Without a name, the position itself: "48,86° N, 2,35° E".
    function coords(lat, lon) {
        const f = n => Math.abs(n).toLocaleString(LANG, { maximumFractionDigits: 2 });
        return `${f(lat)}° ${lat >= 0 ? 'N' : 'S'}, ${f(lon)}° ${lon >= 0 ? 'E' : 'W'}`;
    }

    async function placeName(lat, lon) {
        try {
            const qs = new URLSearchParams({ latitude: lat, longitude: lon, localityLanguage: LANG });
            const d = await getJSON(`https://api.bigdatacloud.net/data/reverse-geocode-client?${qs}`);
            return d.city || d.locality || d.principalSubdivision || coords(lat, lon);
        } catch {
            return coords(lat, lon);
        }
    }

    function locate() {
        if (!navigator.geolocation) { toast(t('weather.geoDenied')); return; }
        const btn = $('weather-locate');
        btn.disabled = true;
        navigator.geolocation.getCurrentPosition(async pos => {
            const { latitude: lat, longitude: lon } = pos.coords;
            const name = await placeName(lat, lon);
            btn.disabled = false;
            setPlace({ name, lat, lon });
        }, () => {
            btn.disabled = false;
            toast(t('weather.geoDenied'));
        }, { timeout: 15000, maximumAge: 30 * 60 * 1000 });
    }

    // ── Open / close ─────────────────────────────────────────────────────────
    function open() {
        if (view === 'weather') return;
        closeMenu();
        saved = { view, request: results.request, offset: results.offset, hasMore: results.hasMore,
                  items: results.items, url: new URLSearchParams(location.search).has('meteo') ? '' : location.search };
        history.replaceState(null, '', '?meteo');
        refresh();
    }

    function close() {
        if (view !== 'weather') return;
        const s = saved;
        saved = null;
        history.replaceState(null, '', s?.url || location.pathname);
        if (s && s.request) {
            Object.assign(results, { request: s.request, offset: s.offset, hasMore: s.hasMore, items: s.items });
            backToResults();
        } else {
            results.request = null;
            showView('empty');
        }
    }

    const typing = target => target.closest?.('input, textarea, select, [contenteditable="true"]');
    document.addEventListener('keydown', e => {
        if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target) || dialog.open) return;
        if (lightbox.classList.contains('is-open') || !picker.hidden) return;
        if (e.key === 'm' || e.key === 'M') { e.preventDefault(); view === 'weather' ? close() : open(); }
        else if (e.key === 'Escape' && view === 'weather') close();
    });

    header.querySelectorAll('input[name="weather-day"]').forEach(r => r.addEventListener('change', () => {
        day = r.value;
        refresh();
    }));
    $('weather-change').addEventListener('click', () => showPicker(true));
    $('weather-locate').addEventListener('click', locate);
    $('weather-close').addEventListener('click', close);
    placeInput.addEventListener('input', () => {
        clearTimeout(lookupTimer);
        lookupTimer = setTimeout(() => lookup(placeInput.value), 300);
    });
    placeInput.addEventListener('keydown', e => {
        if (e.key === 'Escape' && place) { e.stopPropagation(); showPicker(false); }
    });
    pickerForm.addEventListener('submit', async e => {
        e.preventDefault();
        clearTimeout(lookupTimer);
        if (!placeList.querySelector('.weather-place-option')) await lookup(placeInput.value);
        placeList.querySelector('.weather-place-option')?.click();
    });

    if (new URLSearchParams(location.search).has('meteo')) open();

    return { open, close };
})();
