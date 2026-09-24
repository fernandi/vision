'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
const API = '';
const PAGE_SIZE = 24;
const PRELOAD_TIMEOUT_MS = 12000;
const MIN_RATIO = 0.35;          // height / width limits for a grid cell
const MAX_RATIO = 2.2;
const MAX_COLLECTION_REFS = 12;
const MUSEUMS = {
    MET: 'The Met',
    CMA: 'Cleveland Museum of Art',
    artic: 'Art Institute of Chicago',
    Smithonian: 'Smithsonian',
};

const ICON = {
    collect: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>',
    collected: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>',
    visual: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="12" height="12" rx="1"/><circle cx="16.5" cy="16.5" r="3.6"/><line x1="19.2" y1="19.2" x2="22" y2="22"/></svg>',
    hide: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>',
};

// ── Small helpers ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
}

const store = {
    get(key, fallback) {
        try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
        catch { return fallback; }
    },
    set(key, value) {
        try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: session only */ }
    },
};

const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
const itemKey = it => (it.id != null ? `id:${it.id}` : `src:${it.src}`);
const clampRatio = r => Math.min(MAX_RATIO, Math.max(MIN_RATIO, r || 1));
const museumName = source => MUSEUMS[source] || source || '';

function toItem(r) {
    const members = Array.isArray(r.cluster_member_ids) && Number(r.cluster_size) > 1
        ? r.cluster_member_ids.map(Number) : null;
    const id = r.faiss_id ?? r.id;
    return {
        id: id != null && id !== '' ? Number(id) : null,
        src: r.image_url || '',
        url: r.URL || '',
        title: r.Title || 'Untitled',
        author: r.Author && r.Author !== 'N/A' ? r.Author : '',
        source: r.source || '',
        members,
    };
}

let toastTimer = null;
function toast(message) {
    const t = $('toast');
    t.textContent = message;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
}

// ── Collections (browser storage until accounts are wired) ─────────────────
const Collections = {
    data: null,

    load() {
        let data = store.get('glane.collections', null);
        if (!data) {
            const legacy = store.get('archivedBoards', []);
            data = {
                activeId: null,
                list: legacy.map(b => ({
                    id: String(b.id),
                    name: b.name || 'Collection',
                    items: (b.images || []).map(im => ({
                        id: null, src: im.src, url: '', title: im.title || '',
                        author: im.author || '', source: im.source || '', members: null,
                    })),
                })),
            };
        }
        if (!data.list.length) data.list.push({ id: uid(), name: 'My collection', items: [] });
        if (!data.list.some(c => c.id === data.activeId)) data.activeId = data.list[0].id;
        this.data = data;
        this.save();
    },
    save() { store.set('glane.collections', this.data); },
    get(id) { return this.data.list.find(c => c.id === id); },
    active() { return this.get(this.data.activeId); },
    setActive(id) { this.data.activeId = id; this.save(); },
    has(item) { const k = itemKey(item); return this.active().items.some(i => itemKey(i) === k); },
    toggle(item) {
        const c = this.active();
        const k = itemKey(item);
        const idx = c.items.findIndex(i => itemKey(i) === k);
        if (idx >= 0) c.items.splice(idx, 1);
        else c.items.unshift({ ...item });
        this.save();
        return idx < 0;
    },
    create() {
        const c = { id: uid(), name: `Collection ${this.data.list.length + 1}`, items: [] };
        this.data.list.unshift(c);
        this.data.activeId = c.id;
        this.save();
        return c;
    },
    rename(id, name) { const c = this.get(id); if (c && name.trim()) { c.name = name.trim(); this.save(); } },
    remove(id) {
        this.data.list = this.data.list.filter(c => c.id !== id);
        if (!this.data.list.length) this.data.list.push({ id: uid(), name: 'My collection', items: [] });
        if (!this.get(this.data.activeId)) this.data.activeId = this.data.list[0].id;
        this.save();
    },
};

// ── State ───────────────────────────────────────────────────────────────────
const query = { text: '', refs: [], negs: [] };   // refs/negs: { key, src, id?, b64? }
const hiddenIds = new Set(store.get('glane.hidden', []));

let view = 'empty';            // 'empty' | 'results' | 'collection'
let viewingId = null;          // collection shown in the main area
let renderGen = 0;             // bumps on every view change; stale async work checks it
let viewItems = [];            // items currently in the grid, in display order
const results = { request: null, offset: 0, hasMore: false, items: [], loadingGen: null };
let clearSnapshot = null;
let undoTimer = null;

// ── DOM ─────────────────────────────────────────────────────────────────────
const gallery = $('gallery');
const statusEl = $('status');
const searchInput = $('search-input');
const searchBox = $('search-box');
const chipsEl = $('chips');
const emptyState = $('empty-state');
const viewHeader = $('view-header');

function setStatus(text, loading = false) {
    statusEl.hidden = !text;
    statusEl.textContent = text || '';
    statusEl.classList.toggle('loading', loading);
}

// ── Masonry: images are placed in the currently shortest column ────────────
// Heights are tracked as sums of height/width ratios (all columns share a
// width), so placement needs no DOM measurement and keeps ranking order.
const masonry = {
    n: 0,
    cols: [],
    heights: [],
    entries: [],
    GAP: 0.05,

    columnCount() {
        const w = gallery.clientWidth || window.innerWidth;
        const target = window.innerWidth <= 860 ? 150 : 250;
        return Math.max(2, Math.min(7, Math.floor(w / target)));
    },
    reset() { this.entries = []; this.build(); },
    build() {
        this.n = this.columnCount();
        gallery.replaceChildren();
        this.cols = [];
        this.heights = new Array(this.n).fill(0);
        for (let i = 0; i < this.n; i++) this.cols.push(gallery.appendChild(el('div', 'masonry-col')));
        for (const e of this.entries) this.place(e);
    },
    add(node, ratio) {
        const e = { node, ratio };
        this.entries.push(e);
        this.place(e);
    },
    place(e) {
        let best = 0;
        for (let i = 1; i < this.n; i++) if (this.heights[i] < this.heights[best] - 1e-6) best = i;
        this.cols[best].appendChild(e.node);
        this.heights[best] += e.ratio + this.GAP;
        e.col = best;
    },
    remove(node) {
        const idx = this.entries.findIndex(e => e.node === node);
        if (idx < 0) return;
        const e = this.entries[idx];
        this.heights[e.col] -= e.ratio + this.GAP;
        this.entries.splice(idx, 1);
        node.remove();
    },
    refit() { if (this.columnCount() !== this.n) this.build(); },
};

let resizeTimer = null;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => masonry.refit(), 120);
});

function preload(src) {
    return new Promise(resolve => {
        const img = new Image();
        let settled = false;
        const done = v => { if (!settled) { settled = true; resolve(v); } };
        img.onload = () => done({ ok: true, ratio: img.naturalHeight / img.naturalWidth });
        img.onerror = () => done({ ok: false });
        setTimeout(() => done({ ok: true, ratio: 1 }), PRELOAD_TIMEOUT_MS);  // slow: place now, finish loading in place
        img.src = src;
    });
}

// Images that can't load are dropped instead of leaving an empty cell.
async function appendItems(list, gen, recordInResults) {
    const loads = list.map(it => preload(it.src));
    for (let i = 0; i < list.length; i++) {
        const r = await loads[i];
        if (gen !== renderGen) return;
        if (!r.ok) continue;
        masonry.add(createCard(list[i], clampRatio(r.ratio)), clampRatio(r.ratio));
        viewItems.push(list[i]);
        if (recordInResults) results.items.push(list[i]);
    }
}

function dropCard(card) {
    masonry.remove(card);
    viewItems = viewItems.filter(i => i !== card._item);
    results.items = results.items.filter(i => i !== card._item);
}

// ── Cards ───────────────────────────────────────────────────────────────────
function createCard(item, ratio) {
    const card = el('article', 'card');
    card._item = item;

    const frame = el('button', 'card-img');
    frame.type = 'button';
    frame.style.aspectRatio = `1 / ${ratio}`;
    frame.setAttribute('aria-label', `Open ${item.title}`);
    const img = new Image();
    img.src = item.src;
    img.alt = item.title;
    img.draggable = true;
    img.addEventListener('error', () => dropCard(card));
    img.addEventListener('dragstart', e => {
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('application/x-glane-item', JSON.stringify(item));
        e.dataTransfer.setData('text/uri-list', item.src);
    });
    frame.appendChild(img);
    frame.addEventListener('click', () => openLightbox(viewItems.indexOf(item)));
    card.appendChild(frame);

    if (item.members) {
        const badge = el('span', 'card-similar', `+${item.members.length - 1}`);
        badge.appendChild(el('span', 'card-similar-label', ' SIMILAR'));
        card.appendChild(badge);
    }

    const overlay = el('div', 'card-overlay');
    const top = el('div', 'card-top');
    if (view === 'results') {
        const hideBtn = el('button', 'card-icon-btn');
        hideBtn.type = 'button';
        hideBtn.title = 'Not interested';
        hideBtn.setAttribute('aria-label', 'Not interested');
        hideBtn.innerHTML = ICON.hide;
        hideBtn.addEventListener('click', e => { e.stopPropagation(); showHideOverlay(card); });
        top.appendChild(hideBtn);
    }

    const bottom = el('div', 'card-bottom');
    bottom.appendChild(el('div', 'card-title', item.title));
    if (item.author) bottom.appendChild(el('div', 'card-author', item.author));

    const actions = el('div', 'card-actions');
    const collectBtn = el('button', 'card-btn card-collect');
    collectBtn.type = 'button';
    collectBtn.addEventListener('click', e => { e.stopPropagation(); toggleCollect(item); });
    const visualBtn = el('button', 'card-btn');
    visualBtn.type = 'button';
    visualBtn.title = 'Start a visual search from this image';
    visualBtn.setAttribute('aria-label', 'Visual research');
    visualBtn.innerHTML = `${ICON.visual}<span class="card-btn-label">VISUAL RESEARCH</span>`;
    visualBtn.addEventListener('click', e => { e.stopPropagation(); visualResearch(item); });
    actions.append(collectBtn, visualBtn);
    bottom.appendChild(actions);

    overlay.append(top, bottom);
    card.appendChild(overlay);
    paintCollectButton(collectBtn, item);
    return card;
}

function paintCollectButton(btn, item) {
    const on = Collections.has(item);
    btn.classList.toggle('on', on);
    btn.innerHTML = `${on ? ICON.collected : ICON.collect}<span class="card-btn-label">${on ? 'COLLECTED' : 'COLLECT'}</span>`;
    btn.title = on ? `Remove from “${Collections.active().name}”` : `Add to “${Collections.active().name}”`;
    btn.setAttribute('aria-label', btn.title);
}

function syncCollectState() {
    gallery.querySelectorAll('.card').forEach(card => paintCollectButton(card.querySelector('.card-collect'), card._item));
    if (lbItem) $('lb-collect').classList.toggle('on', Collections.has(lbItem));
    renderCollectionList();
}

function toggleCollect(item) {
    const added = Collections.toggle(item);
    const name = Collections.active().name;
    toast(added ? `Added to “${name}”` : `Removed from “${name}”`);
    if (!added && view === 'collection' && viewingId === Collections.data.activeId) {
        const card = [...gallery.querySelectorAll('.card')].find(c => itemKey(c._item) === itemKey(item));
        if (card) dropCard(card);
        $('collection-count').textContent = countLabel(Collections.active().items.length);
    }
    syncCollectState();
}

function showHideOverlay(card) {
    const overlay = el('div', 'hide-overlay');
    overlay.innerHTML = `
        <div class="hide-overlay-top">
            <span class="sb-label">NOT INTERESTED</span>
            <button type="button" class="link-btn" data-action="cancel">CANCEL</button>
        </div>
        <div class="hide-options">
            <button type="button" class="hide-option" data-action="negative">
                <span class="ho-title">Use as negative</span>
                <span class="ho-desc">Steer the search away from this image</span>
            </button>
            <button type="button" class="hide-option" data-action="taste">
                <span class="ho-title">Not my taste</span>
                <span class="ho-desc">Never show it to me again</span>
            </button>
            <button type="button" class="hide-option" data-action="flag">
                <span class="ho-title">WTF flag</span>
                <span class="ho-desc">Report an irrelevant image to the team</span>
            </button>
        </div>`;
    overlay.addEventListener('click', e => {
        e.stopPropagation();
        const action = e.target.closest('[data-action]')?.dataset.action;
        if (!action) return;
        if (action === 'cancel') { overlay.remove(); return; }
        const item = card._item;
        if (action === 'negative') addReference(item, true);
        if (action === 'taste' && item.id != null) {
            hiddenIds.add(item.id);
            store.set('glane.hidden', [...hiddenIds]);
        }
        if (action === 'flag') {
            fetch(`${API}/flag`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ img_path: item.src, faiss_id: item.id }),
            }).catch(() => { });
        }
        dropCard(card);
        maybeLoadMore();
    });
    card.querySelector('.card-img').after(overlay);
}

// ── Search references (chips) ───────────────────────────────────────────────
function renderChips() {
    chipsEl.replaceChildren();
    const add = (ref, negative) => {
        const chip = el('div', negative ? 'chip negative' : 'chip');
        const img = new Image();
        img.src = ref.src;
        img.alt = negative ? 'Negative reference' : 'Reference image';
        const rm = el('button', 'chip-remove', '×');
        rm.type = 'button';
        rm.setAttribute('aria-label', 'Remove image');
        rm.addEventListener('click', () => {
            const list = negative ? query.negs : query.refs;
            list.splice(list.indexOf(ref), 1);
            renderChips();
            if (view === 'results') runSearch();
        });
        chip.append(img, rm);
        chipsEl.appendChild(chip);
    };
    query.refs.forEach(r => add(r, false));
    query.negs.forEach(r => add(r, true));
    $('negative-settings').hidden = !query.negs.length;
    updateClearButton();
}

function refFromItem(item) {
    return { key: itemKey(item), src: item.src, id: item.id };
}

function addReference(item, negative = false) {
    if (item.id == null) { toast('This image can’t be used for search'); return false; }
    const list = negative ? query.negs : query.refs;
    const other = negative ? query.refs : query.negs;
    const key = itemKey(item);
    const clash = other.findIndex(r => r.key === key);
    if (clash >= 0) other.splice(clash, 1);
    if (!list.some(r => r.key === key)) list.push(refFromItem(item));
    renderChips();
    if (negative && view === 'results') runSearch();
    return true;
}

function visualResearch(item) {
    if (addReference(item)) runSearch();
}

async function fileToRef(file) {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
    return { key: `file:${uid()}`, src: dataUrl, b64: dataUrl.split(',')[1] };
}

async function addFiles(files) {
    const images = [...files].filter(f => f.type.startsWith('image/'));
    if (!images.length) return;
    for (const f of images) {
        try { query.refs.push(await fileToRef(f)); }
        catch (err) { console.error('image file', err); toast('This image could not be read'); }
    }
    renderChips();
    runSearch();
}

// ── Search & pagination ─────────────────────────────────────────────────────
function currentSettings() {
    return {
        diversity: $('diversity-toggle').getAttribute('aria-pressed') === 'true' ? 0.5 : 0.0,
        combination_mode: document.querySelector('input[name="combination-mode"]:checked').value,
        negative_mode: document.querySelector('input[name="negative-mode"]:checked').value,
    };
}

function buildRequest() {
    const s = currentSettings();
    const body = {
        query: query.text,
        page_size: PAGE_SIZE,
        diversity: s.diversity,
        combination_mode: s.combination_mode,
    };
    const files = query.refs.filter(r => r.b64).map(r => r.b64);
    const ids = query.refs.filter(r => r.id != null).map(r => r.id);
    if (files.length) body.reference_images = files;
    if (ids.length) body.reference_ids = ids;
    if (files.length || ids.length) body.image_weight = 0.5;
    const negFiles = query.negs.filter(r => r.b64).map(r => r.b64);
    const negIds = query.negs.filter(r => r.id != null).map(r => r.id);
    if (negFiles.length) body.negative_images = negFiles;
    if (negIds.length) body.negative_ids = negIds;
    if (negFiles.length || negIds.length) body.negative_mode = s.negative_mode;
    return body;
}

function showView(name) {
    view = name;
    renderGen++;
    viewItems = [];
    masonry.reset();
    setStatus('');
    emptyState.hidden = name !== 'empty';
    viewHeader.hidden = name !== 'collection';
    if (name !== 'collection') viewingId = null;
    renderCollectionList();
    window.scrollTo(0, 0);
}

function runSearch() {
    query.text = searchInput.value.trim();
    if (!query.text && !query.refs.length) return;
    closeMenu();
    showView('results');
    results.request = buildRequest();
    results.offset = 0;
    results.hasMore = true;
    results.items = [];
    setStatus('Searching', true);
    updateUrl();
    loadPage(renderGen);
}

async function loadPage(gen) {
    if (!results.hasMore || results.loadingGen === gen) return;
    results.loadingGen = gen;
    if (results.offset > 0) setStatus('Loading', true);
    try {
        const resp = await fetch(`${API}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...results.request, offset: results.offset }),
        });
        if (!resp.ok) throw new Error(`Search failed (${resp.status})`);
        const data = await resp.json();
        if (gen !== renderGen) return;
        results.offset += data.results.length;
        results.hasMore = Boolean(data.has_more);
        const list = data.results.map(toItem).filter(it => !hiddenIds.has(it.id));
        await appendItems(list, gen, true);
        if (gen !== renderGen) return;
        setStatus(viewItems.length ? (results.hasMore ? '' : 'End of results') : 'No results');
    } catch (err) {
        console.error(err);
        if (gen === renderGen) setStatus(`${err.message} — try again`);
        results.hasMore = false;
    } finally {
        if (results.loadingGen === gen) results.loadingGen = null;
    }
    if (gen === renderGen) maybeLoadMore();
}

function maybeLoadMore() {
    if (view !== 'results' || !results.hasMore || results.loadingGen === renderGen) return;
    if ($('sentinel').getBoundingClientRect().top < window.innerHeight + 800) loadPage(renderGen);
}

new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) maybeLoadMore();
}, { rootMargin: '800px' }).observe($('sentinel'));

function backToResults() {
    if (!results.request) { showView('empty'); return; }
    showView('results');
    const gen = renderGen;
    const previous = results.items;
    results.items = [];
    appendItems(previous, gen, true).then(() => {
        if (gen === renderGen) setStatus(results.hasMore ? '' : 'End of results');
    });
}

function updateUrl() {
    const params = new URLSearchParams();
    if (query.text) params.set('q', query.text);
    const ids = query.refs.filter(r => r.id != null).map(r => r.id);
    if (ids.length) params.set('ref', ids.join(','));
    const qs = params.toString();
    history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
}

async function restoreFromUrl() {
    const params = new URLSearchParams(location.search);
    const text = params.get('q') || '';
    const ids = (params.get('ref') || '').split(',').filter(Boolean).map(Number)
        .filter(n => Number.isInteger(n) && n >= 0);
    if (!text && !ids.length) return;
    searchInput.value = text;
    if (ids.length) {
        try {
            const resp = await fetch(`${API}/cluster-members`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ faiss_ids: ids }),
            });
            const data = await resp.json();
            data.results.map(toItem).forEach(it => query.refs.push(refFromItem(it)));
            renderChips();
        } catch (err) { console.error('restore refs', err); }
    }
    runSearch();
}

// ── Clear / undo ────────────────────────────────────────────────────────────
function updateClearButton() {
    const has = searchInput.value.trim() || query.refs.length || query.negs.length;
    $('clear-btn').hidden = !has;
}

function clearAll() {
    clearSnapshot = { text: searchInput.value, refs: [...query.refs], negs: [...query.negs] };
    searchInput.value = '';
    query.refs = [];
    query.negs = [];
    renderChips();
    $('undo-btn').hidden = false;
    clearTimeout(undoTimer);
    undoTimer = setTimeout(() => { $('undo-btn').hidden = true; clearSnapshot = null; }, 5000);
    searchInput.focus();
}

function undoClear() {
    if (!clearSnapshot) return;
    searchInput.value = clearSnapshot.text;
    query.refs = clearSnapshot.refs;
    query.negs = clearSnapshot.negs;
    clearSnapshot = null;
    $('undo-btn').hidden = true;
    renderChips();
}

// ── Collections: sidebar list and collection view ───────────────────────────
const countLabel = n => `${n} image${n === 1 ? '' : 's'}`;

function renderCollectionList() {
    const list = $('collection-list');
    list.replaceChildren();
    for (const c of Collections.data.list) {
        const li = el('li');
        const btn = el('button', 'collection-item');
        btn.type = 'button';
        btn.classList.toggle('active', c.id === Collections.data.activeId);
        btn.classList.toggle('viewing', view === 'collection' && c.id === viewingId);
        const cover = el('span', 'collection-cover');
        if (c.items[0]) cover.style.backgroundImage = `url("${encodeURI(c.items[0].src)}")`;
        const text = el('span', 'collection-text');
        text.append(el('span', 'collection-title', c.name), el('span', 'collection-meta', countLabel(c.items.length)));
        btn.append(cover, text);
        btn.addEventListener('click', () => openCollection(c.id));
        li.appendChild(btn);
        list.appendChild(li);
    }
}

function openCollection(id) {
    const c = Collections.get(id);
    if (!c) return;
    Collections.setActive(id);
    closeMenu();
    viewingId = id;
    showView('collection');
    $('collection-name').value = c.name;
    $('collection-count').textContent = countLabel(c.items.length);
    $('back-btn').hidden = !results.request;
    $('collection-search-btn').disabled = !c.items.some(i => i.id != null);
    syncCollectState();
    if (!c.items.length) {
        setStatus('Empty for now. Use COLLECT on any image to add it here.');
        return;
    }
    const gen = renderGen;
    appendItems(c.items, gen, false).then(() => {
        if (gen === renderGen && !viewItems.length) setStatus('These images can no longer be displayed.');
    });
}

$('new-collection-btn').addEventListener('click', () => {
    const c = Collections.create();
    openCollection(c.id);
    const name = $('collection-name');
    name.focus();
    name.select();
});

$('collection-name').addEventListener('change', e => {
    Collections.rename(viewingId, e.target.value);
    e.target.value = Collections.get(viewingId).name;
    syncCollectState();
});
$('collection-name').addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });

$('collection-search-btn').addEventListener('click', () => {
    const c = Collections.get(viewingId);
    const usable = c.items.filter(i => i.id != null).slice(0, MAX_COLLECTION_REFS);
    if (!usable.length) return;
    query.refs = usable.map(refFromItem);
    query.negs = [];
    searchInput.value = '';
    renderChips();
    runSearch();
    if (c.items.length > usable.length) toast(`Searching with the first ${usable.length} images`);
});

$('collection-delete-btn').addEventListener('click', () => {
    const c = Collections.get(viewingId);
    if (!c || !confirm(`Delete “${c.name}”? Its ${countLabel(c.items.length)} will be removed.`)) return;
    Collections.remove(c.id);
    backToResults();
});

$('back-btn').addEventListener('click', backToResults);

// ── Lightbox ────────────────────────────────────────────────────────────────
const lightbox = $('lightbox');
const lbImg = $('lb-img');
let lbIndex = -1;
let lbItem = null;
let lbSimilarGen = 0;

function openLightbox(index) {
    if (index < 0) return;
    lbIndex = index;
    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
    showInLightbox(viewItems[index], true);
    $('lb-close').focus();
}

function closeLightbox() {
    lightbox.hidden = true;
    document.body.style.overflow = '';
    lbImg.removeAttribute('src');
    lbItem = null;
}

function showInLightbox(item, withSimilar) {
    lbItem = item;
    lbImg.style.opacity = '0';
    lbImg.onload = () => { lbImg.style.opacity = '1'; };
    lbImg.src = item.src;
    lbImg.alt = item.title;
    $('lb-title').textContent = item.title;
    $('lb-author').textContent = item.author;
    const museum = $('lb-museum');
    museum.textContent = museumName(item.source);
    if (item.url) museum.href = item.url; else museum.removeAttribute('href');
    museum.title = item.url ? 'See this artwork on the museum website' : '';
    $('lb-download').href = item.src;
    $('lb-collect').classList.toggle('on', Collections.has(item));
    $('lb-prev').disabled = lbIndex <= 0;
    $('lb-next').disabled = lbIndex >= viewItems.length - 1;
    if (withSimilar) loadSimilar(item);
}

async function loadSimilar(item) {
    const gen = ++lbSimilarGen;
    const box = $('lb-similar');
    const strip = $('lb-similar-strip');
    strip.replaceChildren();
    box.hidden = !item.members;
    if (!item.members) return;
    $('lb-similar-label').textContent = `${item.members.length - 1} SIMILAR`;
    try {
        const resp = await fetch(`${API}/cluster-members`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ faiss_ids: item.members }),
        });
        if (!resp.ok) throw new Error(`cluster-members ${resp.status}`);
        const data = await resp.json();
        if (gen !== lbSimilarGen) return;
        for (const member of data.results.map(toItem)) {
            const thumb = el('button', 'lb-thumb');
            thumb.type = 'button';
            thumb.title = member.title;
            thumb.classList.toggle('active', itemKey(member) === itemKey(item));
            const img = new Image();
            img.src = member.src;
            img.alt = member.title;
            img.loading = 'lazy';
            img.addEventListener('error', () => thumb.remove());
            thumb.appendChild(img);
            thumb.addEventListener('click', () => {
                strip.querySelectorAll('.lb-thumb').forEach(t => t.classList.remove('active'));
                thumb.classList.add('active');
                showInLightbox(member, false);
            });
            strip.appendChild(thumb);
        }
    } catch (err) {
        console.error(err);
        box.hidden = true;
    }
}

function stepLightbox(delta) {
    const next = lbIndex + delta;
    if (next < 0 || next >= viewItems.length) return;
    lbIndex = next;
    showInLightbox(viewItems[next], true);
}

$('lb-close').addEventListener('click', closeLightbox);
$('lb-prev').addEventListener('click', () => stepLightbox(-1));
$('lb-next').addEventListener('click', () => stepLightbox(1));
$('lb-collect').addEventListener('click', () => { if (lbItem) toggleCollect(lbItem); });
$('lb-visual').addEventListener('click', () => {
    if (!lbItem) return;
    const item = lbItem;
    closeLightbox();
    visualResearch(item);
});
lightbox.addEventListener('click', e => { if (e.target === lightbox || e.target.classList.contains('lb-stage')) closeLightbox(); });

document.addEventListener('keydown', e => {
    if (lightbox.hidden) {
        if (e.key === 'Escape' && document.body.classList.contains('menu-open')) closeMenu();
        return;
    }
    if (e.key === 'Escape') closeLightbox();
    if (e.key === 'ArrowLeft') stepLightbox(-1);
    if (e.key === 'ArrowRight') stepLightbox(1);
});

let touchX = null;
lightbox.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, { passive: true });
lightbox.addEventListener('touchend', e => {
    if (touchX == null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    touchX = null;
    if (Math.abs(dx) > 60) stepLightbox(dx < 0 ? 1 : -1);
});

// ── Search form, drag & drop, files ─────────────────────────────────────────
$('search-form').addEventListener('submit', e => { e.preventDefault(); runSearch(); });
searchInput.addEventListener('input', updateClearButton);
$('clear-btn').addEventListener('click', clearAll);
$('undo-btn').addEventListener('click', undoClear);
$('file-input').addEventListener('change', e => { addFiles(e.target.files); e.target.value = ''; });

document.querySelectorAll('.suggestion').forEach(btn => btn.addEventListener('click', () => {
    searchInput.value = btn.textContent;
    runSearch();
}));

const dropZone = $('search-form');
dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    searchBox.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', e => {
    if (!dropZone.contains(e.relatedTarget)) searchBox.classList.remove('drag-over');
});
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    searchBox.classList.remove('drag-over');
    const raw = e.dataTransfer.getData('application/x-glane-item');
    if (raw) {
        addReference(JSON.parse(raw));
        searchInput.focus();
        return;
    }
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});

// ── Settings ────────────────────────────────────────────────────────────────
$('settings-toggle').addEventListener('click', e => {
    const open = e.currentTarget.getAttribute('aria-expanded') !== 'true';
    e.currentTarget.setAttribute('aria-expanded', String(open));
    $('settings-panel').hidden = !open;
});
$('diversity-toggle').addEventListener('click', e => {
    const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    e.currentTarget.setAttribute('aria-pressed', String(on));
    if (view === 'results') runSearch();
});
document.querySelectorAll('input[name="combination-mode"], input[name="negative-mode"]').forEach(r =>
    r.addEventListener('change', () => { if (view === 'results') runSearch(); }));

// ── Mobile menu ─────────────────────────────────────────────────────────────
function closeMenu() {
    document.body.classList.remove('menu-open');
    $('menu-btn').setAttribute('aria-expanded', 'false');
}
$('menu-btn').addEventListener('click', () => {
    const open = !document.body.classList.contains('menu-open');
    document.body.classList.toggle('menu-open', open);
    $('menu-btn').setAttribute('aria-expanded', String(open));
});

// On narrow screens the sidebar is a top bar whose height varies with chips.
new ResizeObserver(() => {
    if (document.body.classList.contains('menu-open')) return;
    document.documentElement.style.setProperty('--topbar-h', `${$('sidebar').offsetHeight}px`);
}).observe($('sidebar'));

// ── Account (magic link; accounts are wired in lot 2) ───────────────────────
$('auth-form').addEventListener('submit', async e => {
    e.preventDefault();
    const email = $('auth-email').value.trim();
    const btn = $('auth-btn');
    const statusNote = $('auth-status');
    btn.disabled = true;
    try {
        const resp = await fetch(`${API}/auth/request`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email }),
        });
        if (!resp.ok) throw new Error(String(resp.status));
        statusNote.textContent = `Link sent to ${email}. Check your inbox.`;
    } catch (err) {
        console.error(err);
        statusNote.textContent = 'Could not send the link. Please try again.';
    } finally {
        statusNote.hidden = false;
        btn.disabled = false;
    }
});

// ── Init ────────────────────────────────────────────────────────────────────
Collections.load();
masonry.build();
renderCollectionList();
renderChips();
restoreFromUrl();
