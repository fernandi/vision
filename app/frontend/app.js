'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
const API = '';
const PAGE_SIZE = 24;
const RECOS_SIZE = 30;
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
    cover: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/></svg>',
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
const countLabel = n => `${n} image${n === 1 ? '' : 's'}`;
const cssUrl = src => `url("${encodeURI(src)}")`;

function toItem(r) {
    const members = Array.isArray(r.cluster_member_ids) && Number(r.cluster_size) > 1
        ? r.cluster_member_ids.map(Number) : null;
    const id = r.faiss_id ?? r.id;
    return {
        id: id != null && id !== '' ? Number(id) : null,
        src: r.thumb_url || r.image_url || '',   // grid
        hd: r.image_url || '',                    // full page
        orig: r.original_url || '',               // museum file, used if a copy is missing
        url: r.URL || '',
        title: r.Title || 'Untitled',
        author: r.Author && r.Author !== 'N/A' ? r.Author : '',
        source: r.source || '',
        members,
    };
}

async function postJSON(path, body) {
    const resp = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`Request failed (${resp.status})`);
    return resp.json();
}

let toastTimer = null;
function toast(message, action = null) {
    const t = $('toast');
    const btn = $('toast-action');
    $('toast-text').textContent = message;
    btn.hidden = !action;
    btn.onclick = null;
    if (action) {
        btn.textContent = action.label;
        btn.onclick = () => { t.classList.remove('show'); action.run(); };
    }
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), action ? 5000 : 2400);
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
        for (const c of data.list) {
            c.createdAt ??= createdFromId(c.id);
            c.updatedAt ??= c.createdAt;
        }
        if (!data.v2) {
            // v0.2 no longer creates a default collection: drop the old empty one.
            data.list = data.list.filter(c => !(c.name === 'My collection' && !c.items.length));
            data.v2 = true;
        }
        data.list.sort((a, b) => a.createdAt - b.createdAt);
        if (!data.list.some(c => c.id === data.activeId)) data.activeId = data.list.at(-1)?.id ?? null;
        this.data = data;
        this.save();
    },
    save() { store.set('glane.collections', this.data); },
    // Every change goes through touch(): stamps the collection and queues it for the account.
    touch(id) {
        const c = this.get(id);
        if (c) { c.updatedAt = Math.max(Date.now(), (c.updatedAt || 0) + 1); Sync.markDirty(id); }
        this.save();
    },
    all() { return this.data.list; },
    get(id) { return this.data.list.find(c => c.id === id); },
    active() { return this.get(this.data.activeId) || null; },
    setActive(id) { this.data.activeId = id; this.save(); },
    ensureTarget() {
        if (!this.active()) this.setActive(this.create().id);
        return this.active();
    },
    cover(c) {
        return (c.coverKey && c.items.find(i => itemKey(i) === c.coverKey)) || c.items[0] || null;
    },
    setCover(id, item) { this.get(id).coverKey = itemKey(item); this.touch(id); },
    contains(id, item) {
        const c = this.get(id);
        const k = itemKey(item);
        return Boolean(c) && c.items.some(i => itemKey(i) === k);
    },
    add(id, item) {
        if (this.contains(id, item)) return false;
        this.get(id).items.unshift({ ...item });
        this.touch(id);
        return true;
    },
    removeItem(id, item) {
        const c = this.get(id);
        const k = itemKey(item);
        const idx = c.items.findIndex(i => itemKey(i) === k);
        if (idx < 0) return -1;
        c.items.splice(idx, 1);
        this.touch(id);
        return idx;
    },
    insertAt(id, item, idx) {
        const c = this.get(id);
        if (!this.contains(id, item)) c.items.splice(Math.max(0, idx), 0, { ...item });
        this.touch(id);
    },
    create(name) {
        const now = Date.now();
        const c = { id: uid(), name: name || `Collection ${this.data.list.length + 1}`, items: [], createdAt: now, updatedAt: now, coverKey: null };
        this.data.list.push(c);
        this.touch(c.id);
        return c;
    },
    rename(id, name) { const c = this.get(id); if (c && name.trim()) { c.name = name.trim(); this.touch(id); } },
    remove(id) {
        this.data.list = this.data.list.filter(c => c.id !== id);
        this.fixActive();
        Sync.markDeleted(id);
        this.save();
    },
    fixActive() {
        if (!this.get(this.data.activeId)) this.data.activeId = this.data.list.at(-1)?.id ?? null;
    },
    replaceAll(list) {
        this.data.list = list.sort((a, b) => a.createdAt - b.createdAt);
        this.fixActive();
        this.save();
    },
};

// Ids are Date.now() (legacy) or uid(): base-36 timestamp + 4 random chars.
function createdFromId(id) {
    return /^\d+$/.test(id) ? Number(id) : parseInt(String(id).slice(0, -4), 36) || 0;
}

// ── State ───────────────────────────────────────────────────────────────────
const query = { text: '', refs: [], negs: [] };   // refs/negs: { key, src, id?, b64? }
const hiddenIds = new Set(store.get('glane.hidden', []));

let view = 'empty';            // 'empty' | 'results' | 'collection'
let viewingId = null;          // collection shown in the main area
let renderGen = 0;             // bumps on every view change; stale async work checks it
const results = { request: null, offset: 0, hasMore: false, items: [], loadingGen: null };
const recosCache = new Map();  // collection signature → suggested items
let clearSnapshot = null;
let undoTimer = null;

const statusEl = $('status');
const searchInput = $('search-input');
const searchBox = $('search-box');
const chipsEl = $('chips');

function setStatus(node, text, loading = false) {
    node.hidden = !text;
    node.textContent = text || '';
    node.classList.toggle('loading', loading);
}

// ── Masonry grid: each image goes to the currently shortest column ─────────
// Heights are tracked as sums of height/width ratios (all columns share a
// width), so placement needs no DOM measurement and keeps ranking order.
function makeGrid(container) {
    return {
        container,
        n: 0,
        cols: [],
        heights: [],
        entries: [],
        items: [],            // items shown, in order: lightbox navigation
        hasSkeleton: false,
        GAP: 0.05,

        columnCount() {
            const w = this.container.clientWidth || window.innerWidth;
            const target = window.innerWidth <= 860 ? 150 : 250;
            return Math.max(2, Math.min(7, Math.floor(w / target)));
        },
        reset() {
            this.entries = [];
            this.items = [];
            this.hasSkeleton = false;
            this.build();
        },
        build() {
            this.n = this.columnCount();
            this.container.replaceChildren();
            this.cols = [];
            this.heights = new Array(this.n).fill(0);
            for (let i = 0; i < this.n; i++) this.cols.push(this.container.appendChild(el('div', 'masonry-col')));
            for (const e of this.entries) this.place(e);
        },
        skeleton(count) {
            this.reset();
            for (let i = 0; i < count; i++) {
                const ratio = 0.7 + ((i * 37) % 70) / 100;
                const node = el('div', 'skeleton');
                node.style.aspectRatio = `1 / ${ratio}`;
                node.style.animationDelay = `0s, ${i * 40}ms`;
                this.place({ node, ratio });
            }
            this.hasSkeleton = true;
        },
        add(node, ratio, item) {
            if (this.hasSkeleton) this.reset();
            const e = { node, ratio };
            this.entries.push(e);
            this.items.push(item);
            this.place(e);
        },
        place(e) {
            let best = 0;
            for (let i = 1; i < this.n; i++) if (this.heights[i] < this.heights[best] - 1e-6) best = i;
            this.cols[best].appendChild(e.node);
            this.heights[best] += e.ratio + this.GAP;
            e.col = best;
        },
        remove(node, animate = true) {
            const idx = this.entries.findIndex(e => e.node === node);
            if (idx < 0) return;
            const e = this.entries[idx];
            this.heights[e.col] -= e.ratio + this.GAP;
            this.entries.splice(idx, 1);
            this.items = this.items.filter(i => i !== node._item);
            if (!animate) { node.remove(); return; }
            node.classList.add('leaving');
            node.addEventListener('animationend', () => node.remove(), { once: true });
        },
        cardFor(item) {
            const k = itemKey(item);
            return this.entries.map(e => e.node).find(n => n._item && itemKey(n._item) === k);
        },
        refit() { if (!this.hasSkeleton && this.columnCount() !== this.n) this.build(); },
    };
}

const mainGrid = makeGrid($('gallery'));
const recosGrid = makeGrid($('recos-gallery'));

let resizeTimer = null;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { mainGrid.refit(); recosGrid.refit(); }, 120);
});

function preloadOne(src) {
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

// Mirrored thumbnail first; if it is missing, the museum's own file.
async function preload(item) {
    const r = await preloadOne(item.src);
    if (r.ok || !item.orig || item.orig === item.src) return r;
    const fallback = await preloadOne(item.orig);
    if (fallback.ok) item.src = item.orig;
    return fallback;
}

const fullImage = item => item.hd || item.src;

// Images that can't load are dropped instead of leaving an empty cell.
async function fillGrid(grid, list, gen, mode, onPlaced) {
    const loads = list.map(it => preload(it));
    let batchIndex = 0;
    for (let i = 0; i < list.length; i++) {
        const r = await loads[i];
        if (gen !== renderGen) return false;
        if (!r.ok) continue;
        const ratio = clampRatio(r.ratio);
        grid.add(createCard(list[i], ratio, grid, mode, batchIndex++), ratio, list[i]);
        if (onPlaced) onPlaced(list[i]);
    }
    if (grid.hasSkeleton) grid.reset();
    return true;
}

// ── Cards ───────────────────────────────────────────────────────────────────
// mode 'results': COLLECT targets the active collection.
// mode 'collection': the card belongs to the collection being viewed.
function createCard(item, ratio, grid, mode, batchIndex = 0) {
    const card = el('article', 'card');
    card._item = item;
    card._mode = mode;
    card.style.setProperty('--i', batchIndex);

    const frame = el('button', 'card-img');
    frame.type = 'button';
    frame.style.aspectRatio = `1 / ${ratio}`;
    frame.setAttribute('aria-label', `Open ${item.title}`);
    const img = new Image();
    img.src = item.src;
    img.alt = item.title;
    img.draggable = true;
    img.addEventListener('error', () => {
        if (item.orig && img.src !== new URL(item.orig, location.href).href) img.src = item.orig;
        else grid.remove(card, false);
    });
    img.addEventListener('dragstart', e => {
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('application/x-glane-item', JSON.stringify(item));
        e.dataTransfer.setData('text/uri-list', item.src);
        document.body.classList.add('is-dragging');
    });
    img.addEventListener('dragend', () => document.body.classList.remove('is-dragging'));
    frame.appendChild(img);
    frame.addEventListener('click', () => openLightbox(grid, item));
    card.appendChild(frame);

    if (item.members) {
        const badge = el('span', 'card-similar', `+${item.members.length - 1}`);
        badge.appendChild(el('span', 'card-similar-label', ' SIMILAR'));
        card.appendChild(badge);
    }

    const overlay = el('div', 'card-overlay');
    const top = el('div', 'card-top');
    if (grid === mainGrid && view === 'results') {
        const hideBtn = el('button', 'card-icon-btn');
        hideBtn.type = 'button';
        hideBtn.title = 'Not interested';
        hideBtn.setAttribute('aria-label', 'Not interested');
        hideBtn.innerHTML = ICON.hide;
        hideBtn.addEventListener('click', e => { e.stopPropagation(); showHideOverlay(card, grid); });
        top.appendChild(hideBtn);
    }
    if (mode === 'collection') {
        const coverBtn = el('button', 'card-icon-btn card-cover');
        coverBtn.type = 'button';
        coverBtn.innerHTML = ICON.cover;
        coverBtn.addEventListener('click', e => { e.stopPropagation(); setCover(item); });
        top.appendChild(coverBtn);
    }

    const bottom = el('div', 'card-bottom');
    bottom.appendChild(el('div', 'card-title', item.title));
    if (item.author) bottom.appendChild(el('div', 'card-author', item.author));

    const actions = el('div', 'card-actions');
    const split = el('span', 'card-split');
    const collectBtn = el('button', 'card-btn card-collect');
    collectBtn.type = 'button';
    collectBtn.addEventListener('click', e => { e.stopPropagation(); onCollectClick(card); });
    const chevron = el('button', 'card-btn card-chevron', '▾');
    chevron.type = 'button';
    chevron.title = 'Choose a collection';
    chevron.setAttribute('aria-label', 'Choose a collection');
    chevron.setAttribute('aria-haspopup', 'menu');
    chevron.addEventListener('click', e => { e.stopPropagation(); openPicker(chevron, item, img); });
    split.append(collectBtn, chevron);

    const visualBtn = el('button', 'card-btn');
    visualBtn.type = 'button';
    visualBtn.title = 'Start a visual search from this image';
    visualBtn.setAttribute('aria-label', 'Visual research');
    visualBtn.innerHTML = `${ICON.visual}<span class="card-btn-label">VISUAL RESEARCH</span>`;
    visualBtn.addEventListener('click', e => { e.stopPropagation(); visualResearch(item); });
    actions.append(split, visualBtn);
    bottom.appendChild(actions);

    overlay.append(top, bottom);
    card.appendChild(overlay);
    paintCollectButton(card);
    return card;
}

// The collection a card's COLLECT button refers to.
function cardCollectionId(card) {
    return card._mode === 'collection' ? viewingId : Collections.data.activeId;
}

function paintCollectButton(card) {
    const btn = card.querySelector('.card-collect');
    const cid = cardCollectionId(card);
    const on = Collections.contains(cid, card._item);
    const target = Collections.get(cid);
    btn.classList.toggle('on', on);
    btn.innerHTML = `${on ? ICON.collected : ICON.collect}<span class="card-btn-label">${on ? 'COLLECTED' : 'COLLECT'}</span>`;
    btn.title = on ? `In “${target.name}”: remove or move`
        : target ? `Add to “${target.name}”` : 'Add to a new collection';
    btn.setAttribute('aria-label', btn.title);

    const coverBtn = card.querySelector('.card-cover');
    if (coverBtn) {
        const c = Collections.get(viewingId);
        const isCover = Boolean(c) && itemKey(Collections.cover(c) || {}) === itemKey(card._item);
        coverBtn.classList.toggle('on', isCover);
        coverBtn.title = isCover ? 'Collection cover' : 'Use as collection cover';
        coverBtn.setAttribute('aria-label', coverBtn.title);
    }
}

function repaintAll() {
    document.querySelectorAll('.card').forEach(card => { if (card.querySelector('.card-collect')) paintCollectButton(card); });
    if (lbItem) paintLightboxActions();
    renderSidebarCollections();
    if (view === 'collection') $('collection-count').textContent = countLabel(Collections.get(viewingId).items.length);
}

function setCover(item) {
    const c = Collections.get(viewingId);
    if (!c || !Collections.contains(c.id, item)) return;
    Collections.setCover(c.id, item);
    repaintAll();
    bumpRow(c.id);
    toast(`Cover of “${c.name}” updated`);
}

function onCollectClick(card) {
    const cid = cardCollectionId(card) ?? Collections.ensureTarget().id;
    const item = card._item;
    if (Collections.contains(cid, item)) { openItemDialog(item, cid); return; }
    collectInto(cid, item, card.querySelector('.card-img img'));
    const btn = card.querySelector('.card-collect');
    btn.classList.remove('pop');
    void btn.offsetWidth;
    btn.classList.add('pop');
}

function collectInto(cid, item, fromImg) {
    if (!Collections.add(cid, item)) return;
    const c = Collections.get(cid);
    toast(`Added to “${c.name}”`, {
        label: 'UNDO',
        run: () => { Collections.removeItem(cid, item); afterRemoval(item, cid); repaintAll(); },
    });
    repaintAll();
    flyToCollection(cid, fromImg);
    if (view === 'collection' && viewingId === cid) {
        const recoCard = recosGrid.cardFor(item);
        if (recoCard) recosGrid.remove(recoCard);
        const ratio = recoCard ? parseFloat(recoCard.querySelector('.card-img').style.aspectRatio.split('/')[1]) : 1;
        mainGrid.add(createCard(item, clampRatio(ratio), mainGrid, 'collection'), clampRatio(ratio), item);
        setStatus(statusEl, '');
        repaintAll();
    }
}

function rowFor(cid) {
    return document.querySelector(`.collection-row[data-id="${CSS.escape(cid)}"]`);
}

function bumpRow(cid) {
    const row = rowFor(cid);
    if (!row) return;
    row.classList.remove('bump');
    void row.offsetWidth;
    row.classList.add('bump');
}

// Keeps the grid of the collection being viewed in sync after a removal.
function afterRemoval(item, cid) {
    if (view === 'collection' && viewingId === cid) {
        const card = mainGrid.cardFor(item);
        if (card) mainGrid.remove(card);
        if (!Collections.get(cid).items.length) setStatus(statusEl, 'Empty for now. Use COLLECT on any image to add it here.');
    }
}

function flyToCollection(cid, fromImg) {
    const dest = rowFor(cid)?.querySelector('.collection-cover');
    const bumpTarget = () => bumpRow(cid);
    const b = dest?.getBoundingClientRect();
    if (!b || !fromImg || b.height === 0 || b.bottom < 0 || b.top > window.innerHeight) {
        bumpTarget();
        return;
    }
    const a = fromImg.getBoundingClientRect();
    const ghost = fromImg.cloneNode();
    ghost.className = 'fly-ghost';
    Object.assign(ghost.style, { left: `${a.left}px`, top: `${a.top}px`, width: `${a.width}px`, height: `${a.height}px` });
    document.body.appendChild(ghost);
    const dx = b.left + b.width / 2 - (a.left + a.width / 2);
    const dy = b.top + b.height / 2 - (a.top + a.height / 2);
    const s = Math.max(b.width / a.width, 0.08);
    ghost.animate([
        { transform: 'translate(0, 0) scale(1)', opacity: 1 },
        { transform: `translate(${dx * 0.55}px, ${dy * 0.35 - 40}px) scale(${(1 + s) / 2})`, opacity: 0.95, offset: 0.55 },
        { transform: `translate(${dx}px, ${dy}px) scale(${s})`, opacity: 0.25 },
    ], { duration: 720, easing: 'cubic-bezier(0.55, 0, 0.25, 1)' }).onfinish = () => {
        ghost.remove();
        bumpTarget();
    };
}

function showHideOverlay(card, grid) {
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
            postJSON('/flag', { img_path: item.src, faiss_id: item.id }).catch(() => { });
        }
        grid.remove(card);
        maybeLoadMore();
    });
    card.querySelector('.card-img').after(overlay);
}

// ── Collection picker (popover) ─────────────────────────────────────────────
const picker = $('picker');
let pickerAnchor = null;

function openPicker(anchor, item, fromImg) {
    if (!picker.hidden && pickerAnchor === anchor) { closePicker(); return; }
    closePicker(true);
    pickerAnchor = anchor;
    anchor.setAttribute('aria-expanded', 'true');
    picker.replaceChildren(el('div', 'picker-label', item ? 'COLLECT INTO' : 'NEW IMAGES GO INTO'));

    const row = (c, i) => {
        const btn = el('button', 'picker-item');
        btn.type = 'button';
        btn.setAttribute('role', 'menuitem');
        btn.style.setProperty('--i', i);
        btn.classList.toggle('is-target', c.id === Collections.data.activeId);
        const cover = el('span', 'picker-cover');
        if (c.items[0]) cover.style.backgroundImage = cssUrl(c.items[0].src);
        const text = el('span', 'picker-text');
        text.append(el('span', 'picker-name', c.name), el('span', 'picker-meta', countLabel(c.items.length)));
        btn.append(cover, text);
        if (item && Collections.contains(c.id, item)) btn.appendChild(el('span', 'picker-mark', '✓'));
        btn.addEventListener('click', () => {
            closePicker();
            setTarget(c.id, !item);
            if (item) {
                if (Collections.contains(c.id, item)) toast(`Already in “${c.name}”`);
                else collectInto(c.id, item, fromImg);
            }
        });
        return btn;
    };
    Collections.all().forEach((c, i) => picker.appendChild(row(c, i)));

    const create = el('button', 'picker-item picker-new');
    create.type = 'button';
    create.setAttribute('role', 'menuitem');
    create.style.setProperty('--i', Collections.all().length);
    const plus = el('span', 'picker-cover', '+');
    const text = el('span', 'picker-text');
    text.append(el('span', 'picker-name', 'New collection'));
    create.append(plus, text);
    create.addEventListener('click', () => {
        closePicker();
        const c = Collections.create();
        setTarget(c.id, false);
        if (item) collectInto(c.id, item, fromImg);
        else toast(`New images now go into “${c.name}”`);
    });
    picker.appendChild(create);

    picker.hidden = false;
    const r = anchor.getBoundingClientRect();
    const w = picker.offsetWidth;
    const h = picker.offsetHeight;
    const below = r.bottom + 6 + h < window.innerHeight;
    const left = Math.min(Math.max(8, r.right - w), window.innerWidth - w - 8);
    const top = below ? r.bottom + 6 : Math.max(8, r.top - h - 6);
    picker.style.left = `${left}px`;
    picker.style.top = `${top}px`;
    picker.style.setProperty('--origin', `${below ? 'top' : 'bottom'} ${r.right - left > w / 2 ? 'right' : 'left'}`);
    picker.querySelector('.picker-item').focus({ preventScroll: true });
}

function closePicker(immediate = false) {
    if (picker.hidden) return;
    pickerAnchor?.setAttribute('aria-expanded', 'false');
    pickerAnchor = null;
    if (immediate) { picker.hidden = true; return; }
    picker.classList.add('closing');
    setTimeout(() => { picker.hidden = true; picker.classList.remove('closing'); }, 170);
}

document.addEventListener('click', e => {
    if (!picker.hidden && !picker.contains(e.target) && e.target !== pickerAnchor) closePicker();
});
window.addEventListener('scroll', () => closePicker(), { passive: true });

function setTarget(id, announce) {
    if (Collections.data.activeId === id) return;
    Collections.setActive(id);
    if (announce) toast(`New images now go into “${Collections.get(id).name}”`);
    repaintAll();
}

// ── Dialogs (always ask before an image leaves a collection) ───────────────
const dialog = $('dialog');

function openDialog(build) {
    dialog.replaceChildren();
    dialog.classList.remove('closing');
    build(dialog);
    dialog.showModal();
}

function closeDialog() {
    if (!dialog.open) return;
    dialog.classList.add('closing');
    dialog.addEventListener('animationend', () => { dialog.close(); dialog.classList.remove('closing'); }, { once: true });
}

dialog.addEventListener('click', e => { if (e.target === dialog) closeDialog(); });
dialog.addEventListener('cancel', e => { e.preventDefault(); closeDialog(); });

function dialogButton(label, className, onClick) {
    const b = el('button', className, label);
    b.type = 'button';
    b.addEventListener('click', onClick);
    return b;
}

function openItemDialog(item, fromId) {
    const from = Collections.get(fromId);
    openDialog(d => {
        const head = el('div', 'dialog-head');
        const thumb = new Image();
        thumb.className = 'dialog-thumb';
        thumb.src = item.src;
        thumb.alt = '';
        const text = el('div');
        text.append(el('div', 'dialog-title', item.title), el('div', 'dialog-text', `In “${from.name}”. What do you want to do?`));
        head.append(thumb, text);
        d.appendChild(head);

        const others = Collections.all().filter(c => c.id !== fromId);
        const section = el('div', 'dialog-section');
        section.appendChild(el('div', 'dialog-label', 'MOVE TO'));
        const list = el('div', 'dialog-list');
        others.forEach((c, i) => {
            const b = el('button', 'picker-item');
            b.type = 'button';
            b.style.setProperty('--i', i);
            const cover = el('span', 'picker-cover');
            if (c.items[0]) cover.style.backgroundImage = cssUrl(c.items[0].src);
            const t = el('span', 'picker-text');
            t.append(el('span', 'picker-name', c.name), el('span', 'picker-meta', countLabel(c.items.length)));
            b.append(cover, t);
            if (Collections.contains(c.id, item)) b.appendChild(el('span', 'picker-mark', 'already in'));
            b.addEventListener('click', () => { moveItem(item, fromId, c.id); closeDialog(); });
            list.appendChild(b);
        });
        const create = el('button', 'picker-item picker-new');
        create.type = 'button';
        create.style.setProperty('--i', others.length);
        const t = el('span', 'picker-text');
        t.append(el('span', 'picker-name', 'New collection'));
        create.append(el('span', 'picker-cover', '+'), t);
        create.addEventListener('click', () => { const c = Collections.create(); moveItem(item, fromId, c.id); closeDialog(); });
        list.appendChild(create);
        section.appendChild(list);
        d.appendChild(section);

        const foot = el('div', 'dialog-foot');
        foot.append(
            dialogButton('CANCEL', 'link-btn', closeDialog),
            dialogButton('REMOVE FROM COLLECTION', 'pill-btn danger-btn', () => { removeItem(item, fromId); closeDialog(); }),
        );
        d.appendChild(foot);
    });
}

function removeItem(item, cid) {
    const idx = Collections.removeItem(cid, item);
    afterRemoval(item, cid);
    repaintAll();
    toast(`Removed from “${Collections.get(cid).name}”`, {
        label: 'UNDO',
        run: () => { Collections.insertAt(cid, item, idx); restoreInView(item, cid); repaintAll(); },
    });
}

function moveItem(item, fromId, toId) {
    const idx = Collections.removeItem(fromId, item);
    const added = Collections.add(toId, item);
    afterRemoval(item, fromId);
    repaintAll();
    toast(`Moved to “${Collections.get(toId).name}”`, {
        label: 'UNDO',
        run: () => {
            if (added) Collections.removeItem(toId, item);
            Collections.insertAt(fromId, item, idx);
            restoreInView(item, fromId);
            repaintAll();
        },
    });
}

function restoreInView(item, cid) {
    if (view === 'collection' && viewingId === cid && !mainGrid.cardFor(item)) {
        setStatus(statusEl, '');
        fillGrid(mainGrid, [item], renderGen, 'collection');
    }
}

function confirmDialog({ title, text, confirm, danger = false }) {
    return new Promise(resolve => {
        openDialog(d => {
            d.append(el('div', 'dialog-title', title), el('div', 'dialog-text', text));
            const foot = el('div', 'dialog-foot');
            foot.append(
                dialogButton('CANCEL', 'link-btn', () => { closeDialog(); resolve(false); }),
                dialogButton(confirm, danger ? 'pill-btn danger-btn' : 'pill-btn', () => { closeDialog(); resolve(true); }),
            );
            d.appendChild(foot);
        });
        dialog.addEventListener('close', () => resolve(false), { once: true });
    });
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
    mainGrid.reset();
    recosGrid.reset();
    setStatus(statusEl, '');
    $('empty-state').hidden = name !== 'empty';
    $('view-header').hidden = name !== 'collection';
    $('recos').hidden = true;
    if (name !== 'collection') viewingId = null;
    renderSidebarCollections();
    window.scrollTo({ top: 0, behavior: 'instant' });
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
    mainGrid.skeleton(14);
    updateUrl();
    loadPage(renderGen);
}

async function loadPage(gen) {
    if (!results.hasMore || results.loadingGen === gen) return;
    results.loadingGen = gen;
    if (results.offset > 0) setStatus(statusEl, 'Loading', true);
    try {
        const data = await postJSON('/search', { ...results.request, offset: results.offset });
        if (gen !== renderGen) return;
        results.offset += data.results.length;
        results.hasMore = Boolean(data.has_more);
        const list = data.results.map(toItem).filter(it => !hiddenIds.has(it.id));
        await fillGrid(mainGrid, list, gen, 'results', it => results.items.push(it));
        if (gen !== renderGen) return;
        setStatus(statusEl, mainGrid.items.length ? (results.hasMore ? '' : 'End of results') : 'No results');
    } catch (err) {
        console.error(err);
        if (gen === renderGen) {
            if (mainGrid.hasSkeleton) mainGrid.reset();
            setStatus(statusEl, `${err.message} — try again`);
        }
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
    fillGrid(mainGrid, previous, gen, 'results', it => results.items.push(it)).then(done => {
        if (done) setStatus(statusEl, results.hasMore ? '' : 'End of results');
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
            const data = await postJSON('/cluster-members', { faiss_ids: ids });
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

// ── Sidebar collections ─────────────────────────────────────────────────────
// Rows stay in creation order and are updated in place (not rebuilt), so the
// target row can grow with a CSS transition instead of jumping to the top.
const DRAG_TYPE = 'application/x-glane-item';

function acceptsItemDrop(node, onDrop) {
    node.addEventListener('dragover', e => {
        if (!e.dataTransfer.types.includes(DRAG_TYPE)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        node.classList.add('drop-hover');
    });
    node.addEventListener('dragleave', e => {
        if (!node.contains(e.relatedTarget)) node.classList.remove('drop-hover');
    });
    node.addEventListener('drop', e => {
        const raw = e.dataTransfer.getData(DRAG_TYPE);
        node.classList.remove('drop-hover');
        document.body.classList.remove('is-dragging');
        if (!raw) return;
        e.preventDefault();
        onDrop(JSON.parse(raw));
    });
}

function dropIntoCollection(cid, item) {
    const c = Collections.get(cid);
    if (Collections.contains(cid, item)) { toast(`Already in “${c.name}”`); bumpRow(cid); return; }
    collectInto(cid, item, null);
}

function buildCollectionRow(c) {
    const li = el('li', 'collection-row');
    li.dataset.id = c.id;
    const btn = el('button', 'collection-item');
    btn.type = 'button';
    const cover = el('span', 'collection-cover');
    const text = el('span', 'collection-text');
    text.append(
        el('span', 'collection-kicker', 'COLLECTING INTO'),
        el('span', 'collection-title'),
        el('span', 'collection-meta'),
    );
    btn.append(cover, text);
    btn.addEventListener('click', () => openCollection(c.id));
    li.appendChild(btn);
    acceptsItemDrop(li, item => dropIntoCollection(c.id, item));
    return li;
}

function renderSidebarCollections() {
    const list = $('collection-list');
    const existing = new Map([...list.querySelectorAll('.collection-row')].map(li => [li.dataset.id, li]));
    list.querySelector('.collection-empty')?.remove();

    Collections.all().forEach((c, i) => {
        let li = existing.get(c.id);
        if (li) existing.delete(c.id);
        else { li = buildCollectionRow(c); li.style.setProperty('--i', i); }
        if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);

        const isTarget = c.id === Collections.data.activeId;
        li.classList.toggle('is-target', isTarget);
        li.classList.toggle('is-viewing', view === 'collection' && c.id === viewingId);
        li.querySelector('.collection-title').textContent = c.name;
        li.querySelector('.collection-meta').textContent = countLabel(c.items.length);
        li.querySelector('.collection-item').title = isTarget ? `Open “${c.name}”` : `Open “${c.name}” and collect into it`;
        const cover = Collections.cover(c);
        li.querySelector('.collection-cover').style.backgroundImage = cover ? cssUrl(cover.src) : '';
    });
    existing.forEach(li => li.remove());

    if (!Collections.all().length) {
        list.appendChild(el('li', 'collection-empty', 'No collection yet. Collect an image, or drag it here.'));
    }
}

acceptsItemDrop($('new-collection-btn'), item => {
    const c = Collections.create();
    setTarget(c.id, false);
    collectInto(c.id, item, null);
});

// ── Collection view + automatic suggestions ────────────────────────────────
function openCollection(id) {
    const c = Collections.get(id);
    if (!c) return;
    closeMenu();
    setTarget(id, false);
    viewingId = id;
    showView('collection');
    $('collection-name').value = c.name;
    $('collection-count').textContent = countLabel(c.items.length);
    $('back-btn').hidden = !results.request;
    $('collection-search-btn').disabled = !c.items.some(i => i.id != null);
    const gen = renderGen;
    if (!c.items.length) {
        setStatus(statusEl, 'Empty for now. Use COLLECT on any image to add it here.');
    } else {
        mainGrid.skeleton(Math.min(c.items.length, 8));
        fillGrid(mainGrid, c.items, gen, 'collection').then(done => {
            if (done && !mainGrid.items.length) setStatus(statusEl, 'These images can no longer be displayed.');
        });
    }
    loadRecos(c, gen);
}

async function loadRecos(c, gen) {
    const refs = c.items.filter(i => i.id != null).slice(0, MAX_COLLECTION_REFS);
    const section = $('recos');
    const recosStatus = $('recos-status');
    if (!refs.length) { section.hidden = true; return; }
    section.hidden = false;
    $('recos-sub').textContent = `Found automatically from ${refs.length === c.items.length ? 'the' : 'the latest'} ${countLabel(refs.length)} of this collection.`;
    setStatus(recosStatus, '');

    const s = currentSettings();
    const signature = [s.combination_mode, ...refs.map(r => r.id)].join(',');
    let list = recosCache.get(signature);
    if (!list) {
        recosGrid.skeleton(8);
        try {
            const data = await postJSON('/search', {
                query: '',
                page_size: RECOS_SIZE,
                diversity: 0.5,
                combination_mode: s.combination_mode,
                reference_ids: refs.map(r => r.id),
                image_weight: 0.5,
            });
            list = data.results.map(toItem);
            recosCache.set(signature, list);
        } catch (err) {
            console.error(err);
            if (gen === renderGen) { recosGrid.reset(); setStatus(recosStatus, 'Suggestions are unavailable right now.'); }
            return;
        }
    }
    if (gen !== renderGen) return;
    const inCollection = new Set(c.items.map(itemKey));
    const fresh = list.filter(it => !inCollection.has(itemKey(it)) && !hiddenIds.has(it.id));
    const done = await fillGrid(recosGrid, fresh, gen, 'results');
    if (done && !recosGrid.items.length) setStatus(recosStatus, 'No suggestions for now.');
}

function searchWithCollection() {
    const c = Collections.get(viewingId);
    const usable = c.items.filter(i => i.id != null).slice(0, MAX_COLLECTION_REFS);
    if (!usable.length) return;
    query.refs = usable.map(refFromItem);
    query.negs = [];
    searchInput.value = '';
    renderChips();
    runSearch();
    if (c.items.length > usable.length) toast(`Searching with the latest ${usable.length} images`);
}

$('new-collection-btn').addEventListener('click', () => {
    const c = Collections.create();
    openCollection(c.id);
    const name = $('collection-name');
    name.focus();
    name.select();
});

$('collection-zip-btn').addEventListener('click', async e => {
    const btn = e.currentTarget;
    const c = Collections.get(viewingId);
    const ids = c.items.filter(i => i.id != null).map(i => i.id);
    if (!ids.length) { toast('No image to download'); return; }
    btn.disabled = true;
    btn.textContent = 'PREPARING ZIP…';
    try {
        const resp = await fetch(`${API}/collection-zip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: c.name, faiss_ids: ids }),
        });
        if (!resp.ok) throw new Error(`Zip failed (${resp.status})`);
        const missing = Number(resp.headers.get('X-Missing') || 0);
        const url = URL.createObjectURL(await resp.blob());
        const a = el('a');
        a.href = url;
        a.download = `${c.name.replace(/[<>:"/\\|?*]+/g, ' ').trim() || 'collection'}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        toast(missing ? `Zip ready. ${missing} image${missing > 1 ? 's' : ''} could not be downloaded.` : 'Zip ready');
    } catch (err) {
        console.error(err);
        toast('The zip could not be prepared. Please try again.');
    } finally {
        btn.disabled = false;
        btn.textContent = 'DOWNLOAD ZIP';
    }
});

$('collection-name').addEventListener('change', e => {
    Collections.rename(viewingId, e.target.value);
    e.target.value = Collections.get(viewingId).name;
    repaintAll();
});
$('collection-name').addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });

$('collection-search-btn').addEventListener('click', searchWithCollection);
$('recos-more').addEventListener('click', searchWithCollection);

$('collection-delete-btn').addEventListener('click', async () => {
    const c = Collections.get(viewingId);
    if (!c) return;
    const ok = await confirmDialog({
        title: `Delete “${c.name}”?`,
        text: c.items.length ? `Its ${countLabel(c.items.length)} will be removed from your collections. This can’t be undone.` : 'This collection is empty.',
        confirm: 'DELETE COLLECTION',
        danger: true,
    });
    if (!ok) return;
    Collections.remove(c.id);
    toast(`“${c.name}” deleted`);
    backToResults();
});

$('back-btn').addEventListener('click', backToResults);

// ── Lightbox ────────────────────────────────────────────────────────────────
const lightbox = $('lightbox');
const lbImg = $('lb-img');
let lbGrid = null;
let lbIndex = -1;
let lbItem = null;
let lbSimilarGen = 0;
let lbCloseTimer = null;

function openLightbox(grid, item) {
    lbGrid = grid;
    lbIndex = grid.items.indexOf(item);
    if (lbIndex < 0) return;
    clearTimeout(lbCloseTimer);
    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
    showInLightbox(item, true, 0);
    requestAnimationFrame(() => requestAnimationFrame(() => lightbox.classList.add('is-open')));
    $('lb-close').focus({ preventScroll: true });
}

function closeLightbox() {
    closePicker(true);
    lightbox.classList.remove('is-open');
    document.body.style.overflow = '';
    lbItem = null;
    lbCloseTimer = setTimeout(() => {
        lightbox.hidden = true;
        lbImg.removeAttribute('src');
    }, 300);
}

function paintLightboxActions() {
    const target = Collections.active();
    const collect = $('lb-collect');
    collect.classList.toggle('on', Boolean(target) && Collections.contains(target.id, lbItem));
    collect.title = target ? `Collect into “${target.name}”` : 'Collect into a new collection';

    // Choosing a cover only makes sense while browsing a collection.
    const coverBtn = $('lb-cover');
    const c = view === 'collection' && lbGrid === mainGrid ? Collections.get(viewingId) : null;
    coverBtn.hidden = !c || !Collections.contains(c.id, lbItem);
    if (!coverBtn.hidden) {
        const isCover = itemKey(Collections.cover(c) || {}) === itemKey(lbItem);
        coverBtn.classList.toggle('on', isCover);
        coverBtn.textContent = isCover ? 'COVER' : 'SET AS COVER';
    }
}

function showInLightbox(item, withSimilar, direction) {
    lbItem = item;
    lbImg.getAnimations().forEach(a => a.cancel());
    lbImg.style.opacity = '0';
    lbImg.onload = () => {
        lbImg.style.opacity = '';
        if (direction) {
            lbImg.animate([
                { opacity: 0, transform: `translateX(${direction * 28}px) scale(0.985)` },
                { opacity: 1, transform: 'none' },
            ], { duration: 480, easing: 'cubic-bezier(0.16, 1, 0.3, 1)' });
        } else {
            lbImg.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, easing: 'ease' });
        }
    };
    const candidates = [...new Set([fullImage(item), item.orig, item.src].filter(Boolean))];
    let attempt = 0;
    lbImg.onerror = () => { if (++attempt < candidates.length) lbImg.src = candidates[attempt]; };
    lbImg.src = candidates[0];
    lbImg.alt = item.title;
    $('lb-title').textContent = item.title;
    $('lb-author').textContent = item.author;
    $('lb-author').hidden = !item.author;
    const museum = $('lb-museum');
    museum.textContent = museumName(item.source);
    if (item.url) museum.href = item.url; else museum.removeAttribute('href');
    museum.title = item.url ? 'See this artwork on the museum website' : '';
    $('lb-download').href = fullImage(item);
    paintLightboxActions();
    $('lb-prev').disabled = lbIndex <= 0;
    $('lb-next').disabled = lbIndex >= lbGrid.items.length - 1;
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
        const data = await postJSON('/cluster-members', { faiss_ids: item.members });
        if (gen !== lbSimilarGen) return;
        data.results.map(toItem).forEach((member, i) => {
            const thumb = el('button', 'lb-thumb');
            thumb.type = 'button';
            thumb.title = member.title;
            thumb.style.setProperty('--i', i);
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
                showInLightbox(member, false, 0);
            });
            strip.appendChild(thumb);
        });
    } catch (err) {
        console.error(err);
        box.hidden = true;
    }
}

function stepLightbox(delta) {
    const next = lbIndex + delta;
    if (!lbGrid || next < 0 || next >= lbGrid.items.length) return;
    lbIndex = next;
    showInLightbox(lbGrid.items[next], true, delta);
}

$('lb-close').addEventListener('click', closeLightbox);
$('lb-prev').addEventListener('click', () => stepLightbox(-1));
$('lb-next').addEventListener('click', () => stepLightbox(1));
$('lb-collect').addEventListener('click', () => {
    if (!lbItem) return;
    const cid = Collections.ensureTarget().id;
    if (Collections.contains(cid, lbItem)) openItemDialog(lbItem, cid);
    else collectInto(cid, lbItem, null);
});
$('lb-cover').addEventListener('click', () => { if (lbItem) setCover(lbItem); });
$('lb-collect-menu').addEventListener('click', e => { e.stopPropagation(); if (lbItem) openPicker(e.currentTarget, lbItem, null); });
$('lb-visual').addEventListener('click', () => {
    if (!lbItem) return;
    const item = lbItem;
    closeLightbox();
    visualResearch(item);
});
lightbox.addEventListener('click', e => { if (e.target === lightbox || e.target.classList.contains('lb-stage')) closeLightbox(); });

document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !picker.hidden) { closePicker(); return; }
    if (dialog.open) return;
    if (lightbox.hidden || !lightbox.classList.contains('is-open')) {
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
    $('settings-panel').classList.toggle('open', open);
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

// ── Account: sign-in by email link + collections saved to the account ──────
class HttpError extends Error {
    constructor(status, detail) { super(detail || `HTTP ${status}`); this.status = status; }
}

async function api(path, { method = 'GET', body, keepalive = false } = {}) {
    const resp = await fetch(`${API}${path}`, {
        method,
        keepalive,
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new HttpError(resp.status, typeof data.detail === 'string' ? data.detail : '');
    return data;
}

const toServer = c => ({
    name: c.name,
    created_at: c.createdAt,
    updated_at: c.updatedAt || c.createdAt,
    cover_key: c.coverKey || null,
    items: c.items.filter(i => /^https?:\/\//.test(i.src)),
});
const fromServer = c => ({
    id: c.id, name: c.name, createdAt: c.created_at, updatedAt: c.updated_at, coverKey: c.cover_key, items: c.items,
});

const SYNC_TEXT = {
    pending: 'Saving…',
    saved: 'All changes are saved to your account.',
    offline: 'Offline. Changes will be saved as soon as the connection is back.',
};

const Sync = {
    email: null,
    available: true,
    dirty: new Set(),
    deleted: new Map(),       // id → deletion time
    timer: null,
    busy: false,

    get on() { return Boolean(this.email); },
    markDirty(id) {
        if (!this.on) return;
        this.dirty.add(id);
        this.deleted.delete(id);
        this.schedule();
    },
    markDeleted(id) {
        if (!this.on) return;
        this.dirty.delete(id);
        this.deleted.set(id, Date.now());
        this.schedule();
    },
    schedule(delay = 700) {
        clearTimeout(this.timer);
        this.setState('pending');
        this.timer = setTimeout(() => this.flush(), delay);
    },
    setState(state) { $('sync-status').textContent = SYNC_TEXT[state]; },

    async flush(keepalive = false) {
        if (!this.on) return;
        if (this.busy) { this.schedule(); return; }
        this.busy = true;
        let changed = false;
        try {
            for (const id of [...this.dirty]) {
                const c = Collections.get(id);
                this.dirty.delete(id);
                if (!c) continue;
                try {
                    const res = await api(`/api/collections/${encodeURIComponent(id)}`, { method: 'PUT', body: toServer(c), keepalive });
                    if (res.status === 'stale') {
                        // Changed more recently on another device: adopt that version.
                        const others = Collections.all().filter(x => x.id !== id);
                        Collections.replaceAll(res.deleted ? others : [...others, fromServer(res.collection)]);
                        changed = true;
                    }
                } catch (err) {
                    this.dirty.add(id);
                    throw err;
                }
            }
            for (const [id, at] of [...this.deleted]) {
                this.deleted.delete(id);
                try {
                    await api(`/api/collections/${encodeURIComponent(id)}?updated_at=${at}`, { method: 'DELETE', keepalive });
                } catch (err) {
                    this.deleted.set(id, at);
                    throw err;
                }
            }
            this.setState('saved');
        } catch (err) {
            if (err.status === 401) { this.expired(); return; }
            console.error('sync', err);
            this.setState('offline');
            clearTimeout(this.timer);
            this.timer = setTimeout(() => this.flush(), 8000);
        } finally {
            this.busy = false;
            if (changed) repaintAll();
        }
    },

    // First sign-in on a device: the guest's collections join the account.
    async pullAndMerge() {
        const data = await api('/api/collections');
        const server = new Map(data.collections.map(c => [c.id, fromServer(c)]));
        const deleted = new Set(data.deleted);
        const merged = [];
        let uploaded = 0;
        for (const c of Collections.all()) {
            if (deleted.has(c.id)) continue;
            const remote = server.get(c.id);
            server.delete(c.id);
            if (!remote) { merged.push(c); this.dirty.add(c.id); uploaded++; continue; }
            if ((c.updatedAt || 0) > remote.updatedAt) { merged.push(c); this.dirty.add(c.id); }
            else merged.push(remote);
        }
        merged.push(...server.values());
        Collections.replaceAll(merged);
        repaintAll();
        if (view === 'collection' && !Collections.get(viewingId)) backToResults();
        if (this.dirty.size) await this.flush();
        else this.setState('saved');
        return uploaded;
    },

    expired() {
        this.email = null;
        this.dirty.clear();
        this.deleted.clear();
        renderAccount();
        toast('Your session has ended. Sign in again to keep saving your collections.');
    },
};

function renderAccount() {
    $('account-out').hidden = Sync.on;
    $('account-in').hidden = !Sync.on;
    $('account-email').textContent = Sync.email || '';
    $('auth-form').hidden = !Sync.available;
    $('account-intro').textContent = Sync.available
        ? 'Sign in with your email to keep your collections on every device. No password: we send you a link.'
        : 'Accounts are coming soon. For now, collections are kept in this browser.';
}

function showAuthStatus(text, extraLink = null) {
    const note = $('auth-status');
    note.replaceChildren(text);
    if (extraLink) {
        const a = el('a', null, extraLink.label);
        a.href = extraLink.href;
        a.target = '_blank';
        a.rel = 'noopener';
        note.append(' ', a);
    }
    note.hidden = false;
}

$('auth-form').addEventListener('submit', async e => {
    e.preventDefault();
    const email = $('auth-email').value.trim();
    const btn = $('auth-btn');
    btn.disabled = true;
    btn.textContent = 'SENDING…';
    try {
        const res = await api('/auth/request', { method: 'POST', body: { email } });
        showAuthStatus(`Check your inbox: a sign-in link is on its way to ${email}. It works once, for 15 minutes.`,
            res.dev_outbox ? { label: 'Open the dev outbox', href: res.dev_outbox } : null);
    } catch (err) {
        console.error(err);
        showAuthStatus(err.status === 503 ? 'Sign-in is not available yet.' : (err.message || 'The link could not be sent. Please try again.'));
    } finally {
        btn.disabled = false;
        btn.textContent = 'SEND LINK';
    }
});

$('logout-btn').addEventListener('click', async () => {
    clearTimeout(Sync.timer);
    await Sync.flush();
    try { await api('/auth/logout', { method: 'POST' }); } catch (err) { console.error(err); }
    Sync.email = null;
    // Collections now live in the account: leave nothing behind on a shared computer.
    Collections.replaceAll([]);
    renderAccount();
    if (view === 'collection') backToResults();
    repaintAll();
    toast('Signed out. Your collections are safe in your account.');
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && (Sync.dirty.size || Sync.deleted.size)) Sync.flush(true);
});

async function initAccount() {
    const params = new URLSearchParams(location.search);
    const signedIn = params.has('signed_in');
    const loginError = params.get('login_error');
    if (signedIn || loginError) {
        params.delete('signed_in');
        params.delete('login_error');
        const qs = params.toString();
        history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
    }
    try {
        const s = await api('/auth/session');
        Sync.available = s.available !== false;
        Sync.email = s.authenticated ? s.email : null;
    } catch (err) {
        console.error('session', err);
        Sync.available = false;
    }
    renderAccount();
    if (loginError) toast('This sign-in link has expired or was already used. Ask for a new one.');
    if (!Sync.on) return;
    try {
        const uploaded = await Sync.pullAndMerge();
        if (signedIn) {
            toast(uploaded
                ? `Signed in as ${Sync.email}. ${uploaded} collection${uploaded > 1 ? 's' : ''} from this browser now saved to your account.`
                : `Signed in as ${Sync.email}.`);
        }
    } catch (err) {
        if (err.status === 401) Sync.expired();
        else { console.error(err); Sync.setState('offline'); }
    }
}

// ── Landing: animated wordmark + floating images that preview each prompt ──
// The images come from a static file (landing.json): no search runs on arrival.
const LANDING_SLOTS = [
    { x: '3%', y: '7%', w: '12%', depth: 26, dur: '7.5s', rot: 1.4 },
    { x: '16%', y: '34%', w: '10%', depth: 14, dur: '9s', rot: -1.1 },
    { x: '4%', y: '58%', w: '13%', depth: 32, dur: '8.2s', rot: 0.9 },
    { x: '19%', y: '80%', w: '10%', depth: 18, dur: '10s', rot: -1.6 },
    { x: '77%', y: '6%', w: '12%', depth: 22, dur: '8.8s', rot: -1.2 },
    { x: '89%', y: '31%', w: '9%', depth: 12, dur: '9.6s', rot: 1.3 },
    { x: '75%', y: '55%', w: '11%', depth: 30, dur: '7.8s', rot: -0.8 },
    { x: '87%', y: '76%', w: '11%', depth: 20, dur: '10.5s', rot: 1.1 },
    { x: '39%', y: '86%', w: '8%', depth: 10, dur: '11s', rot: 0.7 },
    { x: '56%', y: '3%', w: '7%', depth: 8, dur: '12s', rot: -0.9 },
];
const LANDING_READY_TIMEOUT_MS = 6000;

const Landing = {
    data: null,
    sets: [],            // [default mix, prompt 1, prompt 2, …]
    current: 0,
    cards: [],
    grid: { items: [] }, // lightbox navigation over the floating images
    swapTimer: null,

    async init() {
        const params = new URLSearchParams(location.search);
        if (params.has('q') || params.has('ref')) { this.finish(); return; }   // arriving on a search
        try {
            this.data = await (await fetch('landing.json')).json();
        } catch (err) {
            console.error('landing', err);
            this.finish();
            return;
        }
        const prompts = this.data.prompts;
        const mix = [];
        for (let round = 0; mix.length < LANDING_SLOTS.length && round < 10; round++) {
            for (const p of prompts) if (p.items[round] && mix.length < LANDING_SLOTS.length) mix.push(p.items[round]);
        }
        this.sets = [mix, ...prompts.map(p => p.items.slice(0, LANDING_SLOTS.length))];
        this.buildPrompts(prompts);
        this.buildCards();
        await this.preload(this.sets[0], true);
        this.finish();
        this.show(0, false);
        // Warm the other sets so hovering a prompt swaps instantly.
        const rest = this.sets.slice(1).flat();
        (window.requestIdleCallback || setTimeout)(() => this.preload(rest, false));
    },

    buildPrompts(prompts) {
        const box = $('landing-prompts');
        prompts.forEach((p, i) => {
            const b = el('button', 'suggestion', p.label);
            b.type = 'button';
            b.style.setProperty('--i', i);
            b.addEventListener('click', () => { searchInput.value = p.label; runSearch(); });
            if (window.matchMedia('(hover: hover)').matches) {
                b.addEventListener('mouseenter', () => this.preview(i + 1, b));
                b.addEventListener('focus', () => this.preview(i + 1, b));
            }
            box.appendChild(b);
        });
    },

    buildCards() {
        const stage = $('landing-stage');
        this.cards = LANDING_SLOTS.map((slot, n) => {
            const card = el('button', 'float-card');
            card.type = 'button';
            for (const [k, v] of Object.entries(slot)) card.style.setProperty(`--${k}`, v);
            card.style.setProperty('--n', n);
            const inner = el('span', 'float-inner');
            const img = new Image();
            img.alt = '';
            img.decoding = 'async';
            img.addEventListener('error', () => {
                if (card._item?.orig && img.src !== new URL(card._item.orig, location.href).href) img.src = card._item.orig;
            });
            inner.append(img, el('span', 'float-title'));
            card.appendChild(inner);
            card.addEventListener('click', () => { if (card._item) openLightbox(this.grid, card._item); });
            stage.appendChild(card);
            return card;
        });
        this.parallax(stage);
    },

    preload(items, track) {
        const bar = $('landing-progress-bar');
        let done = 0;
        const all = items.map(it => preload(it).then(() => {
            done++;
            if (track) bar.style.setProperty('--p', done / items.length);
        }));
        const everything = Promise.all(all);
        return track ? Promise.race([everything, new Promise(r => setTimeout(r, LANDING_READY_TIMEOUT_MS))]) : everything;
    },

    // The wordmark waves at least twice, then each letter stops at the end of
    // its current cycle, so it settles on the resting shape without a jump.
    finish() {
        const landing = $('empty-state');
        const letters = landing.querySelectorAll('.wm-letter');
        letters.forEach(letter => letter.classList.add('is-waving'));
        landing.classList.remove('is-loading');
        requestAnimationFrame(() => letters.forEach(letter => {
            const wave = letter.getAnimations().find(a => a.animationName === 'wmWave');
            if (!wave) return;
            const current = wave.effect.getComputedTiming().currentIteration;
            wave.effect.updateTiming({ iterations: Math.max(2, (current ?? 0) + 1) });
        }));
    },

    show(setIndex, animate) {
        const items = this.sets[setIndex] || [];
        this.current = setIndex;
        this.grid.items = items.slice();
        this.cards.forEach((card, n) => {
            const item = items[n];
            const apply = () => {
                card._item = item;
                card.hidden = !item;
                if (!item) return;
                const img = card.querySelector('img');
                img.src = item.src;
                card.style.setProperty('--ratio', item.ratio || 1.2);
                card.querySelector('.float-title').textContent = item.title;
                card.setAttribute('aria-label', `Open ${item.title}`);
                card.classList.remove('swap-out');
            };
            if (!animate) { apply(); return; }
            setTimeout(() => {
                card.classList.add('swap-out');
                setTimeout(apply, 320);
            }, n * 45);
        });
    },

    preview(setIndex, button) {
        document.querySelectorAll('#landing-prompts .suggestion').forEach(b => b.classList.toggle('is-previewing', b === button));
        if (setIndex === this.current) return;
        clearTimeout(this.swapTimer);
        this.swapTimer = setTimeout(() => this.show(setIndex, true), 90);
    },

    // Images drift toward the mouse, each at its own depth.
    parallax(stage) {
        if (!window.matchMedia('(hover: hover)').matches) return;
        const target = { x: 0, y: 0 };
        const now = { x: 0, y: 0 };
        let running = false;
        const tick = () => {
            now.x += (target.x - now.x) * 0.06;
            now.y += (target.y - now.y) * 0.06;
            stage.style.setProperty('--mx', now.x.toFixed(4));
            stage.style.setProperty('--my', now.y.toFixed(4));
            running = Math.abs(target.x - now.x) + Math.abs(target.y - now.y) > 0.001;
            if (running) requestAnimationFrame(tick);
        };
        $('empty-state').addEventListener('mousemove', e => {
            const r = stage.getBoundingClientRect();
            target.x = ((e.clientX - r.left) / r.width - 0.5) * 2;
            target.y = ((e.clientY - r.top) / r.height - 0.5) * 2;
            if (!running) { running = true; requestAnimationFrame(tick); }
        });
    },
};

// ── Init ────────────────────────────────────────────────────────────────────
Collections.load();
mainGrid.build();
recosGrid.build();
renderSidebarCollections();
renderChips();
restoreFromUrl();
initAccount();
Landing.init();
