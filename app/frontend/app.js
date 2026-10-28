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
    remove: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>',
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
const countLabel = n => t(n === 1 ? 'count.one' : 'count.many', { n });
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
        title: r.Title || t('untitled'),
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
    // Most recently used first: the last created, or the last one an image went
    // into (collections from an account, which carry no usedAt: their last change).
    all() {
        const used = c => c.usedAt || c.updatedAt || c.createdAt;
        return [...this.data.list].sort((a, b) => used(b) - used(a));
    },
    get(id) { return this.data.list.find(c => c.id === id); },
    active() { return this.get(this.data.activeId) || null; },
    setActive(id) { this.data.activeId = id; this.save(); },
    ensureTarget() {
        if (!this.active()) this.setActive(this.create().id);
        return this.active();
    },
    // The cover is the image chosen by the user, otherwise the first one ever
    // put in (items are kept newest first, so the last of the list).
    cover(c) {
        return (c.coverKey && c.items.find(i => itemKey(i) === c.coverKey)) || c.items.at(-1) || null;
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
        this.get(id).usedAt = Date.now();
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
        const c = { id: uid(), name: name || t('collections.defaultName', { n: this.data.list.length + 1 }), items: [], createdAt: now, updatedAt: now, usedAt: now, coverKey: null };
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

let view = 'empty';            // 'empty' | 'results' | 'collection' | 'shared' | 'weather'
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
            const before = new Map(this.entries.map(e => [e.node, e.node.getBoundingClientRect()]));
            this.n = this.columnCount();
            this.container.replaceChildren();
            this.cols = [];
            this.heights = new Array(this.n).fill(0);
            for (let i = 0; i < this.n; i++) this.cols.push(this.container.appendChild(el('div', 'masonry-col')));
            // Cards already on screen change column without playing their entrance again.
            for (const e of this.entries) { e.node.classList.add('is-settled'); this.place(e); }
            this.glide(before);
        },
        // After a new column count, the cards in view slide from where they
        // were to their new place (their width changes at once: an image is
        // never stretched).
        glide(before) {
            const seen = r => r.width && r.bottom > 0 && r.top < innerHeight;
            for (const e of this.entries) {
                const a = before.get(e.node);
                if (!a) continue;
                const b = e.node.getBoundingClientRect();
                if (!seen(a) && !seen(b)) continue;
                const dx = a.left - b.left, dy = a.top - b.top;
                if (Math.abs(dx) + Math.abs(dy) < 1) continue;
                e.node.animate([{ translate: `${dx}px ${dy}px` }, { translate: '0 0' }],
                    { duration: 560, easing: 'cubic-bezier(0.2, 0, 0, 1)' });
            }
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
    card.style.aspectRatio = `1 / ${ratio}`;   // the card is a size container (style.css): it needs its own height
    frame.setAttribute('aria-label', t('card.open', { title: item.title }));
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
    Filets.card(card, frame, img, item);

    if (item.members) {
        const badge = el('span', 'card-similar', `+${item.members.length - 1}`);
        badge.appendChild(el('span', 'card-similar-label', t('card.similar')));
        card.appendChild(badge);
    }

    const overlay = el('div', 'card-overlay');
    const top = el('div', 'card-top');
    if (grid === mainGrid && view === 'results') {
        const hideBtn = el('button', 'card-icon-btn');
        hideBtn.type = 'button';
        hideBtn.title = t('card.hide');
        hideBtn.setAttribute('aria-label', t('card.hide'));
        hideBtn.innerHTML = ICON.hide;
        hideBtn.addEventListener('click', e => { e.stopPropagation(); showHideOverlay(card, grid); });
        top.appendChild(hideBtn);
    }
    if (mode === 'collection') {
        const coverBtn = el('button', 'card-icon-btn card-cover');
        coverBtn.type = 'button';
        coverBtn.innerHTML = ICON.cover;
        coverBtn.addEventListener('click', e => { e.stopPropagation(); setCover(item); });
        // Out of the collection in one click; the message offers to undo it.
        const removeBtn = el('button', 'card-icon-btn card-remove');
        removeBtn.type = 'button';
        removeBtn.innerHTML = ICON.remove;
        removeBtn.title = t('card.remove');
        removeBtn.setAttribute('aria-label', removeBtn.title);
        removeBtn.addEventListener('click', e => { e.stopPropagation(); removeItem(item, viewingId); });
        top.append(coverBtn, removeBtn);
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
    chevron.title = t('card.chooser');
    chevron.setAttribute('aria-label', t('card.chooser'));
    chevron.setAttribute('aria-haspopup', 'menu');
    chevron.addEventListener('click', e => { e.stopPropagation(); openPicker(chevron, item, img); });
    split.append(collectBtn, chevron);

    const visualBtn = el('button', 'card-btn');
    visualBtn.type = 'button';
    visualBtn.title = t('card.visual.title');
    visualBtn.setAttribute('aria-label', t('card.visual'));
    visualBtn.innerHTML = `${ICON.visual}<span class="card-btn-label">${t('card.visual')}</span>`;
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
    btn.innerHTML = `${on ? ICON.collected : ICON.collect}<span class="card-btn-label">${on ? t('card.collected') : t('card.collect')}</span>`;
    btn.title = on ? t('card.inCollection', { name: target.name })
        : target ? t('card.addTo', { name: target.name }) : t('card.addToNew');
    btn.setAttribute('aria-label', btn.title);

    const coverBtn = card.querySelector('.card-cover');
    if (coverBtn) {
        const c = Collections.get(viewingId);
        const isCover = Boolean(c) && itemKey(Collections.cover(c) || {}) === itemKey(card._item);
        coverBtn.classList.toggle('on', isCover);
        coverBtn.title = isCover ? t('card.cover') : t('card.useAsCover');
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
    toast(t('toast.cover', { name: c.name }));
}

// Phone: the collection images go into is chosen at the first COLLECT of each
// visit: there the menu, which shows it, is folded away, and the last one used
// may date from a visit long ago. A reload keeps the choice, a new tab asks.
const COLLECT_CHOSEN = 'glane.collectChosen';
function askWhereToCollect() {
    if (!window.matchMedia('(max-width: 860px)').matches || !Collections.all().length) return false;
    try { return !sessionStorage.getItem(COLLECT_CHOSEN); } catch { return false; }
}
function collectChosen() {
    try { sessionStorage.setItem(COLLECT_CHOSEN, '1'); } catch { /* private mode */ }
}

function onCollectClick(card) {
    const ask = card._mode !== 'collection' && askWhereToCollect();   // before a first collection is made
    const cid = cardCollectionId(card) ?? Collections.ensureTarget().id;
    const item = card._item;
    // Already collected: the checklist of collections, to take it out of one or add it to another.
    if (Collections.contains(cid, item)) { openPicker(card.querySelector('.card-collect'), item, card.querySelector('.card-img img')); return; }
    if (ask) { openPicker(card.querySelector('.card-collect'), item, card.querySelector('.card-img img')); return; }
    collectChosen();
    collectInto(cid, item, card.querySelector('.card-img img'));
    const btn = card.querySelector('.card-collect');
    btn.classList.remove('pop');
    void btn.offsetWidth;
    btn.classList.add('pop');
}

function collectInto(cid, item, fromImg) {
    if (!Collections.add(cid, item)) return;
    const c = Collections.get(cid);
    toast(t('toast.added', { name: c.name }), {
        label: t('toast.undo'),
        run: () => { Collections.removeItem(cid, item); afterRemoval(item, cid); repaintAll(); },
    });
    repaintAll();
    flyToCollection(cid, fromImg, item);
    if (view === 'collection' && viewingId === cid) {
        const recoCard = recosGrid.cardFor(item);
        if (recoCard) recosGrid.remove(recoCard);
        const ratio = recoCard ? parseFloat(recoCard.querySelector('.card-img').style.aspectRatio.split('/')[1]) : 1;
        mainGrid.add(createCard(item, clampRatio(ratio), mainGrid, 'collection'), clampRatio(ratio), item);
        setStatus(statusEl, '');
        repaintAll();
    }
}

// Visible and not covered by anything (e.g. a folded menu).
function isOnScreen(node) {
    const r = node.getBoundingClientRect();
    if (!r.width || !r.height || r.bottom < 0 || r.top > window.innerHeight) return false;
    const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return Boolean(top) && (node.contains(top) || top.contains(node));
}

function rowFor(cid) {
    return document.querySelector(`.collection-row[data-id="${CSS.escape(cid)}"]`);
}

// Keeps the grid of the collection being viewed in sync after a removal.
function afterRemoval(item, cid) {
    if (view === 'collection' && viewingId === cid) {
        const card = mainGrid.cardFor(item);
        if (card) mainGrid.remove(card);
    }
}

// The image folds into a filet of its own pixels and slides into the spine of
// its collection, which stays hidden until it arrives (filets.js).
// A collection's band: one row of pixels from each of its latest images, side
// by side, oldest on the left, so it grows to the right as images come in.
const BAND_ITEMS = 16;
function paintCollectionBand(band, c) {
    band.replaceChildren(...c.items.slice(0, BAND_ITEMS).reverse().map(it => {
        const seg = el('i');
        seg.dataset.key = itemKey(it);
        Filets.paint(seg, Filets.fromItem(it), 'row');
        return seg;
    }));
}

// Collecting: the image's line of pixels rises over the thumbnail, then slides
// into its new segment of the collection's band. When the menu is folded away
// (phone: its button; computer: the star), it slides into what stands for it.
function flyToCollection(cid, fromImg, item) {
    const seg = rowFor(cid)?.querySelector(`.collection-band > i[data-key="${CSS.escape(itemKey(item))}"]`);
    seg?.classList.add('is-waiting');
    // The first collection unfolds the menu: the band leaves once it is in place.
    setTimeout(() => {
        let dest = seg && isOnScreen(seg.closest('.collection-item')) ? seg : null;
        const stand = [$('menu-btn'), MenuFold.star].find(b => isOnScreen(b));
        if (!dest && stand) {
            const m = stand.getBoundingClientRect();
            dest = new DOMRect(m.left + 8, m.top + m.height / 2 - 3, Math.max(6, m.width - 16), 6);
        }
        Filets.fold(fromImg, item, dest, () => {
            seg?.classList.remove('is-waiting');
            Filets.reveal(seg);
        });
    }, Math.max(0, MenuFold.settled - Date.now()));
}

function showHideOverlay(card, grid) {
    const overlay = el('div', 'hide-overlay');
    overlay.innerHTML = `
        <div class="hide-overlay-top">
            <span class="sb-label">${t('hide.title')}</span>
            <button type="button" class="link-btn" data-action="cancel">${t('hide.cancel')}</button>
        </div>
        <div class="hide-options">
            <button type="button" class="hide-option" data-action="negative">
                <span class="ho-title">${t('hide.negative')}</span>
                <span class="ho-desc">${t('hide.negative.desc')}</span>
            </button>
            <button type="button" class="hide-option" data-action="taste">
                <span class="ho-title">${t('hide.taste')}</span>
                <span class="ho-desc">${t('hide.taste.desc')}</span>
            </button>
            <button type="button" class="hide-option" data-action="flag">
                <span class="ho-title">${t('hide.flag')}</span>
                <span class="ho-desc">${t('hide.flag.desc')}</span>
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
    picker.replaceChildren(el('div', 'picker-label', item ? t('picker.collectInto') : t('picker.newGoInto')));

    const rows = [];
    const row = (c, i) => {
        const btn = el('button', 'picker-item');
        btn.type = 'button';
        btn.style.setProperty('--i', i);
        const band = el('span', 'collection-band');
        const cover = el('span', 'picker-cover');
        const meta = el('span', 'picker-meta');
        const text = el('span', 'picker-text');
        text.append(el('span', 'picker-name', c.name), meta);
        btn.append(cover, text, band);
        const paint = () => {
            btn.classList.toggle('is-target', c.id === Collections.data.activeId);
            const first = Collections.cover(c);
            cover.style.backgroundImage = first ? cssUrl(first.src) : '';
            paintCollectionBand(band, c);
            meta.textContent = countLabel(c.items.length);
            if (item) {
                const on = Collections.contains(c.id, item);
                btn.classList.toggle('is-checked', on);
                btn.setAttribute('aria-checked', String(on));
            }
        };
        rows.push(paint);
        if (!item) {
            // Choosing where new images go.
            btn.setAttribute('role', 'menuitem');
            btn.addEventListener('click', () => { closePicker(); setTarget(c.id, true); });
        } else {
            // Each collection is a checkbox: tick to add the image, untick to take it out.
            btn.setAttribute('role', 'menuitemcheckbox');
            btn.appendChild(el('span', 'picker-check'));
            btn.addEventListener('click', e => {
                e.stopPropagation();
                if (Collections.contains(c.id, item)) {
                    removeItem(item, c.id);
                    if (view === 'collection' && viewingId === c.id && !document.body.contains(anchor)) { closePicker(); return; }
                } else {
                    setTarget(c.id, false);
                    collectChosen();
                    collectInto(c.id, item, fromImg);
                }
                rows.forEach(p => p());
            });
        }
        paint();
        return btn;
    };
    Collections.all().forEach((c, i) => picker.appendChild(row(c, i)));

    const create = el('button', 'picker-item picker-new');
    create.type = 'button';
    create.setAttribute('role', 'menuitem');
    create.style.setProperty('--i', Collections.all().length);
    const plus = el('span', 'picker-cover', '+');
    const text = el('span', 'picker-text');
    text.append(el('span', 'picker-name', t('picker.new')));
    create.append(plus, text);
    // A new collection is named before anything goes into it.
    create.addEventListener('click', async () => {
        closePicker();
        const name = await askCollectionName();
        if (name == null) return;
        const c = Collections.create(name);
        setTarget(c.id, false);
        collectChosen();
        if (item) collectInto(c.id, item, fromImg);
        else toast(t('toast.target', { name: c.name }));
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
    if (announce) toast(t('toast.target', { name: Collections.get(id).name }));
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

function removeItem(item, cid) {
    const idx = Collections.removeItem(cid, item);
    afterRemoval(item, cid);
    repaintAll();
    toast(t('toast.removed', { name: Collections.get(cid).name }), {
        label: t('toast.undo'),
        run: () => { Collections.insertAt(cid, item, idx); restoreInView(item, cid); repaintAll(); },
    });
}

function restoreInView(item, cid) {
    if (view === 'collection' && viewingId === cid && !mainGrid.cardFor(item)) {
        setStatus(statusEl, '');
        fillGrid(mainGrid, [item], renderGen, 'collection');
    }
}

// The name of a new collection (the default one already filled in); null if cancelled.
function askCollectionName() {
    return new Promise(resolve => {
        let answered = false;
        const answer = v => { if (!answered) { answered = true; resolve(v); } };
        openDialog(d => {
            const fallback = t('collections.defaultName', { n: Collections.all().length + 1 });
            const form = el('form', 'dialog-form');
            const input = el('input', 'dialog-input');
            input.type = 'text';
            input.value = fallback;
            input.setAttribute('aria-label', t('picker.newName'));
            const create = el('button', 'pill-btn', t('picker.create'));
            create.type = 'submit';
            const foot = el('div', 'dialog-foot');
            foot.append(dialogButton(t('dialog.cancel'), 'link-btn', () => { closeDialog(); answer(null); }), create);
            form.append(el('div', 'dialog-title', t('picker.newName')), input, foot);
            form.addEventListener('submit', e => { e.preventDefault(); closeDialog(); answer(input.value.trim() || fallback); });
            d.appendChild(form);
            requestAnimationFrame(() => { input.focus(); input.select(); });
        });
        dialog.addEventListener('close', () => answer(null), { once: true });
    });
}

function confirmDialog({ title, text, confirm, danger = false }) {
    return new Promise(resolve => {
        openDialog(d => {
            d.append(el('div', 'dialog-title', title), el('div', 'dialog-text', text));
            const foot = el('div', 'dialog-foot');
            foot.append(
                dialogButton(t('dialog.cancel'), 'link-btn', () => { closeDialog(); resolve(false); }),
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
        rm.setAttribute('aria-label', t('search.removeImage'));
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
    updateSettingsUI();
}

function refFromItem(item) {
    return { key: itemKey(item), src: item.src, id: item.id };
}

function addReference(item, negative = false) {
    if (item.id == null) { toast(t('toast.noSearchImage')); return false; }
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
        catch (err) { console.error('image file', err); toast(t('toast.unreadable')); }
    }
    renderChips();
    runSearch();
}

// ── Search & pagination ─────────────────────────────────────────────────────
// Filters read from the works' descriptions (app/backend/facets.py).
const PERIODS = ['antiquity', 'middle_ages', '1400_1600', '1600_1800', '19th', '20th'];
const TECHNIQUES = ['painting', 'drawing', 'print', 'photograph', 'sculpture', 'ceramics',
    'textile', 'metal', 'glass', 'furniture', 'book'];
const checkedValues = name => [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(i => i.value);

function currentFilters() {
    return { periods: checkedValues('period'), techniques: checkedValues('technique') };
}

function currentSettings() {
    return {
        diversity: Number(document.querySelector('input[name="variety"]:checked').value),
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
    const f = currentFilters();
    if (f.periods.length) body.periods = f.periods;
    if (f.techniques.length) body.techniques = f.techniques;
    return body;
}

// ── Browser history ─────────────────────────────────────────────────────────
// Each page (landing, a search, a collection, a shared collection, weather) is one
// history entry and a full-page image one more on top, so the phone's back button
// walks back through the site instead of leaving it.
const Nav = {
    pendingBack: null,
    get canGoBack() { return (history.state?.depth || 0) > 0; },
    push(page, url = location.pathname + location.search) {
        history.replaceState({ ...history.state, scrollY: window.scrollY }, '');
        history.pushState({ ...page, depth: (history.state?.depth || 0) + 1 }, '', url);
    },
    replace(page, url = location.pathname + location.search) {
        history.replaceState({ ...page, depth: history.state?.depth || 0 }, '', url);
    },
    // Goes back one entry; resolves once the browser has done it.
    back() {
        return new Promise(resolve => { this.pendingBack = resolve; history.back(); });
    },
};
history.scrollRestoration = 'manual';

window.addEventListener('popstate', e => {
    if (Nav.pendingBack) { const done = Nav.pendingBack; Nav.pendingBack = null; done(); return; }
    const s = e.state || {};
    if (lbItem && !s.lb) { closeLightbox(true); return; }   // back from a full-page image
    if (s.lb) { history.back(); return; }                   // forward onto a closed image: skip it
    closePicker(true);
    if (dialog.open) dialog.close();
    closeMenu();
    showPage(s);
});

function showPage(s) {
    if (s.view === 'collection' && Collections.get(s.id)) { openCollection(s.id, false); return; }
    if (s.view === 'weather') { Weather.open('none'); return; }
    if (s.view === 'shared' && s.slug) { openShared(s.slug); return; }
    const params = new URLSearchParams(location.search);
    if (!params.has('q') && !params.has('ref')) {
        searchInput.value = '';
        query.refs = [];
        query.negs = [];
        renderChips();
        showView('empty');
        return;
    }
    // Same search still in memory (and in the search bar): show it again as it was.
    const refIds = query.refs.filter(r => r.id != null).map(r => r.id).join(',');
    const intact = results.request && results.url === location.search
        && searchInput.value.trim() === (params.get('q') || '') && refIds === (params.get('ref') || '');
    if (intact) { backToResults(s.scrollY); return; }
    searchFromUrl(params, 'none');
}

function showView(name) {
    // Leaving the weather page puts back the search it covered.
    if (view === 'weather' && name !== 'weather' && typeof Weather !== 'undefined') Weather.leave();
    view = name;
    renderGen++;
    mainGrid.reset();
    recosGrid.reset();
    setStatus(statusEl, '');
    $('empty-state').hidden = name !== 'empty';
    // A visitor who has collected nothing sees the landing alone, without the
    // menu; it comes in with the first search or prompt.
    document.body.classList.toggle('no-menu', name === 'empty' && !Collections.all().some(c => c.items.length));
    if (name === 'empty') {
        Filets.setAmbient(null);
        // Back on the landing: its lines were measured while it was hidden.
        requestAnimationFrame(() => alignWordmarkStar());
        if (typeof Landing !== 'undefined' && Landing.sets.length) requestAnimationFrame(() => {
            if (Landing.layout()) Landing.show(Landing.current, false);
            Landing.wake();
        });
    }
    document.body.classList.toggle('is-landing', name === 'empty');
    $('view-header').hidden = name !== 'collection';
    $('shared-header').hidden = name !== 'shared';
    $('weather-header').hidden = name !== 'weather';
    $('recos').hidden = true;
    if (name !== 'collection') viewingId = null;
    renderSidebarCollections();
    window.scrollTo({ top: 0, behavior: 'instant' });
}

// history: 'auto' adds an entry for a new search and updates the current one when
// only settings or filters change; 'replace' and 'none' never add one.
function runSearch(historyMode = 'auto') {
    query.text = searchInput.value.trim();
    if (!query.text && !query.refs.length) return;
    if (historyMode === 'auto') {
        const now = new URLSearchParams(location.search);
        const next = searchParams();
        const same = view === 'results' && now.get('q') === next.get('q') && now.get('ref') === next.get('ref');
        historyMode = same ? 'replace' : 'push';
    }
    updateUrl(historyMode);   // before showView, which scrolls to the top: the entry keeps the scroll position
    closeMenu();
    showView('results');
    results.request = buildRequest();
    results.offset = 0;
    results.hasMore = true;
    results.items = [];
    mainGrid.skeleton(14);
    loadPage(renderGen);
}

async function loadPage(gen) {
    if (!results.hasMore || results.loadingGen === gen) return;
    results.loadingGen = gen;
    if (results.offset > 0) setStatus(statusEl, t('status.loading'), true);
    const firstPage = results.offset === 0;
    Filets.loading(true);
    try {
        const data = await postJSON('/search', { ...results.request, offset: results.offset });
        if (gen !== renderGen) return;
        results.offset += data.results.length;
        results.hasMore = Boolean(data.has_more);
        const list = data.results.map(toItem).filter(it => !hiddenIds.has(it.id) && !(view === 'weather' && weatherExcluded(it.title)));
        await fillGrid(mainGrid, list, gen, 'results', it => results.items.push(it));
        if (firstPage && gen === renderGen) Filets.setAmbient(list[0]);   // the interface takes the colours of the search
        if (gen !== renderGen) return;
        const filtered = results.request.periods || results.request.techniques;
        setStatus(statusEl, mainGrid.items.length ? (results.hasMore ? '' : t('status.end'))
            : t(filtered ? 'status.noneFiltered' : 'status.none'));
    } catch (err) {
        console.error(err);
        if (gen === renderGen) {
            if (mainGrid.hasSkeleton) mainGrid.reset();
            setStatus(statusEl, t('status.error'));
        }
        results.hasMore = false;
    } finally {
        if (results.loadingGen === gen) results.loadingGen = null;
        Filets.loading(false);
    }
    if (gen === renderGen) maybeLoadMore();
}

// Works that never illustrate the weather (treatise plates came up for rain),
// by the start of their title: every plate of a series goes at once.
const WEATHER_EXCLUDED = ['A Treatise on Landscape Painting and Effect'];
const weatherExcluded = title => WEATHER_EXCLUDED.some(start => title.startsWith(start));

function maybeLoadMore() {
    if ((view !== 'results' && view !== 'weather') || !results.hasMore || results.loadingGen === renderGen) return;
    if ($('sentinel').getBoundingClientRect().top < window.innerHeight + 800) loadPage(renderGen);
}

new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) maybeLoadMore();
}, { rootMargin: '800px' }).observe($('sentinel'));

function backToResults(scrollY = 0) {
    if (!results.request) { showView('empty'); return; }
    showView('results');
    Nav.replace({ view: 'results' }, results.url || location.pathname + location.search);
    const gen = renderGen;
    const previous = results.items;
    results.items = [];
    fillGrid(mainGrid, previous, gen, 'results', it => results.items.push(it)).then(done => {
        if (!done) return;
        setStatus(statusEl, results.hasMore ? '' : t('status.end'));
        if (scrollY) window.scrollTo({ top: scrollY, behavior: 'instant' });
    });
}

function searchParams() {
    const params = new URLSearchParams();
    if (query.text) params.set('q', query.text);
    const ids = query.refs.filter(r => r.id != null).map(r => r.id);
    if (ids.length) params.set('ref', ids.join(','));
    const f = currentFilters();
    if (f.periods.length) params.set('period', f.periods.join(','));
    if (f.techniques.length) params.set('technique', f.techniques.join(','));
    return params;
}

function updateUrl(historyMode) {
    const qs = searchParams().toString();
    const url = qs ? `?${qs}` : location.pathname;
    if (historyMode === 'push') Nav.push({ view: 'results' }, url);
    else if (historyMode !== 'none') Nav.replace({ view: 'results' }, url);
    results.url = location.search;
}

async function restoreFromUrl() {
    const params = new URLSearchParams(location.search);
    if (params.get('c')) { openShared(params.get('c')); return; }
    if (params.has('meteo')) return;   // weather.js opens it
    Nav.replace({ view: 'empty' });
    if (params.get('i')) { openSharedImage(Number(params.get('i'))); return; }
    await searchFromUrl(params, 'replace');
}

// The search an address describes (first visit, or back/forward in history).
async function searchFromUrl(params, historyMode) {
    const text = params.get('q') || '';
    const ids = (params.get('ref') || '').split(',').filter(Boolean).map(Number)
        .filter(n => Number.isInteger(n) && n >= 0);
    setFilters((params.get('period') || '').split(','), (params.get('technique') || '').split(','));
    if (!text && !ids.length) return;
    searchInput.value = text;
    query.refs = [];
    query.negs = [];
    renderChips();
    if (ids.length) {
        try {
            const data = await postJSON('/cluster-members', { faiss_ids: ids });
            data.results.map(toItem).forEach(it => query.refs.push(refFromItem(it)));
            renderChips();
        } catch (err) { console.error('restore refs', err); }
    }
    runSearch(historyMode);
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
    if (Collections.contains(cid, item)) { toast(t('toast.already', { name: c.name })); return; }
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
        el('span', 'collection-kicker', t('collections.kicker')),
        el('span', 'collection-title'),
        el('span', 'collection-meta'),
    );
    btn.append(cover, text, el('span', 'collection-band'));
    btn.addEventListener('click', () => openCollection(c.id));
    // Phone: tapping a row opens it; this button only chooses where images go.
    const pick = el('button', 'collection-pick');
    pick.type = 'button';
    pick.addEventListener('click', () => setTarget(c.id, true));
    li.append(btn, pick);
    acceptsItemDrop(li, item => dropIntoCollection(c.id, item));
    return li;
}

function renderSidebarCollections() {
    const list = $('collection-list');
    const existing = new Map([...list.querySelectorAll('.collection-row')].map(li => [li.dataset.id, li]));

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
        li.querySelector('.collection-item').title = isTarget ? t('collections.open', { name: c.name }) : t('collections.openAndTarget', { name: c.name });
        const pick = li.querySelector('.collection-pick');
        pick.setAttribute('aria-pressed', String(isTarget));
        pick.setAttribute('aria-label', t('collections.select', { name: c.name }));
        const cover = Collections.cover(c);
        li.querySelector('.collection-cover').style.backgroundImage = cover ? cssUrl(cover.src) : '';
        paintCollectionBand(li.querySelector('.collection-band'), c);
    });
    existing.forEach(li => li.remove());

    MenuFold.auto();
}

acceptsItemDrop($('new-collection-btn'), item => {
    const c = Collections.create();
    setTarget(c.id, false);
    collectInto(c.id, item, null);
});

// ── Collection view + automatic suggestions ────────────────────────────────
function openCollection(id, record = true) {
    const c = Collections.get(id);
    if (!c) return;
    closeMenu();
    setTarget(id, false);
    viewingId = id;
    if (record) Nav.push({ view: 'collection', id });   // before showView resets the scroll
    showView('collection');
    Filets.setAmbient(Collections.cover(c));
    $('collection-name').value = c.name;
    $('collection-count').textContent = countLabel(c.items.length);
    $('back-btn').hidden = !results.request;
    $('collection-search-btn').disabled = !c.items.some(i => i.id != null);
    const gen = renderGen;
    if (!c.items.length) {
        setStatus(statusEl, '');   // the menu's note says how to fill it
    } else {
        mainGrid.skeleton(Math.min(c.items.length, 8));
        fillGrid(mainGrid, c.items, gen, 'collection').then(done => {
            if (done && !mainGrid.items.length) setStatus(statusEl, t('status.collectionGone'));
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
    $('recos-sub').textContent = t(refs.length === c.items.length ? 'recos.sub' : 'recos.subLatest', { count: countLabel(refs.length) });
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
            if (gen === renderGen) { recosGrid.reset(); setStatus(recosStatus, t('recos.unavailable')); }
            return;
        }
    }
    if (gen !== renderGen) return;
    const inCollection = new Set(c.items.map(itemKey));
    const fresh = list.filter(it => !inCollection.has(itemKey(it)) && !hiddenIds.has(it.id));
    const done = await fillGrid(recosGrid, fresh, gen, 'results');
    if (done && !recosGrid.items.length) setStatus(recosStatus, t('recos.none'));
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
    if (c.items.length > usable.length) toast(t('toast.latest', { n: usable.length }));
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
    if (!ids.length) { toast(t('toast.noZip')); return; }
    btn.disabled = true;
    btn.textContent = t('view.zipPreparing');
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
        a.download = `Glane Collection ${c.name.replace(/[<>:"/\\|?*]+/g, ' ').trim() || 'collection'}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        toast(missing ? t('toast.zipMissing', { n: missing }) : t('toast.zipReady'));
    } catch (err) {
        console.error(err);
        toast(t('toast.zipFailed'));
    } finally {
        btn.disabled = false;
        btn.textContent = t('view.zip');
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
        title: t('dialog.deleteTitle', { name: c.name }),
        text: c.items.length ? t('dialog.deleteText', { count: countLabel(c.items.length) }) : t('dialog.deleteEmpty'),
        confirm: t('dialog.deleteConfirm'),
        danger: true,
    });
    if (!ok) return;
    Collections.remove(c.id);
    toast(t('toast.deleted', { name: c.name }));
    backToResults();
});

$('back-btn').addEventListener('click', () => {
    Nav.push({ view: 'results' });
    backToResults();
});

// ── Sharing ─────────────────────────────────────────────────────────────────
const shareUrl = (key, value) => `${location.origin}${location.pathname}?${key}=${encodeURIComponent(value)}`;

// Phones get the system share sheet; elsewhere a dialog with the link and COPY.
async function shareLink(url, title, text) {
    if (navigator.share && window.matchMedia('(hover: none)').matches) {
        try { await navigator.share({ title, text, url }); return; }
        catch (err) { if (err.name === 'AbortError') return; }
    }
    openDialog(d => {
        d.append(el('div', 'dialog-title', t('share.title')), el('div', 'dialog-text', text));
        const input = el('input', 'share-link');
        input.readOnly = true;
        input.value = url;
        input.addEventListener('focus', () => input.select());
        d.appendChild(input);
        const foot = el('div', 'dialog-foot');
        const copy = dialogButton(t('share.copy'), 'pill-btn', async () => {
            try { await navigator.clipboard.writeText(url); }
            catch { input.select(); document.execCommand('copy'); }
            copy.textContent = t('share.copied');
        });
        foot.append(dialogButton(t('share.close'), 'link-btn', closeDialog));
        if (navigator.share) foot.append(dialogButton(t('share.send'), 'link-btn', () => navigator.share({ title, url }).catch(() => { })));
        foot.append(copy);
        d.appendChild(foot);
    });
}

$('collection-share-btn').addEventListener('click', async e => {
    const c = Collections.get(viewingId);
    if (!c.items.length) { toast(t('share.empty')); return; }
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
        if (Sync.on) { Sync.dirty.add(c.id); await Sync.flush(); }   // a live link needs the collection on the server
        const res = await api('/api/shares', { method: 'POST', body: { collection_id: c.id, name: c.name, items: toServer(c).items } });
        shareLink(shareUrl('c', res.slug), c.name, t(res.live ? 'share.collectionLive' : 'share.collectionSnapshot'));
    } catch (err) {
        console.error(err);
        toast(t('share.failed'));
    } finally {
        btn.disabled = false;
    }
});

let shared = null;   // collection opened from a ?c= link

async function openShared(slug) {
    closeMenu();
    showView('shared');
    Nav.replace({ view: 'shared', slug });
    const gen = renderGen;
    mainGrid.skeleton(8);
    try {
        shared = await api(`/api/shares/${encodeURIComponent(slug)}`);
    } catch (err) {
        mainGrid.reset();
        $('shared-header').hidden = true;
        setStatus(statusEl, t('shared.missing'));
        return;
    }
    if (gen !== renderGen) return;
    $('shared-name').textContent = shared.name;
    $('shared-count').textContent = countLabel(shared.items.length);
    $('shared-search-btn').disabled = !shared.items.some(i => i.id != null);
    fillGrid(mainGrid, shared.items, gen, 'results');
}

async function openSharedImage(id) {
    if (!Number.isInteger(id) || id < 0) return;
    try {
        const data = await postJSON('/cluster-members', { faiss_ids: [id] });
        const item = data.results.map(toItem)[0];
        if (item) openLightbox({ items: [item] }, item);
    } catch (err) { console.error(err); }
}

$('shared-save-btn').addEventListener('click', () => {
    if (!shared) return;
    const c = Collections.create(shared.name);
    [...shared.items].reverse().forEach(item => Collections.add(c.id, item));
    setTarget(c.id, false);
    repaintAll();
    toast(t('shared.saved', { name: c.name }));
});

$('shared-search-btn').addEventListener('click', () => {
    if (!shared) return;
    query.refs = shared.items.filter(i => i.id != null).slice(0, MAX_COLLECTION_REFS).map(refFromItem);
    query.negs = [];
    searchInput.value = '';
    renderChips();
    runSearch();
});

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
    if (!history.state?.lb) Nav.push({ ...history.state, lb: true });
    requestAnimationFrame(() => requestAnimationFrame(() => lightbox.classList.add('is-open')));
    $('lb-close').focus({ preventScroll: true });
}

// Resolves once the image's history entry is gone (navigate only after that).
function closeLightbox(fromHistory = false) {
    closePicker(true);
    lightbox.classList.remove('is-open');
    document.body.style.overflow = '';
    lbItem = null;
    lbCloseTimer = setTimeout(() => {
        lightbox.hidden = true;
        lbImg.removeAttribute('src');
    }, 300);
    const gone = !fromHistory && history.state?.lb ? Nav.back() : Promise.resolve();
    return gone.then(() => {
        if (new URLSearchParams(location.search).has('i')) Nav.replace(history.state || { view: 'empty' }, location.pathname);
    });
}

function paintLightboxActions() {
    const target = Collections.active();
    const collect = $('lb-collect');
    collect.classList.toggle('on', Boolean(target) && Collections.contains(target.id, lbItem));
    collect.title = target ? t('lb.collectInto', { name: target.name }) : t('lb.collectNew');

    $('lb-share').hidden = lbItem.id == null;

    // Choosing a cover only makes sense while browsing a collection.
    const coverBtn = $('lb-cover');
    const c = view === 'collection' && lbGrid === mainGrid ? Collections.get(viewingId) : null;
    coverBtn.hidden = !c || !Collections.contains(c.id, lbItem);
    $('lb-remove').hidden = coverBtn.hidden;
    if (!coverBtn.hidden) {
        const isCover = itemKey(Collections.cover(c) || {}) === itemKey(lbItem);
        coverBtn.classList.toggle('on', isCover);
        coverBtn.textContent = isCover ? t('lb.isCover') : t('lb.cover');
    }
}

function showInLightbox(item, withSimilar, direction) {
    lbItem = item;
    Filets.setAmbient(item);
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
    museum.title = item.url ? t('lb.museum') : '';
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
    $('lb-similar-label').textContent = t('lb.similar', { n: item.members.length - 1 });
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
$('lb-collect').addEventListener('click', e => {
    if (!lbItem) return;
    const ask = askWhereToCollect();   // before a first collection is made
    const cid = Collections.ensureTarget().id;
    if (Collections.contains(cid, lbItem) || ask) { e.stopPropagation(); openPicker(e.currentTarget, lbItem, null); return; }
    collectChosen();
    collectInto(cid, lbItem, null);
});
$('lb-cover').addEventListener('click', () => { if (lbItem) setCover(lbItem); });
// The image leaves the collection: the full page closes on it (Undo in the message).
$('lb-remove').addEventListener('click', () => {
    if (!lbItem) return;
    const item = lbItem;
    closeLightbox();
    removeItem(item, viewingId);
});
$('lb-share').addEventListener('click', () => { if (lbItem) shareLink(shareUrl('i', lbItem.id), lbItem.title, t('share.image')); });
$('lb-collect-menu').addEventListener('click', e => { e.stopPropagation(); if (lbItem) openPicker(e.currentTarget, lbItem, null); });
$('lb-visual').addEventListener('click', () => {
    if (!lbItem) return;
    const item = lbItem;
    closeLightbox().then(() => visualResearch(item));
});
lightbox.addEventListener('click', e => { if (e.target === lightbox || e.target.classList.contains('lb-stage')) closeLightbox(); });

document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !picker.hidden) { closePicker(); return; }
    if (dialog.open) return;
    if (lightbox.hidden || !lightbox.classList.contains('is-open')) {
        if (e.key === 'Escape' && document.body.classList.contains('menu-open')) dismissMenu();
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
// Each choice explains itself; the heading sums up what is active.
const VARIETY_KEYS = { '0': 'close', '0.3': 'balanced', '0.6': 'varied' };
const checkedValue = name => document.querySelector(`input[name="${name}"]:checked`).value;

function updateSettingsUI() {
    const variety = VARIETY_KEYS[checkedValue('variety')];
    const neg = checkedValue('negative-mode');
    // Combining only means something with at least two elements (words + image, or two images).
    const elements = (searchInput.value.trim() ? 1 : 0) + query.refs.length;
    const locked = elements < 2;
    $('combine-group').classList.toggle('is-locked', locked);
    $('combine-group').querySelectorAll('input').forEach(i => { i.disabled = locked; });
    // No image in the search bar: the choices stay hidden, a note on hover says why.
    const noImage = !query.refs.length;
    $('combine-group').hidden = noImage;
    $('combine-label').classList.toggle('is-unavailable', noImage);
    if (noImage) $('combine-label').dataset.tip = t('settings.combine.locked');
    else delete $('combine-label').dataset.tip;
    $('variety-help').textContent = t(`settings.variety.${variety}.help`);
    $('negative-help').textContent = t(`neg.${neg}.help`);
}

// ── Filters ─────────────────────────────────────────────────────────────────
$('filters-toggle').addEventListener('click', e => {
    const open = e.currentTarget.getAttribute('aria-expanded') !== 'true';
    e.currentTarget.setAttribute('aria-expanded', String(open));
    $('filters-panel').classList.toggle('open', open);
});

function renderFilterChips() {
    for (const [name, values, box] of [['period', PERIODS, 'period-filters'], ['technique', TECHNIQUES, 'technique-filters']]) {
        $(box).replaceChildren(...values.map(value => {
            const label = document.createElement('label');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.name = name;
            input.value = value;
            const span = document.createElement('span');
            span.textContent = t(`${name}.${value}`);
            label.append(input, span);
            return label;
        }));
    }
}

function setFilters(periods, techniques) {
    document.querySelectorAll('input[name="period"]').forEach(i => { i.checked = periods.includes(i.value); });
    document.querySelectorAll('input[name="technique"]').forEach(i => { i.checked = techniques.includes(i.value); });
    updateFiltersUI();
}

// The heading lists the active filters, so they stay visible with the panel closed.
function updateFiltersUI() {
    const f = currentFilters();
    const names = [...f.periods.map(v => t(`period.${v}`)), ...f.techniques.map(v => t(`technique.${v}`))];
    $('filters-summary').textContent = names.join(' · ');
    $('filters-clear').hidden = !names.length;
}

renderFilterChips();
$('filters-panel').addEventListener('change', e => {
    if (!e.target.matches('input[type="checkbox"]')) return;
    updateFiltersUI();
    if (view === 'results') runSearch();
});
$('filters-clear').addEventListener('click', () => {
    setFilters([], []);
    if (view === 'results') runSearch();
});

document.querySelectorAll('input[name="variety"], input[name="combination-mode"], input[name="negative-mode"]').forEach(r =>
    r.addEventListener('change', () => { updateSettingsUI(); if (view === 'results') runSearch(); }));
searchInput.addEventListener('input', updateSettingsUI);
// Help notes ("Drag an image onto a collection…", the settings' explanations):
// shown by default, hidden for the whole site once a visitor turns them off.
const hintsOff = () => { try { return localStorage.getItem('glane.hints') === 'off'; } catch { return false; } };
document.documentElement.classList.toggle('hints-off', hintsOff());
document.querySelector(`input[name="hints"][value="${hintsOff() ? 'off' : 'on'}"]`).checked = true;
document.querySelectorAll('input[name="hints"]').forEach(r => r.addEventListener('change', () => {
    try { if (r.value === 'off') localStorage.setItem('glane.hints', 'off'); else localStorage.removeItem('glane.hints'); } catch { /* private mode */ }
    document.documentElement.classList.toggle('hints-off', r.value === 'off');
}));
// Animations: on for everyone by default; "Reduced" stills the whole site (filets.js).
document.querySelector(`input[name="motion"][value="${Filets.motion() ? 'on' : 'off'}"]`).checked = true;
document.querySelectorAll('input[name="motion"]').forEach(r =>
    r.addEventListener('change', () => { Filets.setMotion(r.value === 'on'); updateSettingsUI(); }));

// ── Mobile menu ─────────────────────────────────────────────────────────────
function closeMenu() {
    document.body.classList.remove('menu-open');
    $('menu-btn').setAttribute('aria-expanded', 'false');
}
$('menu-btn').addEventListener('click', () => {
    const open = !document.body.classList.contains('menu-open');
    if (!open && history.state?.landingMenu) { history.back(); return; }   // opened from the first-visit landing
    document.body.classList.toggle('menu-open', open);
    if (open) requestAnimationFrame(() => alignWordmarkStar());   // the menu's stars, now on screen
    $('menu-btn').setAttribute('aria-expanded', String(open));
});

// Phone, first visit: the landing has no top bar, only the menu button, at the
// same place (to sign in from a new device, for instance). It opens the menu as
// a step of the history: closing it, or going back, without having done
// anything returns to the first-visit landing (showView puts no-menu back).
$('landing-menu').addEventListener('click', () => {
    Nav.push({ view: 'empty', landingMenu: true });
    document.body.classList.remove('no-menu');
    document.body.classList.add('menu-open');
    $('menu-btn').setAttribute('aria-expanded', 'true');
    window.dispatchEvent(new Event('resize'));   // the landing makes room for the top bar
    requestAnimationFrame(() => alignWordmarkStar());
});
// Closed by the visitor: back to where the menu was opened from.
function dismissMenu() {
    if (history.state?.landingMenu) history.back();   // popstate closes it and shows the landing again
    else closeMenu();
}

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

// Payloads always fit the server's limits (app/backend/accounts.py), so one odd
// image can never block a whole collection from saving.
const httpOrEmpty = u => (/^https?:\/\//.test(u || '') ? u.slice(0, 2048) : '');
const toServerItem = i => ({
    id: i.id ?? null,
    src: i.src.slice(0, 2048),
    hd: httpOrEmpty(i.hd),
    orig: httpOrEmpty(i.orig),
    url: httpOrEmpty(i.url),
    title: (i.title || '').slice(0, 500),
    author: (i.author || '').slice(0, 300),
    source: (i.source || '').slice(0, 40),
    members: Array.isArray(i.members) ? i.members.slice(0, 100) : null,
});
const toServer = c => ({
    name: (c.name || '').trim().slice(0, 120) || 'Collection',
    created_at: c.createdAt,
    updated_at: c.updatedAt || c.createdAt,
    cover_key: c.coverKey ? String(c.coverKey).slice(0, 2100) : null,
    items: c.items.filter(i => /^https?:\/\//.test(i.src)).slice(0, 2000).map(toServerItem),
});
const fromServer = c => ({
    id: c.id, name: c.name, createdAt: c.created_at, updatedAt: c.updated_at, coverKey: c.cover_key, items: c.items,
});


const Sync = {
    email: null,
    available: true,
    awaitingLink: false,      // a sign-in link was sent from this tab
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
    setState(state) { $('sync-status').textContent = t(`sync.${state}`); },

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
        toast(t('toast.sessionEnded'));
    },
};

function renderAccount() {
    $('account-out').hidden = Sync.on;
    $('account-in').hidden = !Sync.on;
    $('account-email').textContent = Sync.email || '';
    $('auth-form').hidden = !Sync.available;
    $('account-intro').textContent = Sync.available ? t('account.intro') : t('account.soon');
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
    btn.textContent = t('account.sending');
    try {
        // The guest's collections go with the request: the server adds them to the
        // account when the link is used, even in another browser or the mail app.
        const guest = Sync.on ? [] : Collections.all().slice(0, 50).map(c => ({ id: c.id, ...toServer(c) }));
        const res = await api('/auth/request', { method: 'POST', body: { email, guest_collections: guest } });
        Sync.awaitingLink = true;
        showAuthStatus(t('account.sent', { email }),
            res.dev_outbox ? { label: t('account.devOutbox'), href: res.dev_outbox } : null);
    } catch (err) {
        console.error(err);
        showAuthStatus(t({ 503: 'account.unavailable', 400: 'account.invalid', 422: 'account.invalid', 429: 'account.tooMany' }[err.status] || 'account.failed'));
    } finally {
        btn.disabled = false;
        btn.textContent = t('account.send');
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
    toast(t('toast.signedOut'));
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && (Sync.dirty.size || Sync.deleted.size)) Sync.flush(true);
    if (document.visibilityState === 'visible') checkLinkUsed();
});
window.addEventListener('focus', checkLinkUsed);

// The link was opened in another tab of this browser: sign this tab in too.
async function checkLinkUsed() {
    if (!Sync.awaitingLink || Sync.on) return;
    try {
        const s = await api('/auth/session');
        if (!s.authenticated) return;
        Sync.awaitingLink = false;
        Sync.email = s.email;
        $('auth-status').hidden = true;
        renderAccount();
        await Sync.pullAndMerge();
        toast(t('toast.signedIn', { email: Sync.email }));
    } catch (err) {
        console.error('session', err);
    }
}

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
    if (loginError) toast(t('toast.loginExpired'));
    if (!Sync.on) return;
    try {
        const uploaded = await Sync.pullAndMerge();
        if (signedIn) {
            toast(uploaded
                ? t('toast.signedInMerged', { email: Sync.email, n: uploaded })
                : t('toast.signedIn', { email: Sync.email }));
        }
    } catch (err) {
        if (err.status === 401) Sync.expired();
        else { console.error(err); Sync.setState('offline'); }
    }
}

// ── Landing: shelves of book spines made of stretched pixels, the text on top ─
// The images come from a static file (landing.json): no search runs on arrival,
// and only thumbnails are loaded. Each spine (taller than wide) shows the
// centre column of its image stretched across it (the browser stretches the
// thumbnail: nothing is read back). Hovering a spine makes its image spring
// out sideways, sliding from behind the spine over its neighbours like a book
// pulled from the shelf: the shelf's height, its true proportions, never
// distorted.
const LANDING_READY_TIMEOUT_MS = 6000;
const LANDING_STRETCH = 600;     // one source column fills the spine
const LANDING_GAP_X = 3;         // between spines
const LANDING_GAP_Y = 8;         // between shelves
const LANDING_ARRIVAL_MS = 3000;  // end of the arrival (style.css: the search bar, last, at 1.9 s + 0.9 s)

const Landing = {
    data: null,
    sets: [],            // [default mix, prompt 1, prompt 2, …]
    current: 0,
    cells: [],
    grid: { items: [] }, // lightbox navigation over the visible images
    swapTimer: null,
    layoutKey: '',

    async init() {
        // The landing's own search bar runs the same search as the menu's.
        $('landing-search').addEventListener('submit', e => {
            e.preventDefault();
            const q = $('landing-search-input').value.trim();
            if (!q) return;
            searchInput.value = q;
            $('landing-search-input').value = '';
            // The search opens the menu, where the bar now is, beside the images.
            MenuFold.touched = true;
            MenuFold.set(false);
            runSearch();
        });
        this.dropdown();
        // A click anywhere else (or Escape) tucks the open image back into its spine.
        document.addEventListener('pointerdown', e => { if (!e.target.closest('.lcell.is-open')) this.closeSpines(); });
        document.addEventListener('keydown', e => { if (e.key === 'Escape') this.closeSpines(); });
        const params = new URLSearchParams(location.search);
        if (params.has('q') || params.has('ref') || params.has('c') || params.has('meteo')) { this.finish(); return; }   // arriving on a search or a shared collection
        try {
            this.data = await (await fetch('landing.json')).json();
        } catch (err) {
            console.error('landing', err);
            this.finish();
            return;
        }
        const prompts = this.data.prompts;
        const mix = [];
        for (let round = 0; round < 10; round++) for (const p of prompts) if (p.items[round]) mix.push(p.items[round]);
        this.sets = [mix, ...prompts.map(p => p.items)];
        this.buildPrompts(prompts);
        document.body.classList.toggle('no-menu', view === 'empty' && !Collections.all().some(c => c.items.length));
        window.addEventListener('resize', () => {
            if ($('empty-state').hidden || !this.sets.length) return;
            if (this.layout()) this.show(this.current, false);
        });
        this.layout();
        await this.preload(this.sets[0].slice(0, Math.min(24, this.cells.length)), true);
        this.finish();
        this.show(0, false);
        // Warm the other sets so hovering a prompt swaps instantly.
        const rest = this.sets.slice(1).flat();
        (window.requestIdleCallback || setTimeout)(() => this.preload(rest, false));
    },

    // The prompts: a list that unfolds under the empty search bar when it is
    // clicked, and folds as soon as something is typed; arrows walk it,
    // Escape folds it.
    dropdown() {
        const foot = $('landing-search').closest('.landing-foot');
        const input = $('landing-search-input');
        const list = $('landing-prompts');
        const shown = () => [...list.querySelectorAll('.suggestion:not([hidden])')];
        let quiet = false;   // focus given back by Escape: the list stays folded
        const open = on => {
            on = on && shown().length > 0;
            foot.classList.toggle('prompts-open', on);
            input.setAttribute('aria-expanded', String(on));
        };
        input.addEventListener('focus', () => { if (!quiet) open(!input.value.trim()); quiet = false; });
        input.addEventListener('input', () => open(!input.value.trim()));
        input.addEventListener('click', () => open(!input.value.trim()));   // already focused: no focus event
        document.addEventListener('pointerdown', e => { if (!foot.contains(e.target)) open(false); });
        foot.addEventListener('focusout', e => { if (e.relatedTarget && !foot.contains(e.relatedTarget)) open(false); });
        foot.addEventListener('keydown', e => {
            if (e.key === 'Escape') { open(false); if (document.activeElement !== input) { quiet = true; input.focus(); } return; }
            if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
            e.preventDefault();
            open(true);
            const all = shown();
            const i = all.indexOf(document.activeElement);
            const down = e.key === 'ArrowDown';
            if (i < 0) (down ? all[0] : all.at(-1))?.focus();
            else if (!down && i === 0) input.focus();
            else all[(i + (down ? 1 : -1)) % all.length].focus();
        });
    },

    buildPrompts(prompts) {
        const box = $('landing-prompts');
        prompts.forEach((p, i) => {
            const b = el('button', 'suggestion', promptLabel(p.label));
            b.type = 'button';
            b.style.setProperty('--i', i);
            b.addEventListener('click', () => { searchInput.value = sentenceCase(p.label); runSearch(); });
            if (window.matchMedia('(hover: hover)').matches) {
                b.addEventListener('mouseenter', () => this.preview(i + 1, b));
                b.addEventListener('focus', () => this.preview(i + 1, b));
            }
            box.appendChild(b);
        });
    },

    preload(items, track) {
        const bar = $('landing-progress-bar');   // gone from the page: kept optional
        let done = 0;
        const all = items.map(it => preload(it).then(() => {
            done++;
            if (track && bar) bar.style.setProperty('--p', done / items.length);
        }));
        const everything = Promise.all(all);
        return track ? Promise.race([everything, new Promise(r => setTimeout(r, LANDING_READY_TIMEOUT_MS))]) : everything;
    },

    finish() {
        $('empty-state').classList.remove('is-loading');
        setTimeout(() => $('empty-state').classList.add('has-arrived'), LANDING_ARRIVAL_MS);
    },

    // Shelves of spines filling the whole landing. Returns true when the spines
    // had to be rebuilt (their number changed).
    layout() {
        const field = $('landing-field');
        const box = field.parentElement.getBoundingClientRect();   // the landing: the field may start lower
        if (!box.height) return false;
        const W = box.width, H = box.height;
        // The first shelves stay empty, for the title and the text: one on a
        // computer; on a phone two for a first visit (the text runs under the
        // title), one once the top bar has the logo.
        const freeShelves = innerWidth > 860 || !document.body.classList.contains('no-menu') ? 1 : 2;
        // The window sets how many spines there are; the menu folding or
        // unfolding only makes them wider or narrower.
        const win = `${innerWidth}x${innerHeight}`;
        const sameWindow = this.cols && win === this.win;
        this.win = win;
        const cols = sameWindow ? this.cols : Math.max(8, Math.min(48, Math.round(W / (W < 700 ? 34 : 44))));
        this.W = W;
        this.cols = cols;
        this.cw = (W - LANDING_GAP_X * (cols - 1)) / cols;
        if (!sameWindow) this.shelves = Math.min(6, Math.round((H + LANDING_GAP_Y) / (this.cw * 5 + LANDING_GAP_Y)));
        let rows = Math.max(freeShelves + 2, this.shelves);
        this.ch = (H - LANDING_GAP_Y * (rows - 1)) / rows;
        const shelf = freeShelves * (this.ch + LANDING_GAP_Y);
        rows -= freeShelves;
        field.style.top = shelf ? `${shelf}px` : '';
        field.parentElement.style.setProperty('--shelf', `${shelf}px`);
        this.shelf = shelf;
        this.fitTitle();
        field.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        field.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
        field.style.gap = `${LANDING_GAP_Y}px ${LANDING_GAP_X}px`;
        const key = `${cols}/${rows}/${shelf ? 1 : 0}`;
        if (key === this.layoutKey) { this.cells.forEach(c => this.stretch(c)); return false; }
        this.layoutKey = key;
        field.replaceChildren();
        this.cells = Array.from({ length: cols * rows }, (_, i) => {
            const cell = el('button', 'lcell');
            cell.type = 'button';
            cell._col = i % cols;
            cell.style.setProperty('--d', `${(cell._col + Math.floor(i / cols) * 3) * 25}ms`);
            const img = new Image();
            img.alt = '';
            img.decoding = 'async';
            img.className = 'lcell-photo';
            const drawer = el('span', 'lcell-drawer');
            drawer.appendChild(img);
            cell.append(el('span', 'lcell-band'), drawer);
            // A first click pulls the image out of its spine; a click on it,
            // once out, shows it full page.
            cell.addEventListener('click', () => {
                if (!cell._item) return;
                if (cell.classList.contains('is-open')) { openLightbox(this.grid, cell._item); return; }
                this.closeSpines();
                cell.classList.add('is-open');
            });
            field.appendChild(cell);
            return cell;
        });
        return true;
    },

    // The spine: the image's centre column stretched across it, the image's
    // full height on the shelf's height. The drawer: right beside the spine
    // (on its left near the field's right edge), as wide as the image at the
    // shelf's height; the photo waits inside it, tucked behind the spine.
    closeSpines() { this.cells?.forEach(c => c.classList.remove('is-open')); },

    // Computer: the title as tall as the text beside it (within its shelf).
    fitTitle(shelf = this.shelf || 0) {
        const title = document.querySelector('.landing .wordmark');
        const text = document.querySelector('.landing-tagline');
        title.style.fontSize = '';
        if (innerWidth <= 860 || !shelf || !text.offsetHeight || !title.offsetHeight) return;
        const size = parseFloat(getComputedStyle(title).fontSize);
        const height = Math.min(text.offsetHeight, shelf * 0.8);
        title.style.fontSize = `${(size * height / title.offsetHeight).toFixed(1)}px`;
    },

    stretch(cell) {
        const item = cell._item;
        const band = cell.querySelector('.lcell-band');
        if (!item) { band.style.backgroundImage = ''; return; }
        // In % of the spine: it follows the spine's width while the menu moves.
        band.style.backgroundImage = cssUrl(item.src);
        band.style.backgroundSize = `${LANDING_STRETCH * 100}% 100%`;
        band.style.backgroundPosition = '50% 0';
        const w = Math.min(this.ch / (item.ratio || 1.2), this.W * 0.6);
        const left = cell._col * (this.cw + LANDING_GAP_X);
        const toLeft = left + this.cw + w > this.W;
        const drawer = cell.querySelector('.lcell-drawer');
        drawer.classList.toggle('to-left', toLeft);
        drawer.style.width = `${w.toFixed(1)}px`;
    },

    assign(cell, item) {
        cell._item = item;
        cell.classList.remove('is-ready');
        if (!item) { cell.querySelector('.lcell-band').style.backgroundImage = ''; return; }
        cell.title = item.title;
        cell.setAttribute('aria-label', t('card.open', { title: item.title }));
        const img = cell.querySelector('img');
        img.onload = () => { if (cell._item === item) cell.classList.add('is-ready'); };
        img.onerror = () => { if (item.orig && img.src !== item.orig) img.src = item.orig; };
        img.src = item.src;
        this.stretch(cell);
        if (img.complete && img.naturalWidth) cell.classList.add('is-ready');
    },

    show(setIndex, animate) {
        const set = this.sets[setIndex] || [];
        this.current = setIndex;
        const items = set.slice(0, this.cells.length);
        for (const extra of this.sets[0]) if (items.length < this.cells.length && !items.includes(extra)) items.push(extra);
        this.grid.items = items.slice();
        for (let i = 0; items.length < this.cells.length && this.grid.items.length; i++) items.push(this.grid.items[i % this.grid.items.length]);
        this.cells.forEach((cell, i) => {
            if (!animate || !Filets.motion()) { this.assign(cell, items[i]); return; }
            // A new set spreads across the field from the top left corner.
            setTimeout(() => {
                cell.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140, easing: 'ease-in' }).onfinish = () => {
                    this.assign(cell, items[i]);
                    cell.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 360, easing: 'ease-out' });
                };
            }, parseFloat(cell.style.getPropertyValue('--d')) * 0.5);
        });
    },

    wake() {},   // kept for showView(): nothing runs continuously any more

    preview(setIndex, button) {
        document.querySelectorAll('#landing-prompts .suggestion').forEach(b => b.classList.toggle('is-previewing', b === button));
        if (setIndex === this.current) return;
        clearTimeout(this.swapTimer);
        this.swapTimer = setTimeout(() => this.show(setIndex, true), 90);
    },
};

// ── Notes on hover ─────────────────────────────────────────────────────────
// The browser's own tooltips (title="…") are replaced by the site's: dark,
// small, with a line of pixels at their base. Shown after a short pause,
// gone on press, scroll or leave. Touch screens keep none (no hover).
const Tip = (() => {
    const tip = $('tip'), text = $('tip-text');
    let target = null, timer = null;
    const hide = () => { clearTimeout(timer); timer = null; target = null; tip.classList.remove('is-on'); };
    function place() {
        if (!target) return;
        const r = target.getBoundingClientRect();
        tip.hidden = false;
        const w = tip.offsetWidth, h = tip.offsetHeight;
        const above = r.top - h - 8 > 4;
        const x = Math.min(window.innerWidth - w - 6, Math.max(6, r.left + r.width / 2 - w / 2));
        tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(above ? r.top - h - 8 : r.bottom + 8)}px)`;
        tip.classList.add('is-on');
    }
    if (window.matchMedia('(hover: hover)').matches) {
        document.addEventListener('pointerover', e => {
            const node = e.target.closest?.('[title], [data-tip]');
            if (!node || node === target) return;
            if (node.hasAttribute('title')) {   // code may set title again later: always take the latest
                const v = node.getAttribute('title');
                node.removeAttribute('title');
                if (v) node.dataset.tip = v; else delete node.dataset.tip;
            }
            if (!node.dataset.tip) return;
            hide();
            target = node;
            text.textContent = node.dataset.tip;
            timer = setTimeout(place, 380);
        });
        document.addEventListener('pointerout', e => { if (target && !target.contains(e.relatedTarget)) hide(); });
        document.addEventListener('pointerdown', hide, true);
        window.addEventListener('scroll', hide, { passive: true, capture: true });
    }
    return { hide };
})();

// ── Stars (Flor de Ruina asterisks) ─────────────────────────────────────────
// Each weight of Flor de Ruina draws its asterisk at another place in its em
// box: stacked on one origin the wordmark's five stars would jump about, and a
// menu star would sit off the line. Measure the ink of each glyph (canvas
// metrics, from its baseline in the page) and shift it so its ink is centred on
// its box: the wordmark's star thickens in place, the menu stars sit on the row.
async function alignWordmarkStar() {
    for (const star of document.querySelectorAll('.wm-star, .star-box')) await alignStar(star);
}
async function alignStar(star) {
    if (!star.getClientRects().length) return;
    const glyphs = [...star.children];
    const size = parseFloat(getComputedStyle(glyphs[0]).fontSize);
    const families = glyphs.map(g => getComputedStyle(g).fontFamily.split(',')[0].trim().replace(/["']/g, ''));
    await Promise.all(families.map(f => document.fonts.load(`${size}px "${f}"`, '*').catch(() => {})));
    const ctx = document.createElement('canvas').getContext('2d');
    const box = star.getBoundingClientRect();
    const cx = box.left + box.width / 2, cy = box.top + box.height / 2;
    glyphs.forEach((g, i) => {
        g.style.transform = '';
        ctx.font = `${size}px "${families[i]}"`;
        const m = ctx.measureText('*');
        // The glyph's line is one em (line-height: 1): the font's ascent and
        // descent sit centred in it, which gives the baseline.
        const r = g.getBoundingClientRect();
        const origin = r.left;
        const baseline = r.top + (size - (m.fontBoundingBoxAscent + m.fontBoundingBoxDescent)) / 2 + m.fontBoundingBoxAscent;
        const inkX = origin + (m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2;
        const inkY = baseline + (m.actualBoundingBoxDescent - m.actualBoundingBoxAscent) / 2;
        g.style.transform = `translate(${((cx - inkX) / size).toFixed(3)}em, ${((cy - inkY) / size).toFixed(3)}em)`;
    });
}
window.addEventListener('resize', () => alignWordmarkStar());

// ── Menu folded behind the logo's star (computer) ───────────────────────────
// Folded while there is no collection (nothing to show in it yet), open as
// soon as there is one. The star, the logo's own, folds and unfolds it: open,
// it stands after "Glane"; folded, it waits alone at the top left. Once the
// visitor has used it, the menu never changes by itself again.
const MENU_MS = 420;   // as --t-slow, the sidebar's slide
const MenuFold = {
    star: $('menu-star'),
    touched: false,
    settled: 0,   // when the running fold or unfold ends
    get folded() { return document.body.classList.contains('menu-folded'); },
    set(folded) {
        if (folded !== this.folded) {
            document.body.classList.toggle('menu-folded', folded);
            this.settled = Date.now() + MENU_MS;
            // The grids and the landing measure their width again once the menu has moved.
            setTimeout(() => window.dispatchEvent(new Event('resize')), MENU_MS + 20);
        }
        this.star.setAttribute('aria-expanded', String(!folded));
        this.star.dataset.tip = t(folded ? 'menu.show' : 'menu.hide');
        this.star.setAttribute('aria-label', this.star.dataset.tip);
    },
    auto() { if (!this.touched) this.set(!Collections.all().length); },
    // Open, the star sits where the logo's would: right after "Glane".
    place() {
        const brand = document.querySelector('.brand');
        this.star.style.setProperty('--open-x', `${brand.offsetLeft + brand.offsetWidth}px`);
        this.star.style.setProperty('--open-y', `${brand.offsetTop}px`);
    },
};
// The logo leads home; already there, a click does nothing (no reload).
document.querySelector('.brand').addEventListener('click', e => { if (view === 'empty') e.preventDefault(); });
MenuFold.star.addEventListener('click', () => { MenuFold.touched = true; MenuFold.set(!MenuFold.folded); });
window.addEventListener('resize', () => MenuFold.place());

// ── Init ────────────────────────────────────────────────────────────────────
// Reveal the page once Junicode and the title face are ready (at most 3 s on a slow network).
Promise.race([
    Promise.all(['400 1em Junicode', 'italic 400 1em Junicode', '600 1em Junicode', '400 1em "Flor Semilla"'].map(f => document.fonts.load(f))),
    new Promise(resolve => setTimeout(resolve, 3000)),
]).finally(() => { document.documentElement.classList.remove('is-booting'); MenuFold.place(); Landing.fitTitle(); alignWordmarkStar(); });

applyI18n();
Collections.load();
mainGrid.build();
recosGrid.build();
renderSidebarCollections();
renderChips();
restoreFromUrl();
initAccount();
Landing.init();
