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
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

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
const countLabel = n => `${n} image${n === 1 ? '' : 's'}`;
const cssUrl = src => `url("${encodeURI(src)}")`;

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
        if (!data.list.length) data.list.push(this.blank('My collection'));
        if (!data.list.some(c => c.id === data.activeId)) data.activeId = data.list[0].id;
        this.data = data;
        this.save();
    },
    blank(name) { return { id: uid(), name, items: [] }; },
    save() { store.set('glane.collections', this.data); },
    all() { return this.data.list; },
    get(id) { return this.data.list.find(c => c.id === id); },
    active() { return this.get(this.data.activeId); },
    setActive(id) { this.data.activeId = id; this.save(); },
    contains(id, item) { const k = itemKey(item); return this.get(id).items.some(i => itemKey(i) === k); },
    add(id, item) {
        if (this.contains(id, item)) return false;
        this.get(id).items.unshift({ ...item });
        this.save();
        return true;
    },
    removeItem(id, item) {
        const c = this.get(id);
        const k = itemKey(item);
        const idx = c.items.findIndex(i => itemKey(i) === k);
        if (idx < 0) return -1;
        c.items.splice(idx, 1);
        this.save();
        return idx;
    },
    insertAt(id, item, idx) {
        const c = this.get(id);
        if (!this.contains(id, item)) c.items.splice(Math.max(0, idx), 0, { ...item });
        this.save();
    },
    create(name) {
        const c = this.blank(name || `Collection ${this.data.list.length + 1}`);
        this.data.list.unshift(c);
        this.save();
        return c;
    },
    rename(id, name) { const c = this.get(id); if (c && name.trim()) { c.name = name.trim(); this.save(); } },
    remove(id) {
        this.data.list = this.data.list.filter(c => c.id !== id);
        if (!this.data.list.length) this.data.list.push(this.blank('My collection'));
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
            if (!animate || reduceMotion.matches) { node.remove(); return; }
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
async function fillGrid(grid, list, gen, mode, onPlaced) {
    const loads = list.map(it => preload(it.src));
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
    img.addEventListener('error', () => grid.remove(card, false));
    img.addEventListener('dragstart', e => {
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('application/x-glane-item', JSON.stringify(item));
        e.dataTransfer.setData('text/uri-list', item.src);
    });
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
    const name = Collections.get(cid).name;
    btn.classList.toggle('on', on);
    btn.innerHTML = `${on ? ICON.collected : ICON.collect}<span class="card-btn-label">${on ? 'COLLECTED' : 'COLLECT'}</span>`;
    btn.title = on ? `In “${name}”: remove or move` : `Add to “${name}”`;
    btn.setAttribute('aria-label', btn.title);
}

function repaintAll() {
    document.querySelectorAll('.card').forEach(card => { if (card.querySelector('.card-collect')) paintCollectButton(card); });
    if (lbItem) $('lb-collect').classList.toggle('on', Collections.contains(Collections.data.activeId, lbItem));
    renderSidebarCollections();
    if (view === 'collection') $('collection-count').textContent = countLabel(Collections.get(viewingId).items.length);
}

function onCollectClick(card) {
    const cid = cardCollectionId(card);
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
    if (cid === Collections.data.activeId) flyToTarget(fromImg);
    if (view === 'collection' && viewingId === cid) {
        const recoCard = recosGrid.cardFor(item);
        if (recoCard) recosGrid.remove(recoCard);
        const ratio = recoCard ? parseFloat(recoCard.querySelector('.card-img').style.aspectRatio.split('/')[1]) : 1;
        mainGrid.add(createCard(item, clampRatio(ratio), mainGrid, 'collection'), clampRatio(ratio), item);
        setStatus(statusEl, '');
    }
    repaintAll();
}

// Keeps the grid of the collection being viewed in sync after a removal.
function afterRemoval(item, cid) {
    if (view === 'collection' && viewingId === cid) {
        const card = mainGrid.cardFor(item);
        if (card) mainGrid.remove(card);
        if (!Collections.get(cid).items.length) setStatus(statusEl, 'Empty for now. Use COLLECT on any image to add it here.');
    }
}

function flyToTarget(fromImg) {
    const dest = $('target-cover');
    const bumpTarget = () => {
        const card = $('target-card');
        card.classList.remove('bump');
        void card.offsetWidth;
        card.classList.add('bump');
    };
    const b = dest.getBoundingClientRect();
    if (!fromImg || reduceMotion.matches || b.height === 0 || b.bottom < 0 || b.top > window.innerHeight) {
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
    if (immediate || reduceMotion.matches) { picker.hidden = true; return; }
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
    renderSidebarCollections();
    const card = $('target-card');
    card.classList.remove('bump');
    void card.offsetWidth;
    card.classList.add('bump');
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
    if (reduceMotion.matches) { dialog.close(); return; }
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

// ── Sidebar: target collection + other collections ─────────────────────────
let lastTargetCount = null;

function renderSidebarCollections() {
    const target = Collections.active();
    $('target-name').textContent = target.name;
    const countEl = $('target-count');
    countEl.textContent = countLabel(target.items.length);
    if (lastTargetCount !== null && lastTargetCount !== target.items.length) {
        countEl.classList.remove('bump');
        void countEl.offsetWidth;
        countEl.classList.add('bump');
    }
    lastTargetCount = target.items.length;

    const cover = $('target-cover');
    const covers = target.items.slice(0, 4);
    cover.classList.toggle('single', covers.length === 1);
    const key = covers.map(itemKey).join('|');
    if (cover.dataset.key !== key) {
        cover.dataset.key = key;
        cover.replaceChildren(...covers.map(it => {
            const s = el('span');
            s.style.backgroundImage = cssUrl(it.src);
            return s;
        }));
    }

    const list = $('collection-list');
    list.replaceChildren();
    Collections.all().filter(c => c.id !== target.id).forEach((c, i) => {
        const li = el('li');
        li.style.setProperty('--i', i);
        const btn = el('button', 'collection-item');
        btn.type = 'button';
        btn.title = `Open “${c.name}”`;
        const cv = el('span', 'collection-cover');
        if (c.items[0]) cv.style.backgroundImage = cssUrl(c.items[0].src);
        const text = el('span', 'collection-text');
        text.append(el('span', 'collection-title', c.name), el('span', 'collection-meta', countLabel(c.items.length)));
        btn.append(cv, text);
        btn.addEventListener('click', () => openCollection(c.id));
        li.appendChild(btn);
        list.appendChild(li);
    });
}

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

$('target-open').addEventListener('click', () => openCollection(Collections.data.activeId));
$('target-change').addEventListener('click', e => { e.stopPropagation(); openPicker(e.currentTarget, null); });

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
    }, reduceMotion.matches ? 0 : 300);
}

function showInLightbox(item, withSimilar, direction) {
    lbItem = item;
    lbImg.getAnimations().forEach(a => a.cancel());
    lbImg.style.opacity = '0';
    lbImg.onload = () => {
        lbImg.style.opacity = '';
        if (direction && !reduceMotion.matches) {
            lbImg.animate([
                { opacity: 0, transform: `translateX(${direction * 28}px) scale(0.985)` },
                { opacity: 1, transform: 'none' },
            ], { duration: 480, easing: 'cubic-bezier(0.16, 1, 0.3, 1)' });
        } else if (!reduceMotion.matches) {
            lbImg.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, easing: 'ease' });
        }
    };
    lbImg.src = item.src;
    lbImg.alt = item.title;
    $('lb-title').textContent = item.title;
    $('lb-author').textContent = item.author;
    const museum = $('lb-museum');
    museum.textContent = museumName(item.source);
    if (item.url) museum.href = item.url; else museum.removeAttribute('href');
    museum.title = item.url ? 'See this artwork on the museum website' : '';
    $('lb-download').href = item.src;
    $('lb-collect').classList.toggle('on', Collections.contains(Collections.data.activeId, item));
    $('lb-collect').title = `Collect into “${Collections.active().name}”`;
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
    const cid = Collections.data.activeId;
    if (Collections.contains(cid, lbItem)) openItemDialog(lbItem, cid);
    else collectInto(cid, lbItem, null);
});
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

// ── Account (magic link; accounts are wired in lot 2) ───────────────────────
$('auth-form').addEventListener('submit', async e => {
    e.preventDefault();
    const email = $('auth-email').value.trim();
    const btn = $('auth-btn');
    const statusNote = $('auth-status');
    btn.disabled = true;
    try {
        await postJSON('/auth/request', { email });
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
mainGrid.build();
recosGrid.build();
renderSidebarCollections();
renderChips();
restoreFromUrl();
