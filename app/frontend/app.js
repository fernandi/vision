const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const gallery = document.getElementById('gallery');
const diversityToggle = document.getElementById('diversity-toggle');
const filterBtn = document.getElementById('filter-btn');
const filterPanel = document.getElementById('filter-panel');
const suggestionBtns = document.querySelectorAll('.suggestion-btn');

const API_URL = "";
const PAGE_SIZE = 20;

// Current diversity value: read from toggle state
function getDiversity() {
    return diversityToggle.getAttribute('aria-pressed') === 'true' ? 0.5 : 0.0;
}

// Current combination mode: read from radio group
function getCombinationMode() {
    const checked = document.querySelector('input[name="combination-mode"]:checked');
    return checked ? checked.value : 'purified';
}

// Negative mode
function getNegativeMode() {
    const checked = document.querySelector('input[name="negative-mode"]:checked');
    return checked ? checked.value : 'directed';
}

// ── State ────────────────────────────────────────────────────────────────────
let currentQuery = "";
let currentOffset = 0;
let isLoading = false;
let hasMore = true;
// Multimodal: array of { src, base64 } objects
let referenceImages = [];
// Negative images: array of { src, base64 } objects
let referenceNegativeImages = [];
// Snapshot for undo-clear
let clearSnapshot = null;
let undoClearTimer = null;
// Set of imgPaths the user has hidden (ne plus voir)
const hiddenImgPaths = new Set();
// Boards: archived via NEW button (persisted in localStorage)
let archivedBoards = JSON.parse(localStorage.getItem('archivedBoards') || '[]');

// ── Saved images ────────────────────────────────────────────────────────────
// Map: imgPath → { title, author, source, memberIds, clusterSize }
const savedItems = new Map();
const savedPanel = document.getElementById('saved-panel');
const savedThumbs = document.getElementById('saved-thumbs');

const ICON_EXPAND = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>`;
const ICON_PIN_EMPTY = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76Z"/></svg>`;
const ICON_PIN_FILLED = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17v5" fill="none"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76Z"/></svg>`;
const ICON_EYE_SLASH = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`;
let currentLbPath = ''; // tracks the image currently shown in the lightbox

// ── Image chips DOM ref ─────────────────────────────────────────────────────────
const searchContainer = document.getElementById('search-container');
const chipsContainer = document.getElementById('image-chips-container');

// ── Sentinel: lives inside the gallery, always at the bottom ───────────────────
let sentinel = null;

const observer = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting && !isLoading && hasMore && currentQuery) {
        loadNextPage();
    }
}, { rootMargin: '300px' });

function placeSentinel() {
    if (sentinel) { observer.unobserve(sentinel); sentinel.remove(); }
    sentinel = document.createElement('div');
    sentinel.id = 'scroll-sentinel';
    gallery.appendChild(sentinel);
    observer.observe(sentinel);
}

// ── Core search / pagination ───────────────────────────────────────────────────
async function search(query) {
    if (!query) return;
    currentQuery = query;
    currentOffset = 0;
    hasMore = true;

    document.body.classList.add('has-results');
    gallery.innerHTML = '<div class="placeholder-text">Searching...</div>';

    await loadNextPage(true);
}

async function loadNextPage(firstPage = false) {
    if (isLoading || !hasMore) return;
    isLoading = true;

    let loader = null;
    if (!firstPage) {
        loader = document.createElement('div');
        loader.className = 'load-more-spinner';
        loader.innerHTML = '<span></span><span></span><span></span>';
        if (sentinel) gallery.insertBefore(loader, sentinel);
        else gallery.appendChild(loader);
    }

    try {
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: currentQuery,
                page_size: PAGE_SIZE,
                offset: currentOffset,
                diversity: getDiversity(),
                combination_mode: getCombinationMode(),
                ...(referenceImages.length
                    ? { reference_images: referenceImages.map(r => r.base64), image_weight: 0.5 }
                    : {}),
                ...(referenceNegativeImages.length
                    ? {
                        negative_images: referenceNegativeImages.map(r => r.base64),
                        negative_mode: getNegativeMode()
                    }
                    : {}),
            }),
        });

        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();

        if (firstPage) gallery.innerHTML = '';

        if (firstPage && data.results.length === 0) {
            gallery.innerHTML = '<div class="placeholder-text">No results found.</div>';
            hasMore = false;
            return;
        }

        appendCards(data.results);
        currentOffset += data.results.length;
        hasMore = data.has_more;
        placeSentinel();

    } catch (err) {
        if (firstPage) {
            gallery.innerHTML = `<div class="placeholder-text">Error: ${err.message}</div>`;
        }
        console.error(err);
    } finally {
        if (loader) loader.remove();
        isLoading = false;
    }
}

// ── Render helpers ────────────────────────────────────────────────────────────
function appendCards(results) {
    if (sentinel) sentinel.remove();

    results.forEach(item => {
        const card = document.createElement('div');
        card.className = 'image-card';
        const imgPath = item.image_url || `/images/${item.filename}`;
        const titleRaw = item.Title || 'Untitled';
        const title = titleRaw.length > 140 ? titleRaw.slice(0, 140) + '…' : titleRaw;
        const hasGroup = item.cluster_size != null && item.cluster_size > 1
            && Array.isArray(item.cluster_member_ids);
        const simCount = hasGroup ? item.cluster_size - 1 : 0;

        const titleHTML = hasGroup
            ? `<div class="title title-with-similar">
                 <span>${title}  +${simCount}</span>
               </div>`
            : `<div class="title">${title}</div>`;

        // Image wrapped in positioning context for action buttons
        card.innerHTML = `
            <div class="card-img-wrap">
                <img src="${imgPath}" alt="${title}" loading="lazy">
            </div>
            <div class="image-info">
                ${titleHTML}
                <div class="author">${item.Author || 'Unknown'}</div>
            </div>
        `;

        // ── Action buttons (bottom-right of image) ───────────────────────────
        const imgWrap = card.querySelector('.card-img-wrap');
        const actions = document.createElement('div');
        actions.className = 'image-card-actions';

        // Expand → lightbox
        const expandBtn = document.createElement('button');
        expandBtn.className = 'card-action-btn card-action-expand';
        expandBtn.title = 'Agrandir';
        expandBtn.innerHTML = ICON_EXPAND;
        expandBtn.addEventListener('click', e => {
            e.stopPropagation();
            openLightbox(imgPath, title, item.Author, item.source,
                hasGroup ? item.cluster_member_ids : null,
                hasGroup ? item.cluster_size : null);
        });

        // Save → saved panel
        const saveBtn = document.createElement('button');
        saveBtn.className = 'card-action-btn card-action-save';
        saveBtn.dataset.src = imgPath;
        saveBtn.title = 'Épingler';
        const isSaved = savedItems.has(imgPath);
        saveBtn.innerHTML = isSaved ? ICON_PIN_FILLED : ICON_PIN_EMPTY;
        if (isSaved) saveBtn.classList.add('saved');
        saveBtn.addEventListener('click', e => {
            e.stopPropagation();
            toggleSaved(imgPath, title, item.Author, item.source,
                hasGroup ? item.cluster_member_ids : null,
                hasGroup ? item.cluster_size : null);
        });

        // Hide → 'ne plus voir' overlay
        const hideBtn = document.createElement('button');
        hideBtn.className = 'card-action-btn card-action-hide';
        hideBtn.title = 'Ne plus voir';
        hideBtn.innerHTML = ICON_EYE_SLASH;
        hideBtn.addEventListener('click', e => {
            e.stopPropagation();
            showHideOverlay(card, imgWrap, imgPath, title, item.Author, item.source,
                hasGroup ? item.cluster_member_ids : null,
                hasGroup ? item.cluster_size : null);
        });

        actions.appendChild(expandBtn);
        actions.appendChild(saveBtn);
        actions.appendChild(hideBtn);
        imgWrap.appendChild(actions);

        // ── Draggable ────────────────────────────────────────────────────────────
        const img = card.querySelector('img');
        img.draggable = true;
        img.addEventListener('dragstart', e => {
            e.dataTransfer.effectAllowed = 'copy';
            e.dataTransfer.setData('text/uri-list', imgPath);
            e.dataTransfer.setData('text/plain', imgPath);
        });



        gallery.appendChild(card);
    });
}

// ── Lightbox (single image + optional similar strip) ──────────────────────────
const lightbox = document.getElementById('lightbox');
const lbImg = document.getElementById('lightbox-img');
const lbTitle = document.getElementById('lightbox-title');
const lbAuthor = document.getElementById('lightbox-author');
const lbMuseum = document.getElementById('lightbox-museum');
const lbDownload = document.getElementById('lightbox-download');
const lbPin = document.getElementById('lightbox-pin');
const lbClose = document.querySelector('#lightbox .lightbox-close');
const lbSimilaires = document.getElementById('lb-similaires');
const lbSimLabel = document.getElementById('lb-sim-label');
const lbSimStrip = document.getElementById('lb-sim-strip');

const MUSEUM_NAMES = {
    artic: 'ART INSTITUTE OF CHICAGO',
    met: 'THE MET',
    rijks: 'RIJKSMUSEUM',
    cleveland: 'CLEVELAND MUSEUM OF ART',
    harvard: 'HARVARD ART MUSEUMS',
    lacma: 'LACMA',
    chicago: 'ART INSTITUTE OF CHICAGO',
};

/**
 * Open the lightbox.
 * @param {string}   src            - image URL
 * @param {string}   title
 * @param {string}   author
 * @param {string}   source         - museum key
 * @param {number[]|null} memberIds - FAISS ids of cluster members (null if no cluster)
 * @param {number|null}   clusterSize
 */
async function openLightbox(src, title, author, source, memberIds = null, clusterSize = null) {
    currentLbPath = src;
    lbImg.src = src;
    lbTitle.textContent = (title || 'Untitled').toUpperCase();
    lbAuthor.textContent = (author && author !== 'N/A') ? author.toUpperCase() : '';
    lbMuseum.textContent = MUSEUM_NAMES[source] || (source || '').toUpperCase();
    lbDownload.href = src;
    lbDownload.download = src.split('/').pop();
    lbPin.classList.toggle('pinned', savedItems.has(src));

    // Reset similar strip
    lbSimilaires.classList.remove('has-items');
    lbSimStrip.innerHTML = '';
    lbSimLabel.textContent = '';

    lightbox.classList.add('open');
    document.body.style.overflow = 'hidden';

    // Load similar images if cluster
    if (memberIds && memberIds.length > 1) {
        const simCount = clusterSize - 1;
        lbSimLabel.textContent = `${simCount} IMAGE${simCount > 1 ? 'S' : ''} SIMILAIRE${simCount > 1 ? 'S' : ''}`;
        lbSimilaires.classList.add('has-items');

        // Render placeholder thumbs while loading
        memberIds.forEach((_, i) => {
            const ph = document.createElement('div');
            ph.className = 'lb-sim-thumb';
            lbSimStrip.appendChild(ph);
        });

        try {
            const resp = await fetch(`${API_URL}/cluster-members`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ faiss_ids: memberIds }),
            });
            if (!resp.ok) throw new Error('Failed to load similaires');
            const data = await resp.json();

            lbSimStrip.innerHTML = '';
            data.results.forEach(member => {
                const memberPath = member.image_url || `/images/${member.filename}`;
                const isActive = memberPath === src;

                const thumb = document.createElement('div');
                thumb.className = 'lb-sim-thumb' + (isActive ? ' active' : '');
                thumb.title = member.Title || '';
                thumb.innerHTML = `<img src="${memberPath}" alt="${member.Title || ''}" loading="lazy">`;

                thumb.addEventListener('click', () => {
                    if (isActive) return;
                    // Swap main image smoothly
                    lbImg.style.opacity = '0';
                    lbImg.src = memberPath;
                    lbImg.onload = () => { lbImg.style.opacity = '1'; };
                    lbTitle.textContent = (member.Title || 'Untitled').toUpperCase();
                    lbAuthor.textContent = (member.Author && member.Author !== 'N/A')
                        ? member.Author.toUpperCase() : '';
                    lbMuseum.textContent = MUSEUM_NAMES[member.source] || (member.source || '').toUpperCase();
                    lbDownload.href = memberPath;
                    lbDownload.download = memberPath.split('/').pop();
                    // Sync pin state for the newly shown image
                    currentLbPath = memberPath;
                    lbPin.classList.toggle('pinned', savedItems.has(memberPath));
                    // Update active state
                    lbSimStrip.querySelectorAll('.lb-sim-thumb').forEach(t => t.classList.remove('active'));
                    thumb.classList.add('active');
                });

                lbSimStrip.appendChild(thumb);
            });
        } catch (err) {
            console.error('Similar strip error:', err);
        }
    }
}

function closeLightbox() {
    lightbox.classList.remove('open');
    document.body.style.overflow = '';
    lbImg.src = '';
    lbImg.style.opacity = '';
    lbSimilaires.classList.remove('has-items');
    lbSimStrip.innerHTML = '';
}

lbClose.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

// Lightbox PIN button
lbPin.addEventListener('click', () => {
    if (!currentLbPath) return;
    const d = savedItems.get(currentLbPath);
    toggleSaved(
        currentLbPath,
        d ? d.title : lbTitle.textContent,
        d ? d.author : lbAuthor.textContent,
        d ? d.source : null,
        d ? d.memberIds : null,
        d ? d.clusterSize : null
    );
});

// ── Saved images management ─────────────────────────────────────────────────────────
function toggleSaved(imgPath, title, author, source, memberIds, clusterSize) {
    const nowPinned = !savedItems.has(imgPath);
    if (nowPinned) {
        savedItems.set(imgPath, { title, author, source, memberIds, clusterSize });
        addSavedThumb(imgPath, title, author, source, memberIds, clusterSize);
    } else {
        savedItems.delete(imgPath);
        savedThumbs.querySelectorAll('.saved-thumb').forEach(t => {
            if (t.dataset.src === imgPath) t.remove();
        });
    }
    // Sync ALL gallery pin buttons for this image
    document.querySelectorAll('.card-action-save').forEach(b => {
        if (b.dataset.src === imgPath) {
            b.classList.toggle('saved', nowPinned);
            b.innerHTML = nowPinned ? ICON_PIN_FILLED : ICON_PIN_EMPTY;
        }
    });
    // Sync lightbox PIN button if same image is open
    if (currentLbPath === imgPath) {
        lbPin.classList.toggle('pinned', nowPinned);
    }
    document.body.classList.toggle('has-saved', savedItems.size > 0);
}

function addSavedThumb(imgPath, title, author, source, memberIds, clusterSize) {
    const thumb = document.createElement('div');
    thumb.className = 'saved-thumb';
    thumb.dataset.src = imgPath;
    thumb.title = title;

    const img = document.createElement('img');
    img.src = imgPath;
    img.alt = title;
    thumb.appendChild(img);

    // × remove button
    const rm = document.createElement('button');
    rm.className = 'saved-thumb-remove';
    rm.title = 'Retirer';
    rm.textContent = '×';
    rm.addEventListener('click', e => {
        e.stopPropagation();
        // toggleSaved will remove from savedItems, remove thumb, and sync all buttons
        toggleSaved(imgPath, title, author, source, memberIds, clusterSize);
    });
    thumb.appendChild(rm);

    // Click thumbnail → open lightbox
    thumb.addEventListener('click', () => {
        const d = savedItems.get(imgPath);
        if (d) openLightbox(imgPath, d.title, d.author, d.source, d.memberIds, d.clusterSize);
    });

    savedThumbs.appendChild(thumb);
}

// ── Cluster group lightbox (kept for backward compat, no longer primary entry) ─
const clusterLightbox = document.getElementById('cluster-lightbox');
const clusterGrid = document.getElementById('cluster-grid');
const clusterLabel = document.getElementById('cluster-label');
const clusterBack = document.getElementById('cluster-back');
const clusterClose = document.querySelector('.cluster-close');

function closeClusterLightbox() {
    clusterLightbox.classList.remove('open');
    document.body.style.overflow = '';
    clusterGrid.innerHTML = '';
}

clusterBack.addEventListener('click', closeClusterLightbox);
clusterClose.addEventListener('click', closeClusterLightbox);

// Escape: close panel → cluster lightbox → regular lightbox (priority order)
document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (filterPanel.classList.contains('open')) { closeFilterPanel(); return; }
    if (clusterLightbox.classList.contains('open')) { closeClusterLightbox(); return; }
    if (lightbox.classList.contains('open')) { closeLightbox(); }
});

// ── Event listeners ────────────────────────────────────────────────────────────
searchBtn.addEventListener('click', () => search(searchInput.value.trim()));
searchInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') search(searchInput.value.trim());
});
suggestionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        searchInput.value = btn.innerText;
        search(btn.innerText);
    });
});

// ── Diversity toggle ────────────────────────────────────────────────────────────
diversityToggle.addEventListener('click', () => {
    const isOn = diversityToggle.getAttribute('aria-pressed') === 'true';
    diversityToggle.setAttribute('aria-pressed', isOn ? 'false' : 'true');
    if (currentQuery) search(currentQuery);
});

// ── Combination mode radios ──────────────────────────────────────────────────────
document.querySelectorAll('input[name="combination-mode"]').forEach(radio => {
    radio.addEventListener('change', () => {
        if (currentQuery) search(currentQuery);
    });
});


// ── Filter panel (+) ──────────────────────────────────────────────────────────
function openFilterPanel() {
    filterPanel.classList.add('open');
    filterBtn.setAttribute('aria-expanded', 'true');
    filterPanel.removeAttribute('aria-hidden');
}

function closeFilterPanel() {
    filterPanel.classList.remove('open');
    filterBtn.setAttribute('aria-expanded', 'false');
    filterPanel.setAttribute('aria-hidden', 'true');
}

filterBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = filterPanel.classList.contains('open');
    isOpen ? closeFilterPanel() : openFilterPanel();
});

document.addEventListener('click', (e) => {
    if (filterPanel.classList.contains('open')
        && !filterPanel.contains(e.target)
        && e.target !== filterBtn) {
        closeFilterPanel();
    }
});

// ── Image chips: helpers ──────────────────────────────────────────────────────
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

/** Append a new chip. Does NOT trigger search. */
function addImageChip(src, base64) {
    const idx = referenceImages.length;
    referenceImages.push({ src, base64 });

    const chip = document.createElement('div');
    chip.className = 'image-chip';
    chip.dataset.idx = idx;

    const thumb = document.createElement('img');
    thumb.src = src;
    thumb.alt = 'Reference image';
    chip.appendChild(thumb);

    const btn = document.createElement('button');
    btn.className = 'image-chip-remove';
    btn.ariaLabel = 'Remove image';
    btn.textContent = '×';
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeImageChip(chip, Number(chip.dataset.idx));
    });
    chip.appendChild(btn);

    chipsContainer.appendChild(chip);
}

/** Remove a specific chip and re-index remaining ones. Re-runs search if query active. */
function removeImageChip(chipEl, idx) {
    chipEl.remove();
    referenceImages.splice(idx, 1);
    chipsContainer.querySelectorAll('.image-chip').forEach((c, i) => { c.dataset.idx = i; });
    if (currentQuery) search(currentQuery);
}

/** Remove all chips. */
function clearAllChips(rerun = false) {
    chipsContainer.innerHTML = '';
    referenceImages = [];
    if (rerun && currentQuery) search(currentQuery);
}

// ── Drag & drop onto search container ────────────────────────────────────────
// Each drop ADDS a chip. Search is NOT triggered — user presses Enter or button.

searchContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    searchContainer.classList.add('drag-over');
});

searchContainer.addEventListener('dragleave', (e) => {
    if (!searchContainer.contains(e.relatedTarget)) {
        searchContainer.classList.remove('drag-over');
    }
});

searchContainer.addEventListener('drop', async (e) => {
    e.preventDefault();
    searchContainer.classList.remove('drag-over');
    searchInput.focus();

    try {
        const uriList = e.dataTransfer.getData('text/uri-list')
            || e.dataTransfer.getData('text/plain');

        if (uriList && uriList.trim()) {
            const url = uriList.trim().split('\n')[0];
            const isLocal = url.startsWith('/') || url.startsWith(location.origin);
            if (isLocal) {
                const resp = await fetch(url);
                const blob = await resp.blob();
                const b64 = await blobToBase64(blob);
                addImageChip(url, b64);
                return;
            }
        }

        const files = e.dataTransfer.files;
        if (files && files.length) {
            for (const file of files) {
                if (!file.type.startsWith('image/')) continue;
                const b64 = await blobToBase64(file);
                const src = URL.createObjectURL(file);
                addImageChip(src, b64);
            }
        }

    } catch (err) {
        console.error('Drop error:', err);
    }
});

// ── Utility: fetch image URL → base64 ────────────────────────────────────────
async function imgUrlToBase64(url) {
    const resp = await fetch(url);
    const blob = await resp.blob();
    return blobToBase64(blob);
}

// ── Update has-content class for clear button visibility ─────────────────────
function updateHasContent() {
    const hasText = searchInput.value.trim().length > 0;
    const hasImages = referenceImages.length > 0 || referenceNegativeImages.length > 0;
    searchContainer.classList.toggle('has-content', hasText || hasImages);
}
searchInput.addEventListener('input', updateHasContent);

// ── Clear all ─────────────────────────────────────────────────────────────────
function clearAll() {
    clearSnapshot = {
        text: searchInput.value,
        images: [...referenceImages],
        negImages: [...referenceNegativeImages],
    };
    searchInput.value = '';
    clearAllChips(false);
    clearAllNegativeChips(false);
    updateHasContent();

    const undoBtn = document.getElementById('undo-clear-btn');
    undoBtn.style.display = 'flex';
    clearTimeout(undoClearTimer);
    undoClearTimer = setTimeout(() => {
        undoBtn.style.display = 'none';
        clearSnapshot = null;
    }, 5000);
}

function undoClear() {
    if (!clearSnapshot) return;
    searchInput.value = clearSnapshot.text;
    // Reset state
    referenceImages = [];
    referenceNegativeImages = [];
    chipsContainer.innerHTML = '';
    // Rebuild from snapshot
    clearSnapshot.images.forEach(r => addImageChip(r.src, r.base64));
    clearSnapshot.negImages.forEach(r => addNegativeChipEl(r.src, r.base64));
    document.getElementById('undo-clear-btn').style.display = 'none';
    clearSnapshot = null;
    updateHasContent();
    if (currentQuery) search(currentQuery);
}

document.getElementById('clear-all-btn').addEventListener('click', clearAll);
document.getElementById('undo-clear-btn').addEventListener('click', undoClear);

// ── Board system ──────────────────────────────────────────────────────────────
function _boardLabel() {
    const now = new Date();
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const isToday = now.getTime() >= today.getTime();
    if (isToday) {
        return now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    }
    return now.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: '2-digit' })
        + ' ' + now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
}

function newBoard() {
    if (savedItems.size === 0) return;
    const archived = {
        id: Date.now(),
        name: _boardLabel(),
        createdAt: new Date().toISOString(),
        images: Array.from(savedItems.entries()).map(([src, d]) => ({ src, ...d })),
    };
    archivedBoards.unshift(archived);
    localStorage.setItem('archivedBoards', JSON.stringify(archivedBoards));

    // Clear current board
    savedItems.clear();
    savedThumbs.innerHTML = '';
    document.body.classList.remove('has-saved');

    if (document.body.classList.contains('pinbar-expanded')) {
        renderExpandedBoards();
    }
}

// ── Pinbar: expand / collapse / use for search ────────────────────────────────
// SVG icons for the expand toggle button
const ICON_EXPAND_PINBAR  = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="7 9 12 4 17 9"/><polyline points="17 15 12 20 7 15"/></svg>`;
const ICON_COLLAPSE_PINBAR = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 9 12 14 7 9"/><polyline points="7 15 12 10 17 15"/></svg>`;

const _expandBtn = document.getElementById('pinbar-expand-btn');

function pinbarExpand() {
    document.body.classList.add('pinbar-expanded');
    _expandBtn.innerHTML = ICON_COLLAPSE_PINBAR;
    _expandBtn.title = 'Réduire';
    renderExpandedBoards();
}

function pinbarCollapse() {
    document.body.classList.remove('pinbar-expanded');
    _expandBtn.innerHTML = ICON_EXPAND_PINBAR;
    _expandBtn.title = 'Expand / Collapse boards';
}

function pinbarToggle() {
    document.body.classList.contains('pinbar-expanded') ? pinbarCollapse() : pinbarExpand();
}

function renderExpandedBoards() {
    const list = document.getElementById('pinbar-boards-list');
    list.innerHTML = '';

    // Current board row
    if (savedItems.size > 0) {
        const row = document.createElement('div');
        row.className = 'pinbar-board-row';
        const label = document.createElement('div');
        label.className = 'pinbar-board-label';
        label.textContent = 'BOARD ACTUEL';
        const thumbs = document.createElement('div');
        thumbs.className = 'pinbar-board-thumbs';
        savedItems.forEach(({ title }, src) => {
            const img = document.createElement('img');
            img.src = src; img.alt = title;
            img.className = 'pinbar-board-thumb';
            img.addEventListener('click', () => {
                const d = savedItems.get(src);
                if (d) openLightbox(src, d.title, d.author, d.source, d.memberIds, d.clusterSize);
            });
            thumbs.appendChild(img);
        });
        row.appendChild(label);
        row.appendChild(thumbs);
        list.appendChild(row);
    }

    // Archived boards
    archivedBoards.forEach(board => {
        const row = document.createElement('div');
        row.className = 'pinbar-board-row';
        const label = document.createElement('div');
        label.className = 'pinbar-board-label';
        label.textContent = board.name;
        const thumbs = document.createElement('div');
        thumbs.className = 'pinbar-board-thumbs';
        (board.images || []).forEach(img => {
            const el = document.createElement('img');
            el.src = img.src; el.alt = img.title || '';
            el.className = 'pinbar-board-thumb';
            el.addEventListener('click', () => {
                openLightbox(img.src, img.title, img.author, img.source, img.memberIds, img.clusterSize);
            });
            thumbs.appendChild(el);
        });
        row.appendChild(label);
        row.appendChild(thumbs);
        list.appendChild(row);
    });
}

async function pinbarUseForSearch() {
    // Convert all saved thumbnails to chips in the search bar
    for (const [src] of savedItems) {
        if (referenceImages.some(r => r.src === src)) continue;
        try {
            const b64 = await imgUrlToBase64(src);
            addImageChip(src, b64);
        } catch (e) { console.error('pinbarUseForSearch:', e); }
    }
    updateHasContent();
    if (searchInput.value.trim() || referenceImages.length) {
        search(searchInput.value.trim());
    }
}

// Pinbar button listeners
document.getElementById('pinbar-use-search-btn').addEventListener('click', pinbarUseForSearch);
document.getElementById('pinbar-new-btn').addEventListener('click', newBoard);
_expandBtn.addEventListener('click', pinbarToggle);

// ── "Ne plus voir" overlay ────────────────────────────────────────────────────
let _hideTimerHandle = null;

function showHideOverlay(card, imgWrap, imgPath, title, author, source, memberIds, clusterSize) {
    hiddenImgPaths.add(imgPath);

    const overlay = document.createElement('div');
    overlay.className = 'hide-overlay';
    overlay.innerHTML = `
        <div class="hide-overlay-top">
            <button class="hide-undo-btn" title="Annuler">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="9 14 4 9 9 4"/><path d="M20 20v-7a4 4 0 0 0-4-4H4"/>
                </svg>
            </button>
            <div class="hide-timer-area">
                <svg width="18" height="18" viewBox="0 0 18 18">
                    <circle cx="9" cy="9" r="7.5" fill="none" stroke="#000" stroke-width="1.5" opacity="0.12"/>
                    <circle cx="9" cy="9" r="7.5" fill="none" stroke="#000" stroke-width="1.5"
                            class="timer-circle"/>
                </svg>
                <button class="hide-skip-btn">SKIP</button>
            </div>
        </div>
        <div class="hide-options">
            <button class="hide-option" data-action="negative">
                <span class="ho-title">Use as negative</span>
                <span class="ho-desc">Exclude its specific traits from search</span>
            </button>
            <button class="hide-option" data-action="taste">
                <span class="ho-title">Not my taste</span>
                <span class="ho-desc">Hide from my results forever</span>
            </button>
            <button class="hide-option" data-action="flag">
                <span class="ho-title">WTF Flag</span>
                <span class="ho-desc">For us – doesn't affect your search</span>
            </button>
        </div>
    `;

    imgWrap.appendChild(overlay);

    // Auto-dismiss after 10s
    _hideTimerHandle = setTimeout(() => dismissHideOverlay(card, overlay, imgPath), 10000);

    // Undo: remove overlay, restore
    overlay.querySelector('.hide-undo-btn').addEventListener('click', () => {
        clearTimeout(_hideTimerHandle);
        hiddenImgPaths.delete(imgPath);
        overlay.remove();
    });

    // SKIP: dismiss without action
    overlay.querySelector('.hide-skip-btn').addEventListener('click', () => {
        clearTimeout(_hideTimerHandle);
        dismissHideOverlay(card, overlay, imgPath);
    });

    // Options
    overlay.querySelectorAll('.hide-option').forEach(btn => {
        btn.addEventListener('click', async () => {
            clearTimeout(_hideTimerHandle);
            const action = btn.dataset.action;
            if (action === 'negative') {
                try {
                    const b64 = await imgUrlToBase64(imgPath);
                    addNegativeChipEl(imgPath, b64);
                    updateHasContent();
                    if (currentQuery) search(currentQuery);
                } catch (e) { console.error('negative:', e); }
            } else if (action === 'taste') {
                // Persist in localStorage
                const denylist = JSON.parse(localStorage.getItem('denylist') || '[]');
                if (!denylist.includes(imgPath)) {
                    denylist.push(imgPath);
                    localStorage.setItem('denylist', JSON.stringify(denylist));
                }
            } else if (action === 'flag') {
                // Send flag to backend (fire & forget)
                fetch('/flag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ img_path: imgPath }),
                }).catch(() => { });
            }
            dismissHideOverlay(card, overlay, imgPath);
        });
    });
}

function dismissHideOverlay(card, overlay, imgPath) {
    overlay.remove();
    // Fade card out and load replacement
    card.style.transition = 'opacity 0.3s';
    card.style.opacity = '0';
    setTimeout(() => {
        card.remove();
        if (hasMore && !isLoading) loadNextPage();
    }, 300);
}

// ── Negative chips ────────────────────────────────────────────────────────────
function _ensureNegSeparator() {
    if (chipsContainer.querySelector('.neg-separator')) return;
    const sep = document.createElement('span');
    sep.className = 'neg-separator';
    sep.textContent = '−';
    chipsContainer.appendChild(sep);
}

function _removeNegSeparatorIfEmpty() {
    if (referenceNegativeImages.length === 0) {
        const sep = chipsContainer.querySelector('.neg-separator');
        if (sep) sep.remove();
        // Hide negative mode section
        document.getElementById('negative-mode-section').hidden = true;
    }
}

function addNegativeChipEl(src, base64, skipPush = false) {
    if (!skipPush) {
        referenceNegativeImages.push({ src, base64 });
    }
    _ensureNegSeparator();
    // Show negative mode section in filter panel
    document.getElementById('negative-mode-section').hidden = false;

    const idx = referenceNegativeImages.length - 1;
    const chip = document.createElement('div');
    chip.className = 'image-chip negative';
    chip.dataset.idx = idx;
    chip.dataset.neg = '1';

    const thumb = document.createElement('img');
    thumb.src = src; thumb.alt = 'Negative reference';
    chip.appendChild(thumb);

    const btn = document.createElement('button');
    btn.className = 'image-chip-remove';
    btn.ariaLabel = 'Remove negative image';
    btn.textContent = '×';
    btn.addEventListener('click', e => {
        e.stopPropagation();
        removeNegativeChip(chip, Number(chip.dataset.idx));
    });
    chip.appendChild(btn);
    chipsContainer.appendChild(chip);
}

function removeNegativeChip(chipEl, idx) {
    chipEl.remove();
    referenceNegativeImages.splice(idx, 1);
    chipsContainer.querySelectorAll('.image-chip.negative').forEach((c, i) => { c.dataset.idx = i; });
    _removeNegSeparatorIfEmpty();
    updateHasContent();
    if (currentQuery) search(currentQuery);
}

function clearAllNegativeChips(rerun = false) {
    chipsContainer.querySelectorAll('.image-chip.negative').forEach(c => c.remove());
    referenceNegativeImages = [];
    _removeNegSeparatorIfEmpty();
    if (rerun && currentQuery) search(currentQuery);
}

// Negative mode change → retrigger search
document.querySelectorAll('input[name="negative-mode"]').forEach(radio => {
    radio.addEventListener('change', () => {
        if (currentQuery && referenceNegativeImages.length) search(currentQuery);
    });
});

// ── Magic link auth (pinbar expanded view) ────────────────────────────────────
const pinbarMagicBtn = document.getElementById('pinbar-magic-btn');
const pinbarEmailInput = document.getElementById('pinbar-email-input');
const pinbarAuthStatus = document.getElementById('pinbar-auth-status');

pinbarMagicBtn.addEventListener('click', async () => {
    const email = pinbarEmailInput.value.trim();
    if (!email) return;
    pinbarMagicBtn.disabled = true;
    pinbarMagicBtn.textContent = '...';
    try {
        const resp = await fetch('/auth/request', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email }),
        });
        const data = await resp.json();
        pinbarAuthStatus.hidden = false;
        if (data.status === 'sent') {
            pinbarAuthStatus.textContent = `✓ Link sent to ${email} – check your inbox (or the server console in dev mode)`;
        } else {
            pinbarAuthStatus.textContent = 'Error – please try again';
        }
    } catch (e) {
        pinbarAuthStatus.hidden = false;
        pinbarAuthStatus.textContent = 'Network error';
    } finally {
        pinbarMagicBtn.disabled = false;
        pinbarMagicBtn.textContent = 'SEND LINK';
    }
});

// ── Update task.md: mark done ─────────────────────────────────────────────────
// (no-op in production — this comment is just a marker for the dev)
