const searchInput    = document.getElementById('search-input');
const searchBtn      = document.getElementById('search-btn');
const gallery        = document.getElementById('gallery');
const diversityToggle = document.getElementById('diversity-toggle');
const filterBtn      = document.getElementById('filter-btn');
const filterPanel    = document.getElementById('filter-panel');
const suggestionBtns = document.querySelectorAll('.suggestion-btn');

const API_URL   = "";
const PAGE_SIZE = 20;

// Current diversity value: read from toggle state
function getDiversity() {
    return diversityToggle.getAttribute('aria-pressed') === 'true' ? 0.5 : 0.0;
}

// Current combination mode: read from radio group
function getCombinationMode() {
    const checked = document.querySelector('input[name="combination-mode"]:checked');
    return checked ? checked.value : 'centroid';
}

// ── State ────────────────────────────────────────────────────────────────────
let currentQuery  = "";
let currentOffset = 0;
let isLoading     = false;
let hasMore       = true;
// Multimodal: array of { src, base64 } objects
let referenceImages = [];

// ── Saved images ────────────────────────────────────────────────────────────
// Map: imgPath → { title, author, source, memberIds, clusterSize }
const savedItems  = new Map();
const savedPanel  = document.getElementById('saved-panel');
const savedThumbs = document.getElementById('saved-thumbs');

const ICON_EXPAND = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>`;
const ICON_BOOKMARK_EMPTY  = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>`;
const ICON_BOOKMARK_FILLED = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>`;

// ── Image chips DOM ref ─────────────────────────────────────────────────────────
const searchContainer   = document.getElementById('search-container');
const chipsContainer    = document.getElementById('image-chips-container');

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
    currentQuery  = query;
    currentOffset = 0;
    hasMore       = true;

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
                query:            currentQuery,
                page_size:        PAGE_SIZE,
                offset:           currentOffset,
                diversity:        getDiversity(),
                combination_mode: getCombinationMode(),
                ...(referenceImages.length
                    ? { reference_images: referenceImages.map(r => r.base64), image_weight: 0.5 }
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
        const card     = document.createElement('div');
        card.className = 'image-card';
        const imgPath  = item.image_url || `/images/${item.filename}`;
        const title    = item.Title || 'Untitled';
        const hasGroup = item.cluster_size != null && item.cluster_size > 1
                         && Array.isArray(item.cluster_member_ids);
        const simCount = hasGroup ? item.cluster_size - 1 : 0;

        const titleHTML = hasGroup
            ? `<div class="title title-with-similar">
                 <span>${title}</span>
                 <span class="similar-count">&nbsp;·&nbsp;${simCount}&nbsp;SIMILAIRE${simCount > 1 ? 'S' : ''}</span>
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
        expandBtn.title     = 'Agrandir';
        expandBtn.innerHTML = ICON_EXPAND;
        expandBtn.addEventListener('click', e => {
            e.stopPropagation();
            openLightbox(imgPath, title, item.Author, item.source,
                hasGroup ? item.cluster_member_ids : null,
                hasGroup ? item.cluster_size       : null);
        });

        // Save → saved panel
        const saveBtn = document.createElement('button');
        saveBtn.className   = 'card-action-btn card-action-save';
        saveBtn.dataset.src = imgPath;
        saveBtn.title       = 'Enregistrer';
        const isSaved = savedItems.has(imgPath);
        saveBtn.innerHTML   = isSaved ? ICON_BOOKMARK_FILLED : ICON_BOOKMARK_EMPTY;
        if (isSaved) saveBtn.classList.add('saved');
        saveBtn.addEventListener('click', e => {
            e.stopPropagation();
            toggleSaved(imgPath, title, item.Author, item.source,
                hasGroup ? item.cluster_member_ids : null,
                hasGroup ? item.cluster_size       : null,
                saveBtn);
        });

        actions.appendChild(expandBtn);
        actions.appendChild(saveBtn);
        imgWrap.appendChild(actions);

        // ── Draggable ────────────────────────────────────────────────────────────
        const img = card.querySelector('img');
        img.draggable = true;
        img.addEventListener('dragstart', e => {
            e.dataTransfer.effectAllowed = 'copy';
            e.dataTransfer.setData('text/uri-list', imgPath);
            e.dataTransfer.setData('text/plain',    imgPath);
        });

        // ── Card click → lightbox ─────────────────────────────────────────────────
        card.addEventListener('click', () => {
            openLightbox(imgPath, title, item.Author, item.source,
                hasGroup ? item.cluster_member_ids : null,
                hasGroup ? item.cluster_size       : null);
        });

        gallery.appendChild(card);
    });
}

// ── Lightbox (single image + optional similar strip) ──────────────────────────
const lightbox    = document.getElementById('lightbox');
const lbImg       = document.getElementById('lightbox-img');
const lbTitle     = document.getElementById('lightbox-title');
const lbAuthor    = document.getElementById('lightbox-author');
const lbMuseum    = document.getElementById('lightbox-museum');
const lbDownload  = document.getElementById('lightbox-download');
const lbClose     = document.querySelector('#lightbox .lightbox-close');
const lbSimilaires = document.getElementById('lb-similaires');
const lbSimLabel  = document.getElementById('lb-sim-label');
const lbSimStrip  = document.getElementById('lb-sim-strip');

const MUSEUM_NAMES = {
    artic:     'ART INSTITUTE OF CHICAGO',
    met:       'THE MET',
    rijks:     'RIJKSMUSEUM',
    cleveland: 'CLEVELAND MUSEUM OF ART',
    harvard:   'HARVARD ART MUSEUMS',
    lacma:     'LACMA',
    chicago:   'ART INSTITUTE OF CHICAGO',
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
    lbImg.src = src;
    lbTitle.textContent   = (title  || 'Untitled').toUpperCase();
    lbAuthor.textContent  = (author && author !== 'N/A') ? author.toUpperCase() : '';
    lbMuseum.textContent  = MUSEUM_NAMES[source] || (source || '').toUpperCase();
    lbDownload.href       = src;
    lbDownload.download   = src.split('/').pop();

    // Reset similar strip
    lbSimilaires.classList.remove('has-items');
    lbSimStrip.innerHTML  = '';
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
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ faiss_ids: memberIds }),
            });
            if (!resp.ok) throw new Error('Failed to load similaires');
            const data = await resp.json();

            lbSimStrip.innerHTML = '';
            data.results.forEach(member => {
                const memberPath = member.image_url || `/images/${member.filename}`;
                const isActive   = memberPath === src;

                const thumb = document.createElement('div');
                thumb.className = 'lb-sim-thumb' + (isActive ? ' active' : '');
                thumb.title     = member.Title || '';
                thumb.innerHTML = `<img src="${memberPath}" alt="${member.Title || ''}" loading="lazy">`;

                thumb.addEventListener('click', () => {
                    if (isActive) return;
                    // Swap main image smoothly
                    lbImg.style.opacity = '0';
                    lbImg.src = memberPath;
                    lbImg.onload = () => { lbImg.style.opacity = '1'; };
                    lbTitle.textContent   = (member.Title  || 'Untitled').toUpperCase();
                    lbAuthor.textContent  = (member.Author && member.Author !== 'N/A')
                                            ? member.Author.toUpperCase() : '';
                    lbMuseum.textContent  = MUSEUM_NAMES[member.source] || (member.source || '').toUpperCase();
                    lbDownload.href       = memberPath;
                    lbDownload.download   = memberPath.split('/').pop();
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

// ── Saved images management ─────────────────────────────────────────────────────────
function toggleSaved(imgPath, title, author, source, memberIds, clusterSize, btn) {
    if (savedItems.has(imgPath)) {
        savedItems.delete(imgPath);
        savedThumbs.querySelectorAll('.saved-thumb').forEach(t => {
            if (t.dataset.src === imgPath) t.remove();
        });
        btn.classList.remove('saved');
        btn.innerHTML = ICON_BOOKMARK_EMPTY;
    } else {
        savedItems.set(imgPath, { title, author, source, memberIds, clusterSize });
        addSavedThumb(imgPath, title, author, source, memberIds, clusterSize);
        btn.classList.add('saved');
        btn.innerHTML = ICON_BOOKMARK_FILLED;
    }
    document.body.classList.toggle('has-saved', savedItems.size > 0);
}

function addSavedThumb(imgPath, title, author, source, memberIds, clusterSize) {
    const thumb = document.createElement('div');
    thumb.className  = 'saved-thumb';
    thumb.dataset.src = imgPath;
    thumb.title      = title;

    const img = document.createElement('img');
    img.src  = imgPath;
    img.alt  = title;
    thumb.appendChild(img);

    // × remove button
    const rm = document.createElement('button');
    rm.className   = 'saved-thumb-remove';
    rm.title       = 'Retirer';
    rm.textContent = '×';
    rm.addEventListener('click', e => {
        e.stopPropagation();
        savedItems.delete(imgPath);
        thumb.remove();
        document.body.classList.toggle('has-saved', savedItems.size > 0);
        // Un-highlight any visible save buttons for this image
        document.querySelectorAll('.card-action-save').forEach(b => {
            if (b.dataset.src === imgPath) {
                b.classList.remove('saved');
                b.innerHTML = ICON_BOOKMARK_EMPTY;
            }
        });
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
const clusterGrid     = document.getElementById('cluster-grid');
const clusterLabel    = document.getElementById('cluster-label');
const clusterBack     = document.getElementById('cluster-back');
const clusterClose    = document.querySelector('.cluster-close');

function closeClusterLightbox() {
    clusterLightbox.classList.remove('open');
    document.body.style.overflow = '';
    clusterGrid.innerHTML = '';
}

clusterBack.addEventListener('click',  closeClusterLightbox);
clusterClose.addEventListener('click', closeClusterLightbox);

// Escape: close panel → cluster lightbox → regular lightbox (priority order)
document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (filterPanel.classList.contains('open'))        { closeFilterPanel();        return; }
    if (clusterLightbox.classList.contains('open'))    { closeClusterLightbox();    return; }
    if (lightbox.classList.contains('open'))           { closeLightbox(); }
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
        reader.onload  = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

/** Append a new chip. Does NOT trigger search. */
function addImageChip(src, base64) {
    const idx = referenceImages.length;
    referenceImages.push({ src, base64 });

    const chip = document.createElement('div');
    chip.className   = 'image-chip';
    chip.dataset.idx = idx;

    const thumb = document.createElement('img');
    thumb.src = src;
    thumb.alt = 'Reference image';
    chip.appendChild(thumb);

    const btn = document.createElement('button');
    btn.className   = 'image-chip-remove';
    btn.ariaLabel   = 'Remove image';
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
                const b64  = await blobToBase64(blob);
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
