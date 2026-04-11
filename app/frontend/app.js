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

// ── State ──────────────────────────────────────────────────────────────────────
let currentQuery  = "";
let currentOffset = 0;
let isLoading     = false;
let hasMore       = true;
// Multimodal: base64-encoded reference image for text+image blending
let referenceImage = null;

// ── Image chip DOM refs ─────────────────────────────────────────────────────────
const searchContainer  = document.getElementById('search-container');
const imageChip        = document.getElementById('image-chip');
const imageChipThumb   = document.getElementById('image-chip-thumb');
const imageChipRemove  = document.getElementById('image-chip-remove');

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
                query:           currentQuery,
                page_size:       PAGE_SIZE,
                offset:          currentOffset,
                diversity:       getDiversity(),
                ...(referenceImage ? { reference_image: referenceImage, image_weight: 0.5 } : {}),
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
        const simCount = hasGroup ? item.cluster_size - 1 : 0; // exclude representative itself

        // Title: shows similar count inline on hover only (no permanent badge)
        const titleHTML = hasGroup
            ? `<div class="title title-with-similar">
                 <span>${title}</span>
                 <span class="similar-count">&nbsp;·&nbsp;${simCount}&nbsp;SIMILAIRE${simCount > 1 ? 'S' : ''}</span>
               </div>`
            : `<div class="title">${title}</div>`;

        card.innerHTML = `
            <img src="${imgPath}" alt="${title}" loading="lazy">
            <div class="image-info">
                ${titleHTML}
                <div class="author">${item.Author || 'Unknown'}</div>
            </div>
        `;

        // Make gallery image draggable → can be dropped into search bar
        const img = card.querySelector('img');
        img.draggable = true;
        img.addEventListener('dragstart', e => {
            e.dataTransfer.effectAllowed = 'copy';
            e.dataTransfer.setData('text/uri-list', imgPath);
            e.dataTransfer.setData('text/plain',    imgPath);
        });

        // All cards open the regular lightbox; cluster cards also load the strip
        card.addEventListener('click', () => {
            openLightbox(
                imgPath, title, item.Author, item.source,
                hasGroup ? item.cluster_member_ids : null,
                hasGroup ? item.cluster_size       : null
            );
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

// ── Filter panel (+) ────────────────────────────────────────────────────────────
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

// Close panel when clicking outside
document.addEventListener('click', (e) => {
    if (filterPanel.classList.contains('open')
        && !filterPanel.contains(e.target)
        && e.target !== filterBtn) {
        closeFilterPanel();
    }
});


// ── Image chip: helpers ────────────────────────────────────────────────────────
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload  = () => resolve(reader.result.split(',')[1]); // strip data:...;base64, prefix
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

/** Show the chip with a given src URL and base64 string. Does NOT trigger search. */
function setImageChip(src, base64) {
    referenceImage       = base64;
    imageChipThumb.src   = src;
    imageChip.classList.add('visible');
    imageChip.removeAttribute('aria-hidden');
}

/** Remove the chip. If a query is active, re-runs it without the image. */
function clearImageChip(rerun = true) {
    referenceImage = null;
    imageChipThumb.src = '';
    imageChip.classList.remove('visible');
    imageChip.setAttribute('aria-hidden', 'true');
    if (rerun && currentQuery) search(currentQuery);
}

imageChipRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImageChip(true);   // clear + re-run search without image
});

// ── Drag & drop onto search container ─────────────────────────────────────────
// Drop just sets the reference image chip.
// Search is NOT triggered automatically — the user presses Enter or the button.

searchContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    searchContainer.classList.add('drag-over');
});

searchContainer.addEventListener('dragleave', (e) => {
    // Only remove class if leaving the container entirely (not crossing a child)
    if (!searchContainer.contains(e.relatedTarget)) {
        searchContainer.classList.remove('drag-over');
    }
});

searchContainer.addEventListener('drop', async (e) => {
    e.preventDefault();
    searchContainer.classList.remove('drag-over');

    // Focus the text input so the user can type (or just press Enter)
    searchInput.focus();

    try {
        // Priority 1: image URL (from gallery drag or browser image drag)
        const uriList = e.dataTransfer.getData('text/uri-list')
                     || e.dataTransfer.getData('text/plain');

        if (uriList && uriList.trim()) {
            const url = uriList.trim().split('\n')[0]; // take first URL if multiple
            // Only accept same-origin URLs or relative paths to avoid CORS issues
            const isLocal = url.startsWith('/') || url.startsWith(location.origin);
            if (isLocal) {
                const resp  = await fetch(url);
                const blob  = await resp.blob();
                const b64   = await blobToBase64(blob);
                setImageChip(url, b64);
                return;
            }
        }

        // Priority 2: file dropped from OS
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            const b64 = await blobToBase64(file);
            const src = URL.createObjectURL(file);
            setImageChip(src, b64);
            return;
        }

    } catch (err) {
        console.error('Drop error:', err);
    }
});

