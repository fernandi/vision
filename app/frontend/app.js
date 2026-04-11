const searchInput    = document.getElementById('search-input');
const searchBtn      = document.getElementById('search-btn');
const gallery        = document.getElementById('gallery');
const diversityToggle = document.getElementById('diversity-toggle');
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
                query:     currentQuery,
                page_size: PAGE_SIZE,
                offset:    currentOffset,
                diversity: getDiversity(),
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

        // Cluster badge
        const clusterBadge = (item.cluster_size != null)
            ? `<span class="cluster-badge" title="${item.cluster_size} images similaires">×${item.cluster_size}</span>`
            : '';

        card.innerHTML = `
            <img src="${imgPath}" alt="${title}" loading="lazy">
            <div class="image-info">
                <div class="title">${title}</div>
                <div class="author">${item.Author || 'Unknown'}</div>
            </div>
            ${clusterBadge}
        `;

        card.addEventListener('click', () => {
            if (hasGroup) {
                openClusterView(item.cluster_member_ids, title, item.cluster_size);
            } else {
                openLightbox(imgPath, title, item.Author, item.source);
            }
        });

        gallery.appendChild(card);
    });
}

// ── Lightbox (single image) ──────────────────────────────────────────────────────
const lightbox   = document.getElementById('lightbox');
const lbImg      = document.getElementById('lightbox-img');
const lbTitle    = document.getElementById('lightbox-title');
const lbAuthor   = document.getElementById('lightbox-author');
const lbMuseum   = document.getElementById('lightbox-museum');
const lbDownload = document.getElementById('lightbox-download');
const lbClose    = document.querySelector('#lightbox .lightbox-close');

const MUSEUM_NAMES = {
    artic:     'ART INSTITUTE OF CHICAGO',
    met:       'THE MET',
    rijks:     'RIJKSMUSEUM',
    cleveland: 'CLEVELAND MUSEUM OF ART',
    harvard:   'HARVARD ART MUSEUMS',
    lacma:     'LACMA',
    chicago:   'ART INSTITUTE OF CHICAGO',
};

function openLightbox(src, title, author, source) {
    lbImg.src = src;
    lbTitle.textContent   = (title || 'Untitled').toUpperCase();
    lbAuthor.textContent  = (author && author !== 'N/A') ? author.toUpperCase() : '';
    lbMuseum.textContent  = MUSEUM_NAMES[source] || (source || '').toUpperCase();
    lbDownload.href       = src;
    lbDownload.download   = src.split('/').pop();
    lightbox.classList.add('open');
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    lightbox.classList.remove('open');
    document.body.style.overflow = '';
    lbImg.src = '';
}

lbClose.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

// ── Cluster group lightbox ──────────────────────────────────────────────────────
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

async function openClusterView(memberIds, representativeTitle, clusterSize) {
    // Open overlay immediately with a loading spinner
    clusterLabel.textContent = `${clusterSize} image${clusterSize > 1 ? 's' : ''} similaires`;
    clusterGrid.innerHTML = '<div class="cluster-loading"><span></span><span></span><span></span></div>';
    clusterLightbox.classList.add('open');
    document.body.style.overflow = 'hidden';

    try {
        const resp = await fetch(`${API_URL}/cluster-members`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ faiss_ids: memberIds }),
        });
        if (!resp.ok) throw new Error('Failed to load cluster');
        const data = await resp.json();
        renderClusterGrid(data.results);
    } catch (err) {
        clusterGrid.innerHTML = `<p style="color:#666;padding:2rem">Erreur: ${err.message}</p>`;
    }
}

function renderClusterGrid(items) {
    clusterGrid.innerHTML = '';
    items.forEach(item => {
        const imgPath = item.image_url || `/images/${item.filename}`;
        const title   = item.Title || 'Untitled';

        const thumb = document.createElement('div');
        thumb.className = 'cluster-thumb';
        thumb.innerHTML = `
            <img src="${imgPath}" alt="${title}" loading="lazy">
            <div class="cluster-thumb-info">${title}</div>
        `;
        thumb.addEventListener('click', () => {
            // Close cluster view, open regular lightbox
            closeClusterLightbox();
            openLightbox(imgPath, title, item.Author, item.source);
        });
        clusterGrid.appendChild(thumb);
    });
}

clusterBack.addEventListener('click',  closeClusterLightbox);
clusterClose.addEventListener('click', closeClusterLightbox);

// Escape: close cluster first, then regular lightbox if open
document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (clusterLightbox.classList.contains('open')) { closeClusterLightbox(); return; }
    if (lightbox.classList.contains('open'))        { closeLightbox(); }
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
    // Re-run the current query with the new diversity value
    if (currentQuery) search(currentQuery);
});
