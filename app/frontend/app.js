const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const gallery = document.getElementById('gallery');
const suggestionBtns = document.querySelectorAll('.suggestion-btn');

// API URL (same host since we serve static files from FastAPI)
const API_URL = "";

async function search(query) {
    if (!query) return;

    // Trigger hero → compact header transition
    document.body.classList.add('has-results');

    gallery.innerHTML = '<div class="placeholder-text">Searching...</div>';

    try {
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query, limit: 30 })
        });

        if (!response.ok) {
            throw new Error('Search failed');
        }

        const data = await response.json();
        renderResults(data.results);

    } catch (error) {
        gallery.innerHTML = `<div class="placeholder-text">Error: ${error.message}</div>`;
        console.error(error);
    }
}

function renderResults(results) {
    gallery.innerHTML = '';

    if (results.length === 0) {
        gallery.innerHTML = '<div class="placeholder-text">No results found.</div>';
        return;
    }

    results.forEach(item => {
        const card = document.createElement('div');
        card.className = 'image-card';

        const imgPath = item.image_url || `/images/${item.filename}`;
        const title = item.Title || 'Untitled';

        card.innerHTML = `
            <img src="${imgPath}" alt="${title}" loading="lazy">
            <div class="image-info">
                <div class="title">${title}</div>
                <div class="author">${item.Author || 'Unknown'}</div>
            </div>
        `;

        card.addEventListener('click', () => openLightbox(imgPath, title, item.Author, item.source));
        gallery.appendChild(card);
    });
}

// ── Lightbox ──────────────────────────────────────────────────────────────────
const lightbox = document.getElementById('lightbox');
const lbImg = document.getElementById('lightbox-img');
const lbTitle = document.getElementById('lightbox-title');
const lbAuthor = document.getElementById('lightbox-author');
const lbMuseum = document.getElementById('lightbox-museum');
const lbDownload = document.getElementById('lightbox-download');
const lbClose = document.querySelector('.lightbox-close');

const MUSEUM_NAMES = {
    artic: 'ART INSTITUTE OF CHICAGO',
    met: 'THE MET',
    rijks: 'RIJKSMUSEUM',
    cleveland: 'CLEVELAND MUSEUM OF ART',
    harvard: 'HARVARD ART MUSEUMS',
    lacma: 'LACMA',
    chicago: 'ART INSTITUTE OF CHICAGO',
};

function openLightbox(src, title, author, source) {
    lbImg.src = src;
    lbTitle.textContent = (title || 'Untitled').toUpperCase();
    lbAuthor.textContent = (author && author !== 'N/A') ? author.toUpperCase() : '';
    lbMuseum.textContent = MUSEUM_NAMES[source] || (source || '').toUpperCase();
    lbDownload.href = src;
    lbDownload.download = src.split('/').pop();
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
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// Event Listeners
searchBtn.addEventListener('click', () => {
    search(searchInput.value);
});

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        search(searchInput.value);
    }
});

suggestionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        searchInput.value = btn.innerText;
        search(btn.innerText);
    });
});
