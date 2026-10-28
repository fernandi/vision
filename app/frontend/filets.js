// ── Filets: bands of stretched pixels, the visual identity of the interface ──
// A band shows ONE line of pixels of an artwork (a row, or a column) stretched
// across its thickness. Images themselves are never distorted. Nothing is read
// back from the image: the browser stretches the thumbnail already on screen
// (a background a thousand times taller or wider than the band), so it works
// with any image host and costs nothing but a background on a small element.
//
// - Ambient bands ([data-ambient]) take the colours of what is being looked
//   at: the search, the open image, the collection. Before that, generic lines
//   from filets.png (outils/filets_generiques.py).
// - Cards: hovering an image draws its line under it.
// - Gesture: collecting lifts one line out of the image and slides it into the
//   collection's band.

const Filets = (() => {
    const GENERIC = { url: 'filets.png', rows: 16 };
    const STRETCH = 1000;   // the line is magnified 1000×: one source line fills the band
    const cssUrl = u => `url("${encodeURI(u)}")`;

    // Animations are on for everyone; Settings › Animations › Reduced turns them
    // off for the whole site (remembered by the browser). Off: CSS animations and
    // transitions are cut (html.motion-off) and scripted ones finish at once.
    let motionOff = false;
    try { motionOff = localStorage.getItem('glane.animations') === 'off'; } catch { /* private mode */ }
    const nativeAnimate = Element.prototype.animate;
    Element.prototype.animate = function (keyframes, options) {
        if (motionOff) options = typeof options === 'number' ? 0 : { ...options, duration: 0, delay: 0, iterations: 1 };
        return nativeAnimate.call(this, keyframes, options);
    };
    function setMotion(on) {
        motionOff = !on;
        document.documentElement.classList.toggle('motion-off', motionOff);
        try { if (on) localStorage.removeItem('glane.animations'); else localStorage.setItem('glane.animations', 'off'); } catch { /* private mode */ }
    }
    document.documentElement.classList.toggle('motion-off', motionOff);
    const motion = () => !motionOff;

    // Which line of an image: always the centre one. It crosses the subject,
    // so it is far more varied than an edge (often a frame or a backdrop).
    function fromItem(item) {
        if (!item?.src) return null;
        return { url: item.src, at: 0.5 };
    }
    function generic(i = Math.floor(Math.random() * GENERIC.rows)) {
        return { url: GENERIC.url, row: ((i % GENERIC.rows) + GENERIC.rows) % GENERIC.rows };
    }

    // axis 'row': one row of pixels, colours running along the width;
    // axis 'column': one column, colours running down the height.
    function paint(el, src, axis = 'row') {
        if (!src) { el.style.backgroundImage = ''; return; }
        el.style.backgroundImage = cssUrl(src.url);
        el.style.backgroundRepeat = 'no-repeat';
        if (src.row != null) {
            // Generic sheet: 16 lines from 16 artworks, each repeated 3 times.
            // Row: one block fills the band. Column: one column of the sheet
            // (a pixel from each artwork, top to bottom) stretched sideways.
            const along = `${(src.row / (GENERIC.rows - 1)) * 100}%`;
            el.style.backgroundSize = axis === 'row' ? `100% ${GENERIC.rows * 100}%` : `${STRETCH * 100}% 100%`;
            el.style.backgroundPosition = axis === 'row' ? `0 ${along}` : `${along} 0`;
            return;
        }
        // Line `at` of the magnified image in the box: p = (a·s − ½) / (s − 1).
        const p = ((src.at * STRETCH - 0.5) / (STRETCH - 1)) * 100;
        el.style.backgroundSize = axis === 'row' ? `100% ${STRETCH * 100}%` : `${STRETCH * 100}% 100%`;
        el.style.backgroundPosition = axis === 'row' ? `0 ${p}%` : `${p}% 0`;
    }

    // ── Ambient ─────────────────────────────────────────────────────────────
    // A new source is written over the old one, along the band.
    let ambient = generic();
    let ambientKey = 'generic';

    // data-line shifts which line of the image a band reads, so bands painted
    // from the same image (the menu's section edges) differ yet share a palette.
    function lineOf(src, shift) {
        if (!shift) return src;
        if (src.row != null) return { ...src, row: (src.row + Math.round(shift * 10) + GENERIC.rows) % GENERIC.rows };
        return { ...src, at: Math.min(0.96, Math.max(0.04, src.at + shift)) };
    }

    function paintAmbient(el, animate) {
        const axis = el.dataset.axis || 'row';
        const layer = document.createElement('i');
        paint(layer, lineOf(ambient, Number(el.dataset.line) || 0), axis);
        const old = [...el.children];
        el.appendChild(layer);
        if (!animate || !motion() || !el.getClientRects().length) { old.forEach(o => o.remove()); return; }
        const hidden = axis === 'row' ? 'inset(0 100% 0 0)' : 'inset(0 0 100% 0)';
        layer.animate([{ clipPath: hidden }, { clipPath: 'inset(0 0 0 0)' }],
            { duration: 900, easing: 'cubic-bezier(0.65, 0, 0.35, 1)' })
            .onfinish = () => old.forEach(o => o.remove());
    }

    function setAmbient(item) {
        const key = item ? (item.id ?? item.src) : 'generic';
        if (key === ambientKey) return;
        ambientKey = key;
        ambient = fromItem(item) || generic();
        document.querySelectorAll('[data-ambient]').forEach(el => paintAmbient(el, true));
    }

    // ── Cards ───────────────────────────────────────────────────────────────
    // How a card shows its image (object-fit: cover): scale, offsets, and the
    // source row of its line `at`.
    function coverFit(img, w, h, at) {
        const nw = img.naturalWidth, nh = img.naturalHeight;
        const s = Math.max(w / nw, h / nh);
        const row = Math.min(nh - 1, Math.floor(at * nh));
        return { nw, nh, s, row, dw: nw * s, dh: nh * s, ox: (w - nw * s) / 2, oy: (h - nh * s) / 2 };
    }

    // The card's dark hover overlay ends at rgba(0,0,0,.66) on the image's lower
    // edge; the band starts at the same shade so image and band read as one.
    const SHADE = 'linear-gradient(rgba(0, 0, 0, 0.66), rgba(0, 0, 0, 0) 90%)';

    // Hover: the image's centre line of pixels runs out of its lower edge, as
    // if the image continued downward.
    function card(cardEl, frame, img, item) {
        const src = fromItem(item);
        if (!src) return;
        const band = document.createElement('span');
        band.className = 'card-band';
        cardEl.appendChild(band);
        cardEl.addEventListener('pointerenter', () => {
            const w = frame.clientWidth, h = frame.clientHeight;
            if (!img.naturalWidth || !w) return;
            const f = coverFit(img, w, h, src.at);
            band.style.backgroundImage = `${SHADE}, ${cssUrl(img.currentSrc || img.src)}`;
            band.style.backgroundSize = `100% 100%, ${(f.dw / w) * 100}% ${f.nh * 100}%`;
            band.style.backgroundPosition = `0 0, 50% ${(f.row / (f.nh - 1)) * 100}%`;
        });
    }

    // ── Gesture: collecting ─────────────────────────────────────────────────
    // The band under the hovered image (its centre line) rises until it covers
    // the whole thumbnail: the image becomes its line of pixels, at its exact
    // size. Then, in one continuous glide (no pause, no waypoint), it flies
    // into the collection, narrowing into its segment (the same line). The
    // image fades back in behind it. Only ever one row of pixels, stretched in
    // height: the image itself is never distorted.
    // dest: an element, or a DOMRect when the destination is only a place.
    const FOLD_MS = 900;
    const RISE = 0.32;   // share of the run spent covering the thumbnail
    function fold(fromImg, item, dest, done) {
        const b = dest instanceof Element ? dest.getBoundingClientRect() : dest;
        const frame = fromImg?.closest('.card-img') || fromImg;
        const a = frame?.getBoundingClientRect();
        if (!motion() || !a || !b || !b.height || b.bottom < 0 || b.top > innerHeight || !fromImg.naturalWidth) { done(); return; }
        const cardEl = fromImg.closest('.card');
        const f = coverFit(fromImg, a.width, a.height, (fromItem(item) || { at: 0.5 }).at);
        const ghost = document.createElement('div');
        ghost.className = 'filet-fold';
        ghost.style.backgroundImage = `${SHADE}, ${cssUrl(fromImg.currentSrc || fromImg.src)}`;
        document.body.appendChild(ghost);

        // Every frame in px. One source row fills the band's height; across, the
        // row keeps the card's framing, scaled to the band's width. `shade` is
        // how much of the hover overlay's gradient is still on it.
        const k = (x, y, w, h, row, shade) => ({
            left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px`,
            backgroundSize: `100% ${shade * 100}%, ${(f.dw * w) / a.width}px ${f.nh * h}px`,
            backgroundPosition: `0 0, ${(f.ox * w) / a.width}px ${-row * h}px`,
        });
        const start = cardEl?.matches(':hover') ? 13 : 0;   // the hover band (CSS .card-band), if it is out
        cardEl?.classList.add('is-collecting');
        // Each leg ends at rest, so the lift flows into the flight without a jolt.
        ghost.animate([
            { ...k(a.left, a.bottom - 1, a.width, start, f.row, 1), easing: 'cubic-bezier(0.3, 0, 0.2, 1)' },
            { ...k(a.left, a.top, a.width, a.height, f.row, 0), offset: RISE, easing: 'cubic-bezier(0.6, 0, 0.15, 1)' },
            { ...k(b.left, b.top, b.width, b.height, f.row, 0) },
        ], { duration: FOLD_MS, fill: 'forwards' }).onfinish = () => { ghost.remove(); done(); };
        // Hidden while covered, back in a fade once the band has left.
        fromImg.animate([
            { opacity: 1 }, { opacity: 1, offset: RISE - 0.01 }, { opacity: 0, offset: RISE },
            { opacity: 0, offset: RISE + 0.12 }, { opacity: 1, offset: 0.8 }, { opacity: 1 },
        ], { duration: FOLD_MS });
        setTimeout(() => cardEl?.classList.remove('is-collecting'), FOLD_MS);
    }

    // A collection's new segment appears as the band arrives.
    function reveal(seg) {
        if (!seg || !motion()) return;
        seg.animate([{ clipPath: 'inset(0 100% 0 0)' }, { clipPath: 'inset(0 0 0 0)' }],
            { duration: 420, easing: 'cubic-bezier(0.16, 1, 0.3, 1)' });
    }

    // ── Loading: an ambient band travels along the top of the results ─────────
    function loading(on) { document.body.classList.toggle('is-fetching', on); }

    document.querySelectorAll('[data-ambient]').forEach(el => paintAmbient(el, false));

    return { paint, fromItem, generic, setAmbient, card, fold, reveal, loading, setMotion, motion };
})();
