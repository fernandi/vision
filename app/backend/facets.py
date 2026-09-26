"""
Period and technique facets, read from the free-text descriptions (captionEn)
and titles, since the museums' data has no structured date or medium field.

Each work gets two bitmasks: PERIODS and TECHNIQUES (a work may have several
bits, or none when the description says nothing usable).
"""
import re

PERIODS = ["antiquity", "middle_ages", "1400_1600", "1600_1800", "19th", "20th"]
TECHNIQUES = ["painting", "drawing", "print", "photograph", "sculpture", "ceramics",
              "textile", "metal", "glass", "furniture", "book"]

# Year → period bit (BC years are negative).
def _period_of_year(y):
    if y < 500:
        return 0
    if y < 1400:
        return 1
    if y < 1600:
        return 2
    if y < 1800:
        return 3
    if y < 1900:
        return 4
    return 5


CENTURY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)[ -]centur(?:y|ies)(\s*(?:b\.?c\.?|bce))?", re.I)
DECADE = re.compile(r"\b(1[0-9]{2}0|20[0-2]0)s\b")
YEAR = re.compile(r"(?<![\d.])(1[0-9]{3}|20[0-2][0-9])(?![\d.])")

# Cultures and periods named in descriptions, mapped to period bits.
PERIOD_WORDS = [
    (re.compile(r"\b(greek and roman art|egyptian art|ancient near eastern art|ancient|attic|etruscan|cypriot|"
                r"hellenistic|archaic|classical greek|roman imperial|bronze age|iron age|neolithic|"
                r"old kingdom|middle kingdom|new kingdom|ptolemaic|mesopotamian|assyrian|achaemenid)\b"), [0]),
    (re.compile(r"\b(medieval art|medieval|byzantine|romanesque|gothic|carolingian|merovingian|"
                r"heian period|kamakura period|song dynasty|tang dynasty|yuan dynasty)\b"), [1]),
    (re.compile(r"\b(renaissance|ming dynasty|muromachi period|momoyama period|safavid|timurid)\b"), [2]),
    (re.compile(r"\b(baroque|rococo|qing dynasty|mughal)\b"), [3]),
    (re.compile(r"\bedo period\b"), [3, 4]),
    (re.compile(r"\b(meiji period|victorian)\b"), [4]),
    (re.compile(r"\b(taisho period|showa period|art deco|modernis[mt])\b"), [5]),
]

# Department and collection names that say nothing about the object itself.
BOILERPLATE = re.compile(r"\b(drawings and prints|prints and drawings?|drawings \(visual works\)|european sculpture and "
                         r"decorative arts|decorative arts|applied arts of europe|robert lehman collection|the american wing|"
                         r"smithsonian design museum|cooper hewitt|national portrait gallery|google cultural institute)\b")
# Materials that belong to another family in context (painting on silk, photographic silver…).
CONTEXT = [
    (re.compile(r"\b(ink|colou?rs?|pigments?|gold)( and \w+)* on silk\b"), " painting "),
    (re.compile(r"\b(hanging scroll|handscroll|album leaf|folding screen)\b"), " painting "),
    (re.compile(r"\b(gelatin silver|albumen silver|silver print|silver gelatin|glass (plate )?negative)\b"), " photograph "),
]

# Technique families: matched on description fragments and titles.
TECHNIQUE_WORDS = [
    ("painting", r"painting|paintings|oil on canvas|oil on panel|oil on wood|oil on paper|tempera|gouache|acrylic|fresco"),
    ("drawing", r"drawing|drawings|graphite|charcoal|chalk|pen and \w+ ink|watercolou?r|pastel|sketch|sketchbook"),
    ("print", r"print|prints|etching|engraving|lithograph|woodcut|woodblock print|aquatint|mezzotint|"
              r"screenprint|drypoint|linocut|photomechanical print"),
    ("photograph", r"photograph|photographs|albumen|gelatin silver|daguerreotype|cyanotype|platinum print|"
                   r"salted paper print|ambrotype|tintype|carte-de-visite|cabinet card|stereograph"),
    ("sculpture", r"sculpture|statue|statuette|figurine|relief|bust|carving|carved|ivory"),
    ("ceramics", r"ceramic|ceramics|porcelain|earthenware|stoneware|terracotta|faience|pottery|"
                 r"kylix|krater|amphora|lekythos|hydria|oinochoe|maiolica|delftware"),
    ("textile", r"textile|textiles|silk|cotton|linen|lace|embroidery|embroidered|tapestry|costume|wool|"
                r"velvet|brocade|damask|sampler|quilt"),
    ("metal", r"silver|bronze|steel|iron|brass|pewter|copper|jewelry|jewellery|arms and armor|metalwork|goldsmith"),
    ("glass", r"glass|stained glass"),
    ("furniture", r"furniture|armchair|side chair|cabinet|chest of drawers|commode|secretary desk|settee|sofa|"
                  r"bedstead|high chest|dressing table|card table|tea table|looking glass"),
    ("book", r"manuscript|illuminated|sample book|book of hours|codex|bound volume|printed book"),
]
TECHNIQUE_RE = [(TECHNIQUES.index(name), re.compile(rf"\b({words})\b")) for name, words in TECHNIQUE_WORDS]


def extract(title, caption):
    """→ (period_mask, technique_mask) for one work."""
    text = BOILERPLATE.sub(" ", f"{title or ''}, {caption or ''}".lower())
    periods = 0
    for m in CENTURY.finditer(text):
        c = int(m.group(1))
        if m.group(2):                      # BC
            periods |= 1 << 0
        elif 1 <= c <= 21:
            periods |= 1 << _period_of_year(c * 100 - 50)
    for m in DECADE.finditer(text):
        periods |= 1 << _period_of_year(int(m.group(1)))
    if not periods:
        years = [int(y) for y in YEAR.findall(text)]
        for y in years[:3]:
            periods |= 1 << _period_of_year(y)
    for pattern, bits in PERIOD_WORDS:
        if pattern.search(text):
            for b in bits:
                periods |= 1 << b

    for pattern, replacement in CONTEXT:
        text = pattern.sub(replacement, text)
    techniques = 0
    for bit, pattern in TECHNIQUE_RE:
        if pattern.search(text):
            techniques |= 1 << bit
    return periods, techniques
