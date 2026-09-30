"""KisanFlow Phase 2 - NLP language-understanding & word-prediction layer.

Pipeline position (requirement 10)
----------------------------------
    speech-to-text (browser Web Speech API, static/voice.js)
        -> text normalization          )
        -> intent detection            )  this module (nlp.py)
        -> entity extraction           )
        -> conversation state          )
        -> Flask API                   (/api/nlp/parse, /api/nlp/suggest)
        -> centre rules + database     (app.py - unchanged, the authority)
        -> response generation         (i18n keys, rendered in the farmer's language)
        -> text-to-speech              (static/voice.js)

Architecture
------------
A practical, offline, inspectable rule/lexicon pipeline instead of training a
model: curated English/Telugu/Hindi lexicons, edit-distance + phonetic keys for
speech-recognition repair, a small number/date grammar, explicit confidence
scores and a language-independent conversation state. No paid API, no model
download, no build step - it runs inside the existing Flask process.

SCOPE (requirement 11)
----------------------
This module ONLY understands the farmer's words. It never reads or writes
tables and never decides whether a centre can take the produce on a day - that
authority stays in app.py and the database. The voice flow always asks the
existing GET /api/slots and books through POST /api/book.

Confidence policy (requirement 8)
---------------------------------
    HIGH   >= 0.85   accept the value silently
    MEDIUM >= 0.60   echo it back and ask "did you say ...?"
    LOW    <  0.60   ask the farmer to repeat or clarify - never guess

The conversation state stores canonical, language-independent values
('Rice', 'kg', '2026-10-05'); the language only affects the wording that the
front-end renders from the i18n dictionary (requirement 9).
"""
import re
import unicodedata
from datetime import date, timedelta

CONF_HIGH = 0.85          # accept
CONF_MEDIUM = 0.60        # confirm with the farmer
MIN_FRAGMENT = 2          # shorter fragments are never predicted

# --- digits: Telugu / Devanagari numerals are common in typed input ----------
DIGIT_MAP = {'౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4',
             '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9',
             '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
             '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'}

# --- units: every spelling a farmer may use maps to one canonical unit -------
UNIT_VARIANTS = {
    'kg': ['kg', 'kgs', 'kilo', 'kilos', 'kilogram', 'kilograms', 'kilogramme',
           'కిలో', 'కిలోల', 'కిలోలు', 'కేజీ', 'కేజీల', 'కేజీలు', 'కిలోస్',
           'किलो', 'किलोग्राम', 'किलोस'],
    'ton': ['ton', 'tons', 'tonne', 'tonnes', 'టన్', 'టన్న', 'టన్ను', 'టన్నుల',
            'टन', 'टन्स'],
    'quintal': ['quintal', 'quintals', 'qtl', 'క్వింటాల్', 'క్వింటాళ్ల', 'क्विंटल'],
    'bag': ['bag', 'bags', 'బస్తా', 'బస్తాలు', 'बोरी', 'बोरियां'],
}
UNIT_KG = {'kg': 1.0, 'ton': 1000.0, 'quintal': 100.0}   # 'bag' has no fixed weight

# --- crops: the canonical values are exactly what the booking API accepts ----
CROP_VARIANTS = {
    'Rice': ['rice', 'paddy', 'రైస్', 'బియ్యం', 'వరి', 'ధాన్యం', 'चावल', 'धान', 'राइस'],
    'Wheat': ['wheat', 'గోధుమ', 'గోధుమలు', 'गेहूं', 'गेहू', 'गेंहू', 'व्हीट'],
    'Maize': ['maize', 'corn', 'మొక్కజొన్న', 'కార్న్', 'मक्का', 'मकई', 'भुट्टा'],
    'Cotton': ['cotton', 'కపాస్', 'పత్తి', 'कपास', 'कॉटन'],
}
# recognised produce outside the four priced crops -> 'Other' (needs confirmation)
OTHER_CROP_VARIANTS = ['groundnut', 'peanut', 'soybean', 'soya', 'sunflower', 'sugarcane',
                       'turmeric', 'chilli', 'chili', 'onion', 'pulses', 'gram',
                       'వేరుశనగ', 'పొద్దుతిరుగుడు', 'చెరకు', 'పసుపు', 'మిరప', 'ఉల్లి',
                       'मूंगफली', 'सोयाबीन', 'सूरजमुखी', 'गन्ना', 'हल्दी', 'मिर्च', 'प्याज']

# Curated speech-recognition repairs (requirement 3). A recogniser cannot be
# trusted to spell produce names, so the common mis-hearings are listed instead
# of guessed - a wrong crop would silently change what the farmer is selling.
# They are scored slightly below a real word and only reach "high" confidence
# when the assistant is already asking about the crop (context, requirement 4).
CROP_STT_VARIANTS = {
    'Rice': ['rais', 'raiz', 'rise', 'rize', 'ryse', 'raice', 'risee'],
    'Wheat': ['veet', 'weet', 'wheet', 'whete', 'weat', 'vheet'],
    'Maize': ['mays', 'mais', 'mayes', 'mize', 'mayze'],
    'Cotton': ['coton', 'koton', 'cotten', 'katton', 'kataan'],
}
UNIT_STT_VARIANTS = {
    'kg': ['kilow', 'kilows', 'keelo', 'kiloes', 'kilogrem', 'kaygee'],
    'ton': ['tonn', 'tones', 'tawn', 'tonns'],
    'quintal': ['quintel', 'kintal', 'quintle'],
}

MONTHS = {
    'january': 1, 'jan': 1, 'జనవరి': 1, 'जनवरी': 1,
    'february': 2, 'feb': 2, 'ఫిబ్రవరి': 2, 'फरवरी': 2, 'फ़रवरी': 2,
    'march': 3, 'mar': 3, 'మార్చి': 3, 'मार्च': 3,
    'april': 4, 'apr': 4, 'ఏప్రిల్': 4, 'अप्रैल': 4,
    'may': 5, 'మే': 5, 'मई': 5,
    'june': 6, 'jun': 6, 'జూన్': 6, 'जून': 6,
    'july': 7, 'jul': 7, 'జూలై': 7, 'जुलाई': 7,
    'august': 8, 'aug': 8, 'ఆగస్టు': 8, 'अगस्त': 8,
    'september': 9, 'sep': 9, 'sept': 9, 'సెప్టెంబర్': 9, 'సెప్టెంబరు': 9, 'सितंबर': 9,
    'october': 10, 'oct': 10, 'అక్టోబర్': 10, 'అక్టోబరు': 10, 'अक्टूबर': 10,
    'november': 11, 'nov': 11, 'నవంబర్': 11, 'नवंबर': 11, 'नवम्बर': 11,
    'december': 12, 'dec': 12, 'డిసెంబర్': 12, 'दिसंबर': 12,
}
MONTH_NAMES = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November',
               12: 'December'}

NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
    'ninety': 90,
    'ఒకటి': 1, 'రెండు': 2, 'మూడు': 3, 'నాలుగు': 4, 'ఐదు': 5, 'ఆరు': 6, 'ఏడు': 7,
    'ఎనిమిది': 8, 'తొమ్మిది': 9, 'పది': 10, 'ఇరవై': 20, 'ముప్పై': 30, 'నలభై': 40,
    'యాభై': 50, 'అరవై': 60, 'డెబ్బై': 70, 'ఎనభై': 80, 'తొంభై': 90,
    'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पांच': 5, 'छह': 6, 'सात': 7, 'आठ': 8,
    'नौ': 9, 'दस': 10, 'ग्यारह': 11, 'बारह': 12, 'बीस': 20, 'तीस': 30, 'चालीस': 40,
    'पचास': 50, 'साठ': 60, 'सत्तर': 70, 'अस्सी': 80, 'नब्बे': 90,
}
MULTIPLIERS = {'hundred': 100, 'hundreds': 100, 'వంద': 100, 'వందల': 100, 'వందలు': 100,
               'सौ': 100, 'thousand': 1000, 'వెయ్యి': 1000, 'हज़ार': 1000, 'हजार': 1000}
HALF_WORDS = {'half': 0.5, 'సగం': 0.5, 'आधा': 0.5}

RELATIVE_PHRASES = [('day after tomorrow', 2), ('next week', 7),
                    ('ఎల్లుండి', 2), ('వచ్చే వారం', 7), ('परसों', 2),
                    ('अगले हफ्ते', 7), ('अगले सप्ताह', 7)]
RELATIVE_WORDS = {'today': 0, 'tonight': 0, 'నేడు': 0, 'ఈరోజు': 0, 'आज': 0,
                  'tomorrow': 1, 'రేపు': 1, 'కల': 1, 'कल': 1}
WEEKDAYS = {'monday': 0, 'mon': 0, 'సోమవారం': 0, 'सोमवार': 0,
            'tuesday': 1, 'tue': 1, 'మంగళవారం': 1, 'मंगलवार': 1,
            'wednesday': 2, 'wed': 2, 'బుధవారం': 2, 'बुधवार': 2,
            'thursday': 3, 'thu': 3, 'గురువారం': 3, 'गुरुवार': 3,
            'friday': 4, 'fri': 4, 'శుక్రవారం': 4, 'शुक्रवार': 4,
            'saturday': 5, 'sat': 5, 'శనివారం': 5, 'शनिवार': 5,
            'sunday': 6, 'sun': 6, 'ఆదివారం': 6, 'रविवार': 6}
SLOT_PARTS = {'morning': 'morning', 'ఉదయం': 'morning', 'सुबह': 'morning',
              'afternoon': 'afternoon', 'మధ్యాహ్నం': 'afternoon', 'दोपहर': 'afternoon',
              'evening': 'evening', 'సాయంత్రం': 'evening', 'शाम': 'evening'}
SLOT_ORDINALS = {'first': 1, 'మొదటి': 1, 'पहला': 1, 'last': -1, 'చివరి': -1, 'आखिरी': -1}
TIME_RE = re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|గంటలకు|గంటకు|बजे)', re.I)
ORDINAL_RE = re.compile(r'(\d+)\s*(st|nd|rd|th)\b', re.I)

# --- intents (requirement 5): weighted phrase/keyword evidence ---------------
INTENT_PATTERNS = {
    'book_slot': [('want to book', 4), ('book a slot', 4), ('book slot', 4), ('booking', 3),
                  ('book', 2), ('slot', 1), ('sell', 2), ('bring', 1),
                  ('బుక్', 3), ('స్లాట్', 1), ('అమ్మ', 2), ('తీసుకొస్త', 1),
                  ('बुक', 3), ('स्लॉट', 1), ('बेच', 2), ('लाना', 1)],
    'check_available_slots': [('how many slots', 4), ('free slots', 4), ('slots left', 4),
                              ('available', 3), ('empty', 2), ('slots', 1),
                              ('ఖాళీ', 3), ('అందుబాటు', 3), ('खाली', 3), ('उपलब्ध', 3)],
    'check_status': [('my turn', 4), ('my status', 4), ('when is my', 3), ('status', 3),
                     ('my token', 3), ('token', 2), ('queue', 2),
                     ('నా వంతు', 4), ('స్థితి', 3), ('నా టోకెన్', 3), ('టోకెన్', 2),
                     ('मेरी बारी', 4), ('स्थिति', 3), ('मेरा टोकन', 3), ('टोकन', 2)],
    'cancel_booking': [('cancel my booking', 5), ('cancel', 3), ('రద్దు', 4), ('रद्द', 4)],
    'confirm_booking': [('yes', 3), ('okay', 3), ('ok', 2), ('confirm', 3), ('sure', 2),
                        ('అవును', 3), ('సరే', 3), ('हाँ', 3), ('ठीक', 3)],
    'deny': [('no thanks', 4), ('not now', 3), ('no', 2), ('dont', 2),
             ('కాదు', 3), ('వద్దు', 3), ('नहीं', 3)],
    'change_date': [('change the date', 5), ('change date', 5), ('another date', 4),
                    ('different date', 4), ('date change', 4), ('వేరే తేదీ', 4),
                    ('తేదీ మార్చు', 5), ('दूसरी तारीख', 4), ('तारीख बदल', 5)],
    'change_crop': [('change the crop', 5), ('change crop', 5), ('another crop', 4),
                    ('different crop', 4), ('వేరే పంట', 4), ('పంట మార్చు', 5),
                    ('दूसरी फसल', 4), ('फसल बदल', 5)],
    'change_quantity': [('change the quantity', 5), ('change quantity', 5),
                        ('another quantity', 4), ('different quantity', 4),
                        ('పరిమాణం మార్చు', 5), ('मात्रा बदल', 5)],
    'help': [('what can you do', 4), ('help me', 4), ('help', 3), ('సహాయం', 3),
             ('మదదు', 3), ('मदद', 3), ('मदद करो', 4)],
    'centre_info': [('centre timings', 4), ('center timing', 4), ('opening hours', 4),
                    ('centre', 2), ('center', 2), ('timing', 2),
                    ('కేంద్రం', 2), ('केंद्र', 2), ('समय', 2)],
}

# --- derived lookup tables (built once, at import) --------------------------
NOISE_WORDS = {'a', 'an', 'the', 'is', 'am', 'are', 'to', 'of', 'for', 'on', 'at', 'in', 'and',
               'my', 'me', 'i', 'please', 'pls', 'about', 'around', 'approximately', 'some',
               'want', 'need', 'would', 'like', 'will', 'can', 'dont', 'not',
               'కు', 'కి', 'న', 'ని', 'ఒక', 'మరియు', 'కావాలి', 'ఉంది',
               'को', 'के', 'में', 'है', 'और', 'पर', 'तारीख', 'चाहिए', 'करना'}

_UNIT_LOOKUP = {}
for _canon, _variants in UNIT_VARIANTS.items():
    for _v in _variants:
        _UNIT_LOOKUP[_v] = _canon
_CROP_LOOKUP = {}
for _canon, _variants in CROP_VARIANTS.items():
    for _v in _variants:
        _CROP_LOOKUP[_v] = _canon
for _v in OTHER_CROP_VARIANTS:
    _CROP_LOOKUP[_v] = 'Other'
_CROP_STT_LOOKUP = {}
for _canon, _variants in CROP_STT_VARIANTS.items():
    for _v in _variants:
        _CROP_STT_LOOKUP[_v] = _canon
_UNIT_STT_LOOKUP = {}
for _canon, _variants in UNIT_STT_VARIANTS.items():
    for _v in _variants:
        _UNIT_STT_LOOKUP[_v] = _canon


def _build_prefix_index(canonical_variants, min_len=MIN_FRAGMENT):
    """prefix -> {canonical value}: drives prediction and clarification (req. 1)."""
    index = {}
    for canon, variants in canonical_variants.items():
        for variant in variants:
            for size in range(min_len, len(variant) + 1):
                index.setdefault(variant[:size], set()).add(canon)
    return {k: sorted(v) for k, v in index.items()}


CROP_PREFIXES = _build_prefix_index(CROP_VARIANTS)
UNIT_PREFIXES = _build_prefix_index(UNIT_VARIANTS)
MONTH_PREFIXES = _build_prefix_index({MONTH_NAMES[m]: [n] for n, m in MONTHS.items()
                                      if len(n) >= 3})


KEEP_CHARS = set('/:.#+-')


def strip_symbols(text):
    """Blank out punctuation while keeping letters, digits and combining marks.

    '.' survives only as a decimal point between digits: a sentence-final full
    stop must not glue itself onto the word in front of it, otherwise 'kilos.'
    stops being an exact lexicon hit and a final unit or crop name is only ever
    fuzzy-matched (or missed completely when it is the last word in the box).
    """
    out = []
    last = len(text) - 1
    for index, char in enumerate(text):
        keep = char in KEEP_CHARS
        if char == '.':
            keep = 0 < index < last and text[index - 1].isdigit() and text[index + 1].isdigit()
        if keep or char.isspace() or unicodedata.category(char)[0] in ('L', 'N', 'M'):
            out.append(char)
        else:
            out.append(' ')
    return ''.join(out)


def normalize(text):
    """Text normalization (requirement 2): case, digits, ordinals, punctuation.

    Indic vowel signs are combining marks (Unicode category Mn) which Python's
    \\w does not match, so the punctuation filter is character-category based -
    otherwise a Telugu word such as బియ్యం would be torn apart.
    """
    raw = text or ''
    s = raw.strip().lower()
    for bad, good in DIGIT_MAP.items():
        s = s.replace(bad, good)
    s = ORDINAL_RE.sub(r'\1', s)                          # 5th / 5th, -> 5
    s = re.sub(r'(\d+)\s*(న|వ|को|वीं|वें|తారీఖు)\b', r'\1', s)   # "5న" / "5 को" -> 5
    s = re.sub(r'\s+', ' ', strip_symbols(s)).strip()
    return {'raw': raw, 'text': s, 'tokens': [t for t in s.split(' ') if t]}


def levenshtein(a, b):
    """Small edit distance - the basis of the fuzzy correction (requirement 3)."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


PHONETIC_PAIRS = (('ph', 'f'), ('ck', 'k'), ('qu', 'k'), ('z', 's'), ('v', 'w'), ('c', 'k'),
                  ('ee', 'i'), ('oo', 'u'), ('ai', 'e'), ('ay', 'e'), ('ou', 'u'), ('ow', 'o'))


def phonetic_key(word):
    """Rough English sound-alike key: 'kilows' and 'kilos' collapse together."""
    w = re.sub(r'[^a-z]', '', (word or '').lower())
    for a, b in PHONETIC_PAIRS:
        w = w.replace(a, b)
    return re.sub(r'(.)\1+', r'\1', w)


def _script(word):
    return 'latin' if all(ord(c) < 128 for c in word) else 'indic'


def _distance_limit(token):
    """How many edits we allow before we stop believing a word (req. 3)."""
    return 1 if len(token) <= 5 else (2 if len(token) <= 8 else 3)


def match_token(token, lookup, expected=False):
    """Map one spoken word onto a lexicon value.

    Returns {'value', 'confidence', 'alternates'} or None. Exact, phonetic and
    edit-distance hits are scored; a short fragment is never guessed - it comes
    back as a low-confidence candidate so the assistant can ask instead
    (requirements 1, 3 and 8).
    """
    tok = (token or '').strip()
    if not tok or len(tok) < 3:
        return None                       # 3+ characters before any fuzzy match
    if tok in lookup:
        return {'value': lookup[tok], 'confidence': 1.0, 'alternates': [], 'distance': 0}
    limit = _distance_limit(tok) + (1 if expected else 0)   # context widens it (req. 4)
    best, best_distance, ties = None, None, []
    key = phonetic_key(tok)
    for candidate, value in lookup.items():
        if len(candidate) < 3 or _script(candidate) != _script(tok):
            continue
        distance = levenshtein(tok, candidate)
        if distance == 0 or (key and key == phonetic_key(candidate)):
            distance = 1
        if distance > limit:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance, ties = value, distance, [value]
        elif distance == best_distance and value not in ties:
            ties.append(value)
    if best is None:
        return None
    confidence = round(max(0.3, 1.0 - 0.15 * best_distance), 2)
    if len(ties) > 1:
        confidence = min(confidence, 0.5)          # ambiguous -> ask, never assume
    return {'value': best, 'confidence': confidence, 'alternates': sorted(set(ties)),
            'distance': best_distance}


def clean_number(value):
    return int(value) if float(value).is_integer() else round(float(value), 3)


def parse_number_at(tokens, index):
    """Read a quantity starting at tokens[index] ('500', '5 hundred', 'five hundred')."""
    if index >= len(tokens):
        return None, index, 0.0
    tok = tokens[index]
    if re.fullmatch(r'\d+(\.\d+)?', tok):
        value, nxt = float(tok), index + 1
        if nxt < len(tokens) and tokens[nxt] in MULTIPLIERS:
            value *= MULTIPLIERS[tokens[nxt]]
            nxt += 1
        return clean_number(value), nxt, 1.0
    if tok in NUMBER_WORDS:
        value, nxt = float(NUMBER_WORDS[tok]), index + 1
        if nxt < len(tokens) and tokens[nxt] in MULTIPLIERS:
            value *= MULTIPLIERS[tokens[nxt]]
            nxt += 1
        if nxt < len(tokens) and tokens[nxt] in HALF_WORDS:      # "one and a half"
            value += HALF_WORDS[tokens[nxt]]
            nxt += 1
        return clean_number(value), nxt, 0.95
    if tok in MULTIPLIERS:
        return clean_number(MULTIPLIERS[tok]), index + 1, 0.8
    return None, index, 0.0


def parse_quantity(tokens, expected=False):
    """Extract quantity + unit and convert to kilograms (requirements 2 and 6).

    Returns (value_in_kg, unit, confidence, ambiguous_unit_options). When the
    farmer gives a bare number the value comes back with unit=None and a lower
    confidence so the caller can apply the unit the assistant just asked in, or
    ask which unit was meant (requirements 4 and 8).
    """
    units = []
    for i, tok in enumerate(tokens):
        hit = _UNIT_LOOKUP.get(tok) or _UNIT_STT_LOOKUP.get(tok)
        if hit:
            units.append((i, hit, 1.0 if tok in _UNIT_LOOKUP else 0.88))
            continue
        mapped = match_token(tok, _UNIT_LOOKUP, expected)
        if mapped and mapped['confidence'] >= CONF_MEDIUM:
            units.append((i, mapped['value'], mapped['confidence']))
    for i, unit, unit_conf in units:
        start = i - 1
        if start >= 1 and tokens[start] in MULTIPLIERS:        # "5 hundred kg"
            start -= 1
        value, _nxt, num_conf = parse_number_at(tokens, start)
        if value is None:
            continue
        factor = UNIT_KG.get(unit)
        if factor is None:                     # 'bag' has no agreed weight
            return value, unit, round(0.5 * unit_conf * num_conf, 2), ['kg', 'ton']
        confidence = round(min(0.99, 0.9 * unit_conf + 0.1 * num_conf), 2)
        return clean_number(value * factor), unit, confidence, []
    for i, _tok in enumerate(tokens):          # a bare number, unit still unknown
        value, _nxt, num_conf = parse_number_at(tokens, i)
        if value is None or value <= 0:
            continue
        return clean_number(value), None, round(0.7 * num_conf, 2), []
    return None, None, 0.0, []


def _resolve_date(year, month, day, today, confidence):
    """Resolve month/day to ISO; a past day rolls to next year (lower confidence)."""
    year = year or today.year
    try:
        candidate = date(year, month, day)
    except ValueError:
        return None, 0.0, 'invalid_date'
    note = ''
    if candidate < today:
        try:
            candidate = date(year + 1, month, day)
        except ValueError:
            return None, 0.0, 'invalid_date'
        confidence = round(confidence * 0.9, 2)
        note = 'rolled_to_next_year'
    return candidate.isoformat(), confidence, note


def parse_date(normalized_text, tokens, today=None):
    """Extract a booking date in EN/TE/HI (requirements 6 and 13).

    Returns (iso_date, confidence, note) - the note explains a rollover or why
    the day had to be asked for again.
    """
    today = today or date.today()
    match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', normalized_text)
    if match:
        try:
            fixed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return fixed.isoformat(), 0.98, ''
        except ValueError:
            return None, 0.0, 'invalid_date'
    match = re.search(r'\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b', normalized_text)
    if match:
        year = int(match.group(3)) if match.group(3) else None
        if year is not None and year < 100:
            year += 2000
        return _resolve_date(year, int(match.group(2)), int(match.group(1)), today, 0.92)
    for index, token in enumerate(tokens):
        if token not in MONTHS:
            continue
        month = MONTHS[token]
        day = None
        for offset in (1, 2, 3, -1, -2):
            if 0 <= index + offset < len(tokens) and re.fullmatch(r'\d{1,2}', tokens[index + offset]):
                value = int(tokens[index + offset])
                if 1 <= value <= 31:
                    day = value
                    break
        if day is None:
            return None, 0.0, 'day_missing'
        return _resolve_date(None, month, day, today, 0.97)
    for phrase, offset in RELATIVE_PHRASES:
        if phrase in normalized_text:
            return (today + timedelta(days=offset)).isoformat(), 0.95, ''
    for token in tokens:
        if token in RELATIVE_WORDS:
            return (today + timedelta(days=RELATIVE_WORDS[token])).isoformat(), 0.95, ''
    for token in tokens:
        if token in WEEKDAYS:
            delta = (WEEKDAYS[token] - today.weekday()) % 7 or 7
            return (today + timedelta(days=delta)).isoformat(), 0.75, 'weekday'
    return None, 0.0, ''


def parse_slot_hint(normalized_text, tokens):
    """Optional slot preference: 'morning' / '9 am' / 'first slot' (requirement 6)."""
    hint = {}
    for token in tokens:
        if token in SLOT_PARTS:
            hint['part'] = SLOT_PARTS[token]
        if token in SLOT_ORDINALS:
            hint['ordinal'] = SLOT_ORDINALS[token]
    match = TIME_RE.search(normalized_text)
    if match:
        hour = int(match.group(1))
        if (match.group(3) or '').lower() == 'pm' and hour < 12:
            hour += 12
        hint['hour'] = hour
        hint['minute'] = int(match.group(2) or 0)
    return hint or None


def parse_token_number(normalized_text):
    """'token 12' / '#12' / 'నా టోకెన్ 12' (requirement 6)."""
    match = re.search(r'(?:token|టోకెన్|टोकन|#)\s*#?(\d{1,4})', normalized_text)
    if match:
        return int(match.group(1)), 0.95
    return None, 0.0


def detect_intent(normalized_text, tokens, expected=None, has_entities=False):
    """Weighted, context-boosted intent detection (requirement 5).

    Returns (intent, score, confidence, meta). Multi-word evidence is matched on
    the normalized text, single words are matched (with a small edit-distance
    allowance) on the tokens, so a mis-heard 'boke' still reaches book_slot.
    """
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for phrase, weight in patterns:
            if ' ' in phrase:
                if phrase in normalized_text:
                    score += weight
            elif phrase in tokens:
                score += weight
            elif len(phrase) >= 5 and any(match_token(t, {phrase: phrase}) for t in tokens):
                score += max(1, weight - 1)
        if score:
            scores[intent] = score
    if expected == 'confirm':                       # a yes/no is only a yes/no here
        for key in ('confirm_booking', 'deny'):
            if key in scores:
                scores[key] += 3
    if not scores:
        if has_entities:                            # entities alone imply a booking
            return 'book_slot', 3, 0.7, {'implicit': True, 'scores': {}}
        return 'unknown', 0, 0.3, {'scores': {}}
    best = max(scores, key=lambda key: (scores[key], key))
    confidence = round(min(0.96, 0.55 + 0.09 * scores[best]), 2)
    return best, scores[best], confidence, {'scores': scores}


def analyze(text, state=None, expected=None, today=None):
    """Understand one utterance: intent + entities + confidence (req. 4, 6, 8).

    `expected` is the entity type the assistant last asked for, which is what
    makes 'ri...' mean rice in a crop question and '500' mean kilograms right
    after "how many kilograms?" (requirements 1 and 4). Pure function: no I/O,
    no database, no centre rules - availability is decided later by the API.
    """
    state = state or {}
    today = today or date.today()
    normalized = normalize(text)
    tokens = normalized['tokens']
    content = [t for t in tokens if t not in NOISE_WORDS]
    entities, confidence, corrections, ambiguous = {}, {}, [], []

    for token in content:
        hit = match_token(token, _CROP_LOOKUP, expected == 'crop')
        if hit is None and token in _CROP_STT_LOOKUP:
            # curated mis-hearing: trust it fully when the assistant is already
            # asking about the crop, otherwise confirm it first (req. 3, 4 and 8)
            hit = {'value': _CROP_STT_LOOKUP[token], 'distance': 1, 'alternates': [],
                   'confidence': 0.87 if expected == 'crop' else 0.8}
        if not hit:
            continue
        entities['crop'] = hit['value']
        confidence['crop'] = hit['confidence']
        if hit['distance']:
            corrections.append({'from': token, 'to': hit['value'], 'kind': 'spelling',
                                'confidence': hit['confidence']})
        if hit['alternates']:
            ambiguous.append({'field': 'crop', 'seen': token, 'options': hit['alternates']})
        break

    iso, date_conf, note = parse_date(normalized['text'], content, today)
    if iso:
        entities['date'] = iso
        confidence['date'] = date_conf
        if note:
            ambiguous.append({'field': 'date', 'note': note})
    elif note:
        ambiguous.append({'field': 'date', 'note': note})
    day_of_month = None
    if iso:
        try:
            day_of_month = date.fromisoformat(iso).day
        except ValueError:
            day_of_month = None

    quantity, unit, q_conf, unit_options = parse_quantity(content, expected == 'quantity')
    if quantity is not None:
        if unit and UNIT_KG.get(unit):
            entities['quantity_kg'] = quantity                  # already in kg
            entities['unit'] = unit
            confidence['quantity'] = confidence['unit'] = q_conf
        elif unit:                                             # e.g. 'bag'
            entities['quantity_value'] = quantity
            entities['unit'] = unit
            confidence['quantity'] = confidence['unit'] = q_conf
        elif day_of_month is not None and quantity == day_of_month:
            # the number belongs to the date parsed from this same utterance
            # ("October 5"), it is never a weight - even when the assistant is
            # waiting for the quantity (a repeated date must not become 5 kg).
            # This guard has to run before the "answer the quantity question"
            # branch below, otherwise a repeated date is accepted as kilograms.
            quantity = None
        elif expected == 'quantity' and state.get('unit_frame') in UNIT_KG:
            frame = state['unit_frame']                        # the unit we just asked in
            entities['quantity_kg'] = clean_number(quantity * UNIT_KG[frame])
            entities['unit'] = frame
            confidence['quantity'] = confidence['unit'] = round(max(q_conf, 0.9), 2)
        else:
            entities['quantity_value'] = quantity
            confidence['quantity'] = q_conf
            ambiguous.append({'field': 'unit', 'seen': str(quantity),
                              'options': sorted(unit_options or ['kg', 'ton'])})
    elif unit_options:
        ambiguous.append({'field': 'unit', 'seen': ' '.join(content), 'options': unit_options})

    token_no, token_conf = parse_token_number(normalized['text'])
    if token_no is not None:
        entities['token'] = token_no
        confidence['token'] = token_conf

    hint = parse_slot_hint(normalized['text'], content)
    if hint:
        entities['slot_hint'] = hint
        confidence['slot_hint'] = 0.8

    intent, score, intent_conf, meta = detect_intent(normalized['text'], content, expected,
                                                     has_entities=bool(entities))
    if expected == 'confirm' and intent in ('confirm_booking', 'deny'):
        intent_conf = max(intent_conf, 0.9)

    merged = dict(state)
    for field in ('crop', 'date', 'unit', 'slot_hint', 'quantity_kg', 'quantity_value'):
        if field in entities:
            merged[field] = entities[field]
    missing = [field for field in ('crop', 'date', 'quantity') if not merged.get(field)
               and not merged.get('quantity_kg')]
    summary = {'intent': intent, 'date': merged.get('date'), 'crop': merged.get('crop'),
               'quantity_kg': merged.get('quantity_kg'), 'confidence': dict(confidence)}
    return {'text': normalized['raw'], 'normalized': normalized['text'], 'intent': intent,
            'intent_confidence': intent_conf, 'score': score, 'entities': entities,
            'confidence': confidence, 'corrections': corrections, 'ambiguous': ambiguous,
            'missing': missing, 'merged': merged, 'summary': summary, 'notes': meta}


ASK_KEYS = {'crop': 'nlp.ask.crop', 'date': 'nlp.ask.date', 'quantity': 'nlp.ask.quantity',
            'unit': 'nlp.ask.unit'}
CONFIRM_KEYS = {'crop': 'nlp.confirm.crop', 'date': 'nlp.confirm.date',
                'quantity': 'nlp.confirm.quantity', 'unit': 'nlp.confirm.unit',
                'quantity_kg': 'nlp.confirm.quantity', 'quantity_value': 'nlp.confirm.quantity'}
CHANGE_INTENTS = {'change_date': 'date', 'change_crop': 'crop', 'change_quantity': 'quantity'}
REPLY_INTENTS = ('cancel_booking', 'check_status', 'check_available_slots', 'help', 'centre_info')


def crop_word(canonical):
    return (CROP_VARIANTS.get(canonical) or [canonical])[0]


def human_date(iso):
    try:
        parsed = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso or ''
    return '%s %d' % (MONTH_NAMES[parsed.month], parsed.day)


def quantity_label(kg):
    return '' if kg is None else '%s kg' % kg


class Conversation(object):
    """Multi-turn dialogue state (requirements 7, 8 and 9).

    Holds canonical, language-independent values ('Rice', 'kg', '2026-10-05') and
    answers with an i18n *key* + parameters, so the very same dialogue renders in
    English, Telugu or Hindi without changing any state.
    """

    def __init__(self, state=None, today=None):
        self.today = today or date.today()
        self.state = {'crop': None, 'date': None, 'quantity_kg': None, 'quantity_value': None,
                      'unit': None, 'unit_frame': 'kg', 'slot_hint': None, 'token': None,
                      'pending': None, 'queue': [], 'language': None}
        if state:
            self.state.update(state)

    # -- state helpers ------------------------------------------------------
    def expected(self):
        """The entity the assistant is waiting for = the context (requirement 4)."""
        if self.state.get('pending'):
            return 'confirm'
        for field in ('crop', 'date', 'quantity'):
            if field == 'quantity':
                if not self.state.get('quantity_kg'):
                    return 'quantity'
            elif not self.state.get(field):
                return field
        return None

    def missing(self):
        out = []
        if not self.state.get('crop'):
            out.append('crop')
        if not self.state.get('date'):
            out.append('date')
        if not self.state.get('quantity_kg'):
            out.append('quantity')
        return out

    def snapshot(self):
        return {key: value for key, value in self.state.items() if key != 'queue'}

    def reset(self):
        self.state.update({'crop': None, 'date': None, 'quantity_kg': None,
                           'quantity_value': None, 'unit': None, 'unit_frame': 'kg',
                           'slot_hint': None, 'pending': None, 'queue': []})

    def _accept(self, field, value, _confidence):
        if field in ('quantity_kg', 'quantity_value'):
            self.state['quantity_kg'] = value if field == 'quantity_kg' else None
            self.state['quantity_value'] = value if field == 'quantity_value' else None
        else:
            self.state[field] = value

    def _hold(self, field, value, confidence):
        """Medium confidence: remember the value but ask before using it (req. 8)."""
        entry = {'field': field, 'value': value, 'confidence': confidence}
        if self.state.get('pending'):
            self.state['queue'].append(entry)
        else:
            self.state['pending'] = entry

    # -- display helpers ----------------------------------------------------
    def _confirm_params(self, field):
        params = {'confidence': (self.state.get('pending') or {}).get('confidence')}
        if field in ('quantity_kg', 'quantity_value'):
            params['quantity'] = quantity_label(self.state.get('quantity_kg') or
                                                self.state.get('quantity_value'))
        elif field == 'crop':
            params['crop'] = self.state.get('crop') or ''
        elif field == 'date':
            params['date'] = self.state.get('date') or ''
        else:
            params['value'] = self.state.get(field) or ''
        return params

    def booking_params(self):
        kg = self.state.get('quantity_kg')
        return {'crop': self.state.get('crop') or '', 'date': self.state.get('date') or '',
                'unit': self.state.get('unit') or 'kg', 'quantity_kg': kg,
                'quantity': quantity_label(kg),
                'quantity_tons': round((kg or 0) / 1000.0, 3),
                'slot_hint': self.state.get('slot_hint')}

    def _ask(self, field, change=None):
        if field == 'quantity':
            self.state['unit_frame'] = 'kg'      # the question below is asked in kilograms
        return {'action': 'ask', 'field': field,
                'question': ('nlp.change.' + field) if change else ASK_KEYS[field],
                'params': {'crop': self.state.get('crop') or '',
                           'crop_word': crop_word(self.state['crop']) if self.state.get('crop') else '',
                           'quantity': quantity_label(self.state.get('quantity_kg')),
                           'date': self.state.get('date') or ''}}

    # -- the dialogue -------------------------------------------------------
    def step(self, text):
        """Feed one farmer utterance and get the assistant's next action."""
        expected = self.expected()
        analysis = analyze(text, state=self.state, expected=expected, today=self.today)
        intent = analysis['intent']
        entities, confidence = analysis['entities'], analysis['confidence']
        notes = {'accepted': [], 'confirm': [], 'clarify': [], 'cleared': None}

        pending = self.state.get('pending')                 # 1. answer to "did you say ...?"
        if pending and expected == 'confirm':
            self.state['pending'] = None
            if intent == 'confirm_booking':
                self._accept(pending['field'], pending['value'], pending['confidence'])
                notes['accepted'].append(pending['field'])
            elif intent == 'deny':
                self.state[pending['field']] = None
                notes['cleared'] = pending['field']
            else:
                self.state['queue'].append(pending)         # keep it, ask again later

        changed = CHANGE_INTENTS.get(intent)                # 2. change requests
        if changed and analysis['intent_confidence'] >= CONF_MEDIUM:
            self.state[changed] = None
            if changed == 'quantity':
                self.state['quantity_kg'] = None
                self.state['quantity_value'] = None
            notes['cleared'] = changed

        for field in ('crop', 'date', 'unit', 'slot_hint'):  # 3. entity merge (req. 8)
            if field not in entities:
                continue
            score = confidence.get(field, 0)
            if score >= CONF_HIGH:
                self._accept(field, entities[field], score)
                notes['accepted'].append(field)
            elif score >= CONF_MEDIUM:
                self._accept(field, entities[field], score)
                self._hold(field, entities[field], score)
                notes['confirm'].append(field)
            else:
                notes['clarify'].append(field)
        if 'quantity_kg' in entities or 'quantity_value' in entities:
            field = 'quantity_kg' if 'quantity_kg' in entities else 'quantity_value'
            score = confidence.get('quantity', 0)
            if score >= CONF_HIGH:
                self._accept(field, entities[field], score)
                if entities.get('unit'):
                    self.state['unit'] = entities['unit']
                notes['accepted'].append('quantity')
            elif score >= CONF_MEDIUM:
                self._accept(field, entities[field], score)
                self._hold(field, entities[field], score)
                notes['confirm'].append('quantity')
            else:
                notes['clarify'].append('quantity')
        if 'token' in entities:
            self.state['token'] = entities['token']

        base = {'intent': intent, 'intent_confidence': analysis['intent_confidence'],
                'entities': entities, 'confidence': confidence,
                'corrections': analysis['corrections'], 'ambiguous': analysis['ambiguous'],
                'state': self.snapshot(), 'analysis': analysis, 'notes': notes,
                'slots_needed': False,
                'understood': bool(entities) or intent != 'unknown'}

        if not base['understood'] and self.missing() and (
                self.state.get('crop') or self.state.get('date') or self.state.get('quantity_kg')):
            # something is already known but this turn was unintelligible: ask
            # the farmer to repeat instead of pretending the sentence was valid
            base['next'] = {'action': 'repeat', 'field': expected,
                            'question': 'nlp.repeat', 'params': {}}
            return base

        if intent in REPLY_INTENTS and analysis['intent_confidence'] >= CONF_MEDIUM:
            base['next'] = {'action': 'reply', 'intent': intent, 'field': None,
                            'question': None, 'params': self.booking_params()}
            return base
        unclear = [item for item in analysis['ambiguous']
                   if item.get('field') in ('unit', 'date', 'crop')]
        if unclear and not self.state.get('pending'):
            item = unclear[0]
            base['next'] = {'action': 'clarify', 'field': item['field'],
                            'question': 'nlp.clarify.' + item['field'],
                            'params': {'seen': item.get('seen', ''),
                                       'options': item.get('options', [])}}
            return base
        if intent in ('confirm_booking', 'deny') and not pending:
            base['next'] = self._ask(self.expected() or 'crop')   # nothing to confirm yet
            return base
        if self.state.get('pending'):
            held = self.state['pending']
            base['next'] = {'action': 'confirm', 'field': held['field'],
                            'question': CONFIRM_KEYS.get(held['field'], 'nlp.confirm.quantity'),
                            'params': self._confirm_params(held['field'])}
            return base
        missing = self.missing()
        if missing:
            base['next'] = self._ask(missing[0], change=changed)
            return base
        base['next'] = {'action': 'ready', 'field': None, 'question': 'nlp.checkingSlots',
                        'params': self.booking_params()}
        base['slots_needed'] = True
        return base


def suggest(prefix, expected=None, state=None, limit=8, today=None):
    """Context-aware word/phrase prediction for typed text (requirements 1, 12).

    Completions come from the lexicon (in the language being typed), from the
    question currently being answered (`expected`) and from what is already known
    (crop / quantity / date). Fragments shorter than MIN_FRAGMENT, or fragments
    with several equally likely meanings, produce NO suggestion - the assistant
    asks instead of guessing.
    """
    raw = (prefix or '').strip().lower()
    if not raw:
        return []
    state = state or {}
    today = today or date.today()
    parts = raw.split(' ')
    last = parts[-1]
    head = ' '.join(parts[:-1])
    number = None
    for token in parts:
        if re.fullmatch(r'\d+(\.\d+)?', token):
            number = token
    out, seen = [], set()

    def add(text, field, confidence):
        if text and text not in seen and len(out) < limit:
            seen.add(text)
            out.append({'text': text, 'field': field, 'confidence': round(confidence, 2)})

    # 1. finish the word being typed, from the lexicons of all three languages
    if len(last) >= MIN_FRAGMENT:
        if expected in (None, 'crop'):
            for word in _CROP_LOOKUP:
                if word.startswith(last):
                    add((head + ' ' + word).strip(), 'crop',
                        0.85 if len(last) >= 3 else 0.6)
        if expected in (None, 'unit', 'quantity', 'date') or number:
            for word in _UNIT_LOOKUP:
                if word.startswith(last):
                    add((head + ' ' + word).strip(), 'unit', 0.8)
        if expected in (None, 'date'):
            for name in MONTHS:
                if len(name) >= 3 and name.startswith(last):
                    add((head + ' ' + name).strip(), 'date', 0.7)
    # 2. a unit fragment straight after a number: "500 k..." -> "500 kg"
    if number and not re.fullmatch(r'\d+(\.\d+)?', last):
        for word, confidence in (('kg', 0.75), ('kilos', 0.7), ('kilograms', 0.65),
                                 ('ton', 0.6), ('quintal', 0.55)):
            if word.startswith(last):
                add((head + ' ' + word).strip(), 'unit', confidence)
    # 3. phrase completion from the conversation context (requirement 12)
    crop = state.get('crop')
    digits = []
    if number:
        digits = [number]
        if re.fullmatch(r'\d{1,2}(\.\d+)?', last):     # "5" -> "50" -> "500"
            digits += [number + '0', number + '00']
    for value in digits:
        if crop:
            add('%s %s kg' % (crop_word(crop), value), 'quantity', 0.7)
            add('%s %s kilos' % (crop_word(crop), value), 'quantity', 0.6)
        else:
            add('%s kg' % value, 'unit', 0.7)
            add('%s kilos' % value, 'unit', 0.65)
    if crop and number:
        known = state.get('date')
        # finish the whole sentence, using the most likely completion of the
        # number the farmer is still typing ("rice 5..." -> "... 500 kg on ...")
        biggest = digits[-1] if digits else number
        add('%s %s kg on %s' % (crop_word(crop), biggest,
                                human_date(known or today.isoformat())),
            'date', 0.75 if known else 0.5)
    # 4. no prefix match but a very close miss: offer the corrected word
    if not out and len(last) >= 3:
        for table, field in ((_CROP_LOOKUP, 'crop'), (_UNIT_LOOKUP, 'unit')):
            hit = match_token(last, table, expected == field)
            if hit and hit['confidence'] >= 0.7:
                word = crop_word(hit['value']) if field == 'crop' else hit['value']
                add((head + ' ' + word).strip(), field, hit['confidence'])
    return out[:limit]


def lexicon():
    """Vocabulary + thresholds served to the browser predictor (requirement 12)."""
    return {'version': 1, 'crops': CROP_VARIANTS, 'units': UNIT_VARIANTS, 'unit_kg': UNIT_KG,
            'months': dict(MONTHS), 'intents': sorted(INTENT_PATTERNS),
            'thresholds': {'high': CONF_HIGH, 'medium': CONF_MEDIUM,
                           'min_fragment': MIN_FRAGMENT}}


# ===========================================================================
# Structured Conversational Booking Assistant (Conversation States & NLP)
# ===========================================================================

WAITING_FOR_DATE = 'WAITING_FOR_DATE'
WAITING_FOR_CROP = 'WAITING_FOR_CROP'
WAITING_FOR_QUANTITY = 'WAITING_FOR_QUANTITY'
CONFIRM_KEEP_DETAILS = 'CONFIRM_KEEP_DETAILS'
WAITING_FOR_TIME = 'WAITING_FOR_TIME'
CONFIRMING_TIME_AM_PM = 'CONFIRMING_TIME_AM_PM'
CHECKING_SLOTS = 'CHECKING_SLOTS'
WAITING_FOR_SLOT_SELECTION = 'WAITING_FOR_SLOT_SELECTION'
WAITING_FOR_SLOT_CONFIRMATION = 'WAITING_FOR_SLOT_CONFIRMATION'
WAITING_FOR_CONFIRMATION = 'WAITING_FOR_CONFIRMATION'
BOOKING = 'BOOKING'
COMPLETED = 'COMPLETED'

AFFIRMATIVE_WORDS = {
    'yes', 'yeah', 'yup', 'yep', 'ok', 'okay', 'sure', 'confirm', 'book it',
    'book', 'done', 'fine', 'book 9', 'book 10', 'book 11', 'book 2', 'book this',
    'అవును', 'సరే', 'హా', 'బుక్ చేయండి', 'బుక్ చెయ్యి', 'ఖచ్చితంగా',
    'हाँ', 'जी', 'ठीक', 'सही', 'बुक करो', 'बुक करें', 'पुष्टि करें'
}

NEGATIVE_WORDS = {
    'no', 'nope', 'not now', 'cancel', 'dont', "don't", 'stop', 'never',
    'కాదు', 'వద్దు', 'రద్దు', 'ఆపు',
    'नहीं', 'ना', 'मत करो', 'रद्द', 'रद्द करो', 'रोकें'
}


def is_affirmative(text):
    t = (text or '').strip().lower()
    if t in AFFIRMATIVE_WORDS:
        return True
    for w in AFFIRMATIVE_WORDS:
        if t.startswith(w + ' ') or t.endswith(' ' + w):
            return True
    return bool(re.search(r'\b(yes|yeah|yup|ok|okay|sure|confirm|book it|అవును|సరే|హా|हाँ|ठीक)\b', t, re.I))


def is_negative(text):
    t = (text or '').strip().lower()
    if t in NEGATIVE_WORDS:
        return True
    for w in NEGATIVE_WORDS:
        if t.startswith(w + ' ') or t.endswith(' ' + w):
            return True
    return bool(re.search(r'\b(no|nope|cancel|dont|కాదు|వద్దు|రద్దు|नहीं|ना)\b', t, re.I))


def format_slot_time_label(time_str):
    """Format slot start_time '09:00 AM' -> '9 AM', '02:00 PM' -> '2 PM'."""
    raw = (time_str or '').strip()
    m = re.match(r'^0?(\d{1,2}):(\d{2})\s*(AM|PM)$', raw, re.I)
    if m:
        h, mn, p = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        return f"{h}:{mn:02d} {p}" if mn else f"{h} {p}"
    return raw


def format_slot_list_spoken(slots, language='en'):
    """Format available slot times into natural spoken list: '10 AM, 11 AM, and 2 PM'."""
    labels = []
    for s in slots:
        t = s.get('start_time') if isinstance(s, dict) else str(s)
        lbl = format_slot_time_label(t)
        if lbl not in labels:
            labels.append(lbl)
    if not labels:
        return ''
    if len(labels) == 1:
        return labels[0]
    joiner = {
        'te': ' మరియు ',
        'hi': ' और '
    }.get(language, ', and ')
    if len(labels) == 2:
        conj = {'te': ' మరియు ', 'hi': ' और '}.get(language, ' and ')
        return labels[0] + conj + labels[1]
    return ', '.join(labels[:-1]) + joiner + labels[-1]


def parse_time_preference(normalized_text, tokens, expected=None):
    """Parse farmer's spoken time preference.
    Handles '9 AM', '9:00 AM', '2 PM', 'morning', 'afternoon', bare '9', 'first slot', etc.
    """
    t = normalized_text or ''
    # 1. Regex match for explicit AM/PM or language time markers
    m = TIME_RE.search(t)
    if m:
        raw_hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        marker = (m.group(3) or '').lower()
        if marker == 'pm' and raw_hour < 12:
            hour_24 = raw_hour + 12
            period = 'PM'
        elif marker == 'am' and raw_hour == 12:
            hour_24 = 0
            period = 'AM'
        elif marker == 'pm':
            hour_24 = raw_hour
            period = 'PM'
        elif marker == 'am':
            hour_24 = raw_hour
            period = 'AM'
        else:
            # Language markers like గంటలకు / बजे
            if 7 <= raw_hour <= 11:
                period = 'AM'
                hour_24 = raw_hour
            elif 1 <= raw_hour <= 6:
                period = 'PM'
                hour_24 = raw_hour + 12
            else:
                period = 'AM'
                hour_24 = raw_hour
        disp_h = raw_hour if (1 <= raw_hour <= 12) else (raw_hour - 12 if raw_hour > 12 else 12)
        lbl = f"{disp_h}:{minute:02d} {period}" if minute else f"{disp_h} {period}"
        std = f"{disp_h:02d}:{minute:02d} {period}"
        return {'hour': disp_h, 'hour_24': hour_24, 'minute': minute, 'period': period,
                'label': lbl, 'standard_time': std, 'is_bare_hour': False, 'raw': m.group(0)}

    # 2. Match phrases with 'morning' / 'afternoon' + hour ("9 morning", "morning 9", "2 afternoon")
    morn_match = re.search(r'(\d{1,2})\s*(?:in the\s+)?(morning|afternoon|evening|ఉదయం|మధ్యాహ్నం|సాయంత్రం|सुबह|दोपहर|शाम)', t)
    if not morn_match:
        morn_match = re.search(r'(morning|afternoon|evening|ఉదయం|మధ్యాహ్నం|సాయంత్రం|सुबह|दोपहर|शाम)\s*(?:at\s+)?(\d{1,2})', t)
    if morn_match:
        g1, g2 = morn_match.groups()
        num_str = g1 if g1.isdigit() else g2
        part_str = g2 if g1.isdigit() else g1
        raw_hour = int(num_str)
        part = SLOT_PARTS.get(part_str, 'morning')
        if part == 'morning':
            period = 'AM'
            hour_24 = raw_hour if raw_hour < 12 else 0
        else:
            period = 'PM'
            hour_24 = raw_hour + 12 if raw_hour < 12 else 12
        disp_h = raw_hour if (1 <= raw_hour <= 12) else (raw_hour - 12 if raw_hour > 12 else 12)
        lbl = f"{disp_h} {period}"
        std = f"{disp_h:02d}:00 {period}"
        return {'hour': disp_h, 'hour_24': hour_24, 'minute': 0, 'period': period,
                'label': lbl, 'standard_time': std, 'is_bare_hour': False, 'part': part, 'raw': morn_match.group(0)}

    # 3. Phrased booking: "book 10", "want 11", "book 9"
    phrase_match = re.search(r'\b(?:book|want|slot|at)\s+(\d{1,2})\s*(am|pm)?\b', t)
    if phrase_match:
        h = int(phrase_match.group(1))
        p = phrase_match.group(2)
        if p:
            period = p.upper()
            hour_24 = h + (12 if period == 'PM' and h < 12 else 0)
            return {'hour': h, 'hour_24': hour_24, 'minute': 0, 'period': period,
                    'label': f"{h} {period}", 'standard_time': f"{h:02d}:00 {period}",
                    'is_bare_hour': False, 'raw': phrase_match.group(0)}
        else:
            period = 'AM' if 7 <= h <= 11 else 'PM'
            hour_24 = h + (12 if period == 'PM' and h < 12 else 0)
            return {'hour': h, 'hour_24': hour_24, 'minute': 0, 'period': period,
                    'label': f"{h} {period}", 'standard_time': f"{h:02d}:00 {period}",
                    'is_bare_hour': True, 'raw': phrase_match.group(0)}

    # 4. Bare hour if expected == 'time' or string is just digits
    if expected in ('time', 'slot') or re.fullmatch(r'\d{1,2}', t.strip()):
        num_m = re.search(r'\b(\d{1,2})\b', t)
        if num_m:
            h = int(num_m.group(1))
            if 1 <= h <= 12:
                period = 'AM' if 7 <= h <= 11 else 'PM'
                hour_24 = h + (12 if period == 'PM' and h < 12 else 0)
                return {'hour': h, 'hour_24': hour_24, 'minute': 0, 'period': period,
                        'label': f"{h} {period}", 'standard_time': f"{h:02d}:00 {period}",
                        'is_bare_hour': True, 'raw': num_m.group(0)}

    # 5. Ordinal preference: 'first', 'second', 'last'
    for tok in tokens:
        if tok in SLOT_ORDINALS:
            ord_val = SLOT_ORDINALS[tok]
            return {'ordinal': ord_val, 'is_bare_hour': False, 'label': tok, 'raw': tok}
    if re.search(r'\b(first|1st|మొదటి|पहला)\b', t):
        return {'ordinal': 1, 'is_bare_hour': False, 'label': 'first slot', 'raw': 'first'}
    if re.search(r'\b(second|2nd|రెండో|दूसरा)\b', t):
        return {'ordinal': 2, 'is_bare_hour': False, 'label': 'second slot', 'raw': 'second'}
    if re.search(r'\b(last|చివరి|आखिरी)\b', t):
        return {'ordinal': -1, 'is_bare_hour': False, 'label': 'last slot', 'raw': 'last'}

    # 6. Part of day preference: 'morning', 'afternoon', 'evening'
    for tok in tokens:
        if tok in SLOT_PARTS:
            return {'part': SLOT_PARTS[tok], 'is_bare_hour': False, 'label': tok, 'raw': tok}

    return None


def match_slot_to_preference(pref, available_slots):
    """Match a parsed time preference to an available backend slot."""
    if not pref or not available_slots:
        return None
    # 1. Match by ordinal
    if 'ordinal' in pref:
        ord_val = pref['ordinal']
        if ord_val == 1 and len(available_slots) >= 1:
            return available_slots[0]
        if ord_val == 2 and len(available_slots) >= 2:
            return available_slots[1]
        if ord_val == -1 and available_slots:
            return available_slots[-1]

    # 2. Match by hour and period
    h = pref.get('hour')
    period = pref.get('period')
    if h is not None:
        for s in available_slots:
            st = s.get('start_time', '')
            m = re.match(r'^0?(\d{1,2}):(\d{2})\s*(AM|PM)$', st, re.I)
            if m:
                sh, sm, sp = int(m.group(1)), int(m.group(2)), m.group(3).upper()
                if period:
                    if sh == h and sp == period:
                        return s
                else:
                    if sh == h:
                        return s

    # 3. Match by part of day (morning / afternoon)
    part = pref.get('part')
    if part == 'morning':
        for s in available_slots:
            st = s.get('start_time', '')
            if 'AM' in st.upper():
                return s
    elif part in ('afternoon', 'evening'):
        for s in available_slots:
            st = s.get('start_time', '')
            if 'PM' in st.upper():
                return s

    return None


def match_slot_choice(text, available_slots):
    """Match farmer's natural slot selection response to an actual available slot."""
    norm = normalize(text)
    pref = parse_time_preference(norm['text'], norm['tokens'], expected='slot')
    return match_slot_to_preference(pref, available_slots)


ASSISTANT_STRINGS = {
    'ask_date': {
        'en': "Which date would you like to book?",
        'te': "మీరు ఏ తేదీన బుకింగ్ చేసుకోవాలనుకుంటున్నారు?",
        'hi': "आप किस तारीख के लिए बुकिंग करना चाहते हैं?"
    },
    'book_again_keep': {
        'en': "Your previous booking was for {crop}, {quantity} kilograms. Would you like to keep these details?",
        'te': "మీ మునుపటి బుకింగ్ {crop}, {quantity} కిలోలకు ఉంది. ఈ వివరాలను ఉంచుకుంటారా?",
        'hi': "आपकी पिछली बुकिंग {crop}, {quantity} किलोग्राम की थी। क्या आप ये विवरण रखना चाहेंगे?"
    },
    'book_again_date_passed': {
        'en': "The previous booking date has passed. Which date would you like to book?",
        'te': "మునుపటి బుకింగ్ తేదీ ముగిసింది. మీరు ఏ తేదీన బుక్ చేయాలనుకుంటున్నారు?",
        'hi': "पिछली बुकिंग की तारीख निकल चुकी है। आप किस तारीख को बुक करना चाहेंगे?"
    },
    'book_again_no_previous': {
        'en': "I could not find a cancelled booking to reuse. Which date would you like to book?",
        'te': "పునఃఉపయోగించడానికి రద్దైన బుకింగ్ కనబడలేదు. మీరు ఏ తేదీన బుక్ చేయాలనుకుంటున్నారు?",
        'hi': "पुनः उपयोग के लिए रद्द बुकिंग नहीं मिली। आप किस तारीख को बुक करना चाहेंगे?"
    },
    'clarify_weekday': {
        'en': "Which {weekday} do you mean? Please tell me the date.",
        'te': "ఏ {weekday} మీ ఉద్దేశం? దయచేసి ఖచ్చితమైన తేదీ చెప్పండి.",
        'hi': "आपका मतलब कौन से {weekday} से है? कृपया तारीख बताएं।"
    },
    'invalid_date': {
        'en': "Sorry, I couldn't understand that. Please say the date again.",
        'te': "క్షమించండి, తేదీ అర్థం కాలేదు. దయచేసి మళ్లీ తేదీ చెప్పండి.",
        'hi': "क्षमा करें, तारीख समझ नहीं आई। कृपया दोबारा तारीख बताएं।"
    },
    'ask_crop': {
        'en': "Which crop would you like to book?",
        'te': "మీరు ఏ పంట బుక్ చేయాలనుకుంటున్నారు?",
        'hi': "आप कौन सी फसल बुक करना चाहते हैं?"
    },
    'invalid_crop': {
        'en': "I did not recognise that crop. Please choose Rice, Wheat, Maize, or Cotton.",
        'te': "ఆ పంట అర్థం కాలేదు. దయచేసి వరి, గోధుమ, మొక్కజొన్న లేదా పత్తి చెప్పండి.",
        'hi': "वह फसल समझ नहीं आई। कृपया चावल, गेहूं, मक्का या कपास कहें।"
    },
    'ask_quantity': {
        'en': "How many kilograms of {crop} do you want to book?",
        'te': "మీరు ఎన్ని కిలోల {crop} బుక్ చేయాలనుకుంటున్నారు?",
        'hi': "आप कितने किलोग्राम {crop} बुक करना चाहते हैं?"
    },
    'invalid_quantity': {
        'en': "Please specify a valid quantity in kilograms.",
        'te': "దయచేసి సరైన పరిమాణాన్ని కిలోలలో చెప్పండి.",
        'hi': "कृपया किलोग्राम में सही मात्रा बताएं।"
    },
    'ask_time': {
        'en': "What time would you prefer?",
        'te': "మీరు ఏ సమయాన్ని ఇష్టపడతారు?",
        'hi': "आप कौन सा समय पसंद करेंगे?"
    },
    'confirm_bare_time': {
        'en': "Do you mean {time}?",
        'te': "మీ ఉద్దేశం {time} నా?",
        'hi': "क्या आपका मतलब {time} है?"
    },
    'multi_prefix': {
        'en': "I have {date}, {crop}, and {quantity}. What time would you prefer?",
        'te': "నా దగ్గర {date}, {crop}, మరియు {quantity} ఉన్నాయి. మీరు ఏ సమయాన్ని ఇష్టపడతారు?",
        'hi': "मेरे पास {date}, {crop}, और {quantity} है। आप कौन सा समय पसंद करेंगे?"
    },
    'updated_quantity': {
        'en': "Updated quantity to {quantity}. What time would you prefer?",
        'te': "పరిమాణాన్ని {quantity} కి నవీకరించాను. మీరు ఏ సమయాన్ని ఇష్టపడతారు?",
        'hi': "मात्रा को {quantity} में अपडेट किया गया। आप कौन सा समय पसंद करेंगे?"
    },
    'updated_time': {
        'en': "Updated preferred time to {time}.",
        'te': "సమయాన్ని {time} కి నవీకరించాను.",
        'hi': "पसंदीदा समय को {time} में अपडेट किया गया।"
    },
    'slot_available_confirm': {
        'en': "{time} is available. Would you like me to book it?",
        'te': "{time} అందుబాటులో ఉంది. నేను దానిని బుక్ చేయనా?",
        'hi': "{time} उपलब्ध है। क्या मैं इसे बुक कर दूँ?"
    },
    'slot_unavailable_choices': {
        'en': "{time} is not available. The available times are {slots}. Which time would you like?",
        'te': "{time} అందుబాటులో లేదు. అందుబాటులో ఉన్న సమయాలు: {slots}. మీకు ఏ సమయం కావాలి?",
        'hi': "{time} उपलब्ध नहीं है। उपलब्ध समय हैं: {slots}। आप कौन सा समय चाहेंगे?"
    },
    'choose_available_slot': {
        'en': "Please choose one of the available slots: {slots}.",
        'te': "దయచేసి అందుబాటులో ఉన్న స్లాట్‌లలో ఒకదాన్ని ఎంచుకోండి: {slots}.",
        'hi': "कृपया उपलब्ध स्लॉट में से एक चुनें: {slots}।"
    },
    'no_slots_date': {
        'en': "There are no available slots for this date. Would you like to choose another date?",
        'te': "ఈ తేదీకి స్లాట్లు అందుబాటులో లేవు. మీరు మరొక తేదీని ఎంచుకోవాలనుకుంటున్నారా?",
        'hi': "इस तारीख के लिए कोई स्लॉट उपलब्ध नहीं है। क्या आप कोई अन्य तारीख चुनना चाहेंगे?"
    },
    'final_summary': {
        'en': "Please confirm your booking.\n\nDate: {date}\nCrop: {crop}\nQuantity: {quantity}\nTime: {time}\n\nWould you like me to confirm this booking?",
        'te': "దయచేసి మీ బుకింగ్‌ను నిర్ధారించండి.\n\nతేదీ: {date}\nపంట: {crop}\nపరిమాణం: {quantity}\nసమయం: {time}\n\nనేను ఈ బుకింగ్‌ను నిర్ధారించాలా?",
        'hi': "कृपया अपनी बुकिंग की पुष्टि करें।\n\nतारीख: {date}\nफसल: {crop}\nमात्रा: {quantity}\nसमय: {time}\n\nक्या मैं इस बुकिंग की पुष्टि करूँ?"
    },
    'booking_success_token': {
        'en': "Your booking has been confirmed.\n\nDate: {date}\nCrop: {crop}\nQuantity: {quantity}\nTime: {time}\nToken number: {token}\n\nPlease arrive at the procurement center at the scheduled time.",
        'te': "మీ బుకింగ్ నిర్ధారించబడింది.\n\nతేదీ: {date}\nపంట: {crop}\nపరిమాణం: {quantity}\nసమయం: {time}\nటోకెన్ నంబర్: {token}\n\nదయచేసి నిర్ణీత సమయానికి కొనుగోలు కేంద్రానికి చేరుకోండి.",
        'hi': "आपकी बुकिंग की पुष्टि हो गई है।\n\nतारीख: {date}\nफसल: {crop}\nमात्रा: {quantity}\nसमय: {time}\nटोकन नंबर: {token}\n\nकृपया निर्धारित समय पर खरीद केंद्र पर पहुँचें।"
    },
    'booking_cancelled': {
        'en': "Booking request was cancelled.",
        'te': "బుకింగ్ అభ్యర్థన రద్దు చేయబడింది.",
        'hi': "बुकिंग अनुरोध रद्द कर दिया गया।"
    },
    'booking_cancelled_prompt': {
        'en': "Booking request was cancelled. Would you like to choose another time or date?",
        'te': "బుకింగ్ అభ్యర్థన రద్దు చేయబడింది. మీరు వేరే సమయం లేదా తేదీ ఎంచుకోవాలనుకుంటున్నారా?",
        'hi': "बुकिंग अनुरोध रद्द कर दिया गया। क्या आप कोई अन्य समय या तारीख चुनना चाहेंगे?"
    },
    'booking_failed': {
        'en': "Booking failed: {error}. Please choose another slot.",
        'te': "బుకింగ్ విఫలమైంది: {error}. దయచేసి మరొక స్లాట్ ఎంచుకోండి.",
        'hi': "बुकिंग विफल रही: {error}। कृपया दूसरा स्लॉट चुनें।"
    },
    'booking_in_progress': {
        'en': "Creating your booking now...",
        'te': "మీ బుకింగ్ సృష్టిస్తోంది...",
        'hi': "आपकी बुकिंग बनाई जा रही है..."
    },
    'repeat': {
        'en': "Sorry, I couldn't understand that. Please say it again.",
        'te': "క్షమించండి, నాకు అర్థం కాలేదు. దయచేసి మళ్ళీ చెప్పండి.",
        'hi': "क्षमा करें, मुझे समझ नहीं आया। कृपया दोबारा कहें।"
    }
}


class BookingAssistant(object):
    """Structured Conversational Booking Assistant.
    Follows a strictly controlled sequence:
    1. Date -> 2. Crop -> 3. Quantity in KG -> 4. Preferred time
    -> 5. Available time slots -> 6. Farmer selects a slot
    -> 7. Confirmation -> 8. Booking creation -> 9. Token details.
    """

    def __init__(self, state=None, today=None, language='en'):
        self.today = today or date.today()
        self.language = language or 'en'
        self.state = {
            'conv_state': WAITING_FOR_DATE,
            'date': None,
            'crop': None,
            'quantity_kg': None,
            'preferred_time': None,
            'preferred_time_label': None,
            'pending_hour': None,
            'pending_time_pref': None,
            'selected_slot': None,
            'available_slots': [],
            'booking_token': None,
        }
        if state:
            self.state.update(state)

    def _msg(self, key, params=None):
        params = params or {}
        entry = ASSISTANT_STRINGS.get(key, {})
        text = entry.get(self.language) or entry.get('en', '')
        for k, v in params.items():
            text = text.replace('{' + k + '}', str(v))
        return text

    def _keep_msg(self):
        """The 'keep previous details?' question, built from the seeded state."""
        return self._msg('book_again_keep', {
            'crop': self.state.get('crop') or '',
            'quantity': self.state.get('quantity_kg') if self.state.get('quantity_kg') is not None else ''
        })

    def _response(self, message, next_state, action='none', action_params=None):
        self.state['conv_state'] = next_state
        return {
            'success': True,
            'message': message,
            'conv_state': next_state,
            'action': action,
            'action_params': action_params or self.booking_params(),
            'state': dict(self.state)
        }

    def booking_params(self):
        kg = self.state.get('quantity_kg')
        tons = round((kg or 0) / 1000.0, 3) if kg else None
        slot = self.state.get('selected_slot')
        return {
            'date': self.state.get('date'),
            'crop': self.state.get('crop'),
            'quantity_kg': kg,
            'quantity_tons': tons,
            'time': self.state.get('preferred_time_label') or (slot.get('start_time') if slot else None),
            'selected_slot': slot
        }

    def start(self):
        """Initial turn when assistant opens."""
        self.state['conv_state'] = WAITING_FOR_DATE
        return self._response(self._msg('ask_date'), WAITING_FOR_DATE)

    def start_book_again(self, previous=None):
        """Start a NEW booking conversation seeded from a previous booking.

        The previous booking itself is never touched: this only reuses its
        crop/quantity as defaults. The date is deliberately NOT copied into
        state['date'] - the old date must be re-stated by the farmer and then
        re-checked against the real slot table before anything is booked.
        """
        prev = previous or {}
        crop = (prev.get('crop') or '').strip()
        quantity_kg = prev.get('quantity_kg')
        previous_date = prev.get('date')
        if not crop and not quantity_kg:
            # Nothing worth reusing: fall back to the normal booking flow.
            self.state['conv_state'] = WAITING_FOR_DATE
            return self._response(self._msg('book_again_no_previous'), WAITING_FOR_DATE)
        self.state['crop'] = crop or None
        self.state['quantity_kg'] = quantity_kg or None
        self.state['previous_date'] = previous_date
        self.state['date'] = None
        self.state['selected_slot'] = None
        self.state['preferred_time'] = None
        self.state['preferred_time_label'] = None
        self.state['pending_time_pref'] = None
        msg = self._msg('book_again_keep', {'crop': crop, 'quantity': quantity_kg})
        return self._response(msg, CONFIRM_KEEP_DETAILS)

    def check_slots_with_backend(self, available_slots):
        """Evaluate preferred time against available slots from backend."""
        self.state['available_slots'] = available_slots or []
        if not available_slots:
            return self._response(self._msg('no_slots_date'), WAITING_FOR_DATE)

        # Match preferred time against available slots
        pref = self.state.get('pending_time_pref') or {
            'hour': None, 'period': None,
            'standard_time': self.state.get('preferred_time'),
            'label': self.state.get('preferred_time_label')
        }
        if not pref.get('hour') and self.state.get('preferred_time_label'):
            norm_lbl = normalize(self.state['preferred_time_label'])
            parsed_pref = parse_time_preference(norm_lbl['text'], norm_lbl['tokens'], expected='time')
            if parsed_pref:
                pref = parsed_pref

        matched = match_slot_to_preference(pref, available_slots)
        if matched:
            self.state['selected_slot'] = matched
            time_lbl = format_slot_time_label(matched.get('start_time'))
            self.state['preferred_time_label'] = time_lbl
            self.state['conv_state'] = WAITING_FOR_SLOT_CONFIRMATION
            msg = self._msg('slot_available_confirm', {'time': time_lbl})
            return self._response(msg, WAITING_FOR_SLOT_CONFIRMATION)
        else:
            req_label = self.state.get('preferred_time_label') or 'The requested time'
            slot_labels = format_slot_list_spoken(available_slots, self.language)
            self.state['conv_state'] = WAITING_FOR_SLOT_SELECTION
            msg = self._msg('slot_unavailable_choices', {'time': req_label, 'slots': slot_labels})
            return self._response(msg, WAITING_FOR_SLOT_SELECTION)

    def confirm_booking_success(self, booking_info):
        """Handle successful booking creation from backend."""
        self.state['conv_state'] = COMPLETED
        token = booking_info.get('token')
        self.state['booking_token'] = token
        slot = self.state.get('selected_slot') or {}
        time_str = self.state.get('preferred_time_label') or format_slot_time_label(slot.get('start_time', ''))
        msg = self._msg('booking_success_token', {
            'date': human_date(self.state['date']),
            'crop': self.state['crop'],
            'quantity': f"{self.state['quantity_kg']} kg",
            'time': time_str,
            'token': token
        })
        return self._response(msg, COMPLETED, action='booking_complete', action_params=booking_info)

    def process_turn(self, text, available_slots=None):
        """Process one conversational turn according to the structured state machine."""
        t_raw = (text or '').strip()
        if not t_raw:
            curr = self.state.get('conv_state', WAITING_FOR_DATE)
            if curr == CONFIRM_KEEP_DETAILS:
                return self._response(self._keep_msg(), curr)
            repeat_key = {
                WAITING_FOR_DATE: 'ask_date',
                WAITING_FOR_CROP: 'ask_crop',
                WAITING_FOR_QUANTITY: 'ask_quantity',
                WAITING_FOR_TIME: 'ask_time',
                WAITING_FOR_CONFIRMATION: 'final_summary'
            }.get(curr, 'repeat')
            return self._response(self._msg(repeat_key, {'crop': self.state.get('crop', '')}), curr)

        normalized = normalize(t_raw)
        tokens = normalized['tokens']
        raw_text = normalized['text']

        curr_state = self.state.get('conv_state', WAITING_FOR_DATE)

        # Check for corrections across turns
        # Correction 1: Time change ("I want 10 AM... actually 11 AM" / "actually 11 AM")
        if ('actually' in raw_text or 'change time' in raw_text) and any(w in raw_text for w in ('am', 'pm', 'clock', 'morning', 'afternoon', 'గంట', 'बजे')):
            tp = parse_time_preference(raw_text, tokens, expected='time')
            if tp and not tp.get('is_bare_hour'):
                self.state['preferred_time'] = tp.get('standard_time')
                self.state['preferred_time_label'] = tp.get('label')
                self.state['pending_time_pref'] = tp
                self.state['conv_state'] = CHECKING_SLOTS
                if available_slots:
                    return self.check_slots_with_backend(available_slots)
                return self._response(self._msg('updated_time', {'time': self.state['preferred_time_label']}),
                                      CHECKING_SLOTS, action='check_slots')

        # Correction 2: Quantity change ("Actually, make that 700 kg" / "change quantity to 700")
        if any(w in raw_text for w in ('actually', 'make that', 'change quantity', 'మార్చు', 'బదల్', 'बदल')):
            q_val, q_unit, q_c, _ = parse_quantity(tokens, expected=True)
            if q_val and q_val > 0:
                self.state['quantity_kg'] = clean_number(q_val)
                self.state['selected_slot'] = None
                self.state['conv_state'] = WAITING_FOR_TIME
                msg = self._msg('updated_quantity', {'quantity': f"{self.state['quantity_kg']} kg"})
                return self._response(msg, WAITING_FOR_TIME)

        # Multi-entity extraction helper: inspect turn for any date, crop, quantity, time
        extracted_date, date_conf, date_note = parse_date(raw_text, tokens, self.today)
        extracted_crop = None
        for tok in tokens:
            m_crop = match_token(tok, _CROP_LOOKUP, expected=False)
            if m_crop and m_crop['confidence'] >= CONF_MEDIUM:
                extracted_crop = m_crop['value']
                break
            if tok in _CROP_STT_LOOKUP:
                extracted_crop = _CROP_STT_LOOKUP[tok]
                break

        day_of_month = None
        if extracted_date:
            try:
                day_of_month = date.fromisoformat(extracted_date).day
            except (ValueError, TypeError):
                day_of_month = None

        q_expected = (curr_state == WAITING_FOR_QUANTITY)
        extracted_qty, extracted_unit, qty_conf, _ = parse_quantity(tokens, expected=q_expected)
        # If the number is bare (no unit) and we weren't expecting quantity, don't treat it as quantity
        if extracted_unit is None and not q_expected:
            extracted_qty = None
        elif day_of_month is not None and extracted_qty == day_of_month and extracted_unit is None:
            extracted_qty = None

        extracted_time = parse_time_preference(raw_text, tokens, expected='time')

        # -------------------------------------------------------------------
        # STATE: CONFIRM_KEEP_DETAILS (Book Again: keep the old crop/quantity?)
        # -------------------------------------------------------------------
        if curr_state == CONFIRM_KEEP_DETAILS:
            declining = is_negative(raw_text) and not is_affirmative(raw_text)
            affirmative = is_affirmative(raw_text)
            new_info = bool(extracted_date or extracted_crop or (extracted_qty and extracted_qty > 0))
            if not (declining or affirmative or new_info):
                # Ambiguous answer: repeat the keep-details question.
                return self._response(self._keep_msg(), CONFIRM_KEEP_DETAILS)

            if declining:
                # The farmer is changing details: the old crop/quantity are
                # dropped unless they were restated in this very turn.
                self.state['crop'] = None
                self.state['quantity_kg'] = None
            if extracted_crop:
                self.state['crop'] = extracted_crop
            if extracted_qty and extracted_qty > 0:
                self.state['quantity_kg'] = clean_number(extracted_qty)
            if extracted_date:
                self.state['date'] = extracted_date
            elif affirmative and not declining:
                prev = self.state.get('previous_date')
                # Reuse the previous date only while it is still in the future;
                # availability is re-verified live before anything is booked.
                if prev and prev >= self.today.isoformat():
                    self.state['date'] = prev

            if not self.state.get('date'):
                prev = self.state.get('previous_date')
                if prev and prev < self.today.isoformat():
                    date_prompt = 'book_again_date_passed'
                else:
                    date_prompt = 'ask_date'
                self.state['conv_state'] = WAITING_FOR_DATE
                return self._response(self._msg(date_prompt), WAITING_FOR_DATE)

            # A date arrived in the same turn: advance through the usual chain.
            if not self.state.get('crop'):
                self.state['conv_state'] = WAITING_FOR_CROP
                return self._response(self._msg('ask_crop'), WAITING_FOR_CROP)
            if not self.state.get('quantity_kg'):
                self.state['conv_state'] = WAITING_FOR_QUANTITY
                return self._response(self._msg('ask_quantity', {'crop': self.state['crop']}), WAITING_FOR_QUANTITY)
            if not self.state.get('preferred_time'):
                self.state['conv_state'] = WAITING_FOR_TIME
                msg = self._msg('multi_prefix', {
                    'date': human_date(self.state['date']),
                    'crop': self.state['crop'],
                    'quantity': f"{self.state['quantity_kg']} kg"
                })
                return self._response(msg, WAITING_FOR_TIME)
            self.state['conv_state'] = CHECKING_SLOTS
            if available_slots:
                return self.check_slots_with_backend(available_slots)
            return self._response("Checking available slots...", CHECKING_SLOTS, action='check_slots')

        # -------------------------------------------------------------------
        # STATE: WAITING_FOR_DATE
        # -------------------------------------------------------------------
        if curr_state == WAITING_FOR_DATE:
            # Check for weekday clarification (e.g. "Monday" -> "Which Monday do you mean? Please tell me the date.")
            if date_note == 'weekday' and not any(w in raw_text for w in ('next', 'వచ్చే', 'अगले', 'coming')):
                weekday_name = 'Monday'
                for tok in tokens:
                    if tok in WEEKDAYS:
                        weekday_name = tok.capitalize()
                        break
                return self._response(self._msg('clarify_weekday', {'weekday': weekday_name}), WAITING_FOR_DATE)

            if not extracted_date:
                return self._response(self._msg('invalid_date'), WAITING_FOR_DATE)

            self.state['date'] = extracted_date
            if extracted_crop:
                self.state['crop'] = extracted_crop
            if extracted_qty and extracted_qty > 0:
                self.state['quantity_kg'] = clean_number(extracted_qty)
            if extracted_time and not extracted_time.get('is_bare_hour'):
                self.state['preferred_time'] = extracted_time.get('standard_time')
                self.state['preferred_time_label'] = extracted_time.get('label')
                self.state['pending_time_pref'] = extracted_time

            # Advance to missing item in controlled order: Crop -> Quantity -> Time
            if not self.state.get('crop'):
                self.state['conv_state'] = WAITING_FOR_CROP
                return self._response(self._msg('ask_crop'), WAITING_FOR_CROP)
            elif not self.state.get('quantity_kg'):
                self.state['conv_state'] = WAITING_FOR_QUANTITY
                return self._response(self._msg('ask_quantity', {'crop': self.state['crop']}), WAITING_FOR_QUANTITY)
            elif not self.state.get('preferred_time'):
                self.state['conv_state'] = WAITING_FOR_TIME
                msg = self._msg('multi_prefix', {
                    'date': human_date(self.state['date']),
                    'crop': self.state['crop'],
                    'quantity': f"{self.state['quantity_kg']} kg"
                })
                return self._response(msg, WAITING_FOR_TIME)
            else:
                self.state['conv_state'] = CHECKING_SLOTS
                if available_slots:
                    return self.check_slots_with_backend(available_slots)
                return self._response("Checking available slots...", CHECKING_SLOTS, action='check_slots')

        # -------------------------------------------------------------------
        # STATE: WAITING_FOR_CROP
        # -------------------------------------------------------------------
        if curr_state == WAITING_FOR_CROP:
            crop_match = extracted_crop
            if not crop_match:
                for tok in tokens:
                    m = match_token(tok, _CROP_LOOKUP, expected=True)
                    if m and m['confidence'] >= CONF_MEDIUM:
                        crop_match = m['value']
                        break
                    if tok in _CROP_STT_LOOKUP:
                        crop_match = _CROP_STT_LOOKUP[tok]
                        break
            if not crop_match:
                return self._response(self._msg('invalid_crop'), WAITING_FOR_CROP)

            self.state['crop'] = crop_match
            if extracted_qty and extracted_qty > 0:
                self.state['quantity_kg'] = clean_number(extracted_qty)
            if extracted_time and not extracted_time.get('is_bare_hour'):
                self.state['preferred_time'] = extracted_time.get('standard_time')
                self.state['preferred_time_label'] = extracted_time.get('label')
                self.state['pending_time_pref'] = extracted_time

            if not self.state.get('quantity_kg'):
                self.state['conv_state'] = WAITING_FOR_QUANTITY
                return self._response(self._msg('ask_quantity', {'crop': self.state['crop']}), WAITING_FOR_QUANTITY)
            elif not self.state.get('preferred_time'):
                self.state['conv_state'] = WAITING_FOR_TIME
                return self._response(self._msg('ask_time'), WAITING_FOR_TIME)
            else:
                self.state['conv_state'] = CHECKING_SLOTS
                if available_slots:
                    return self.check_slots_with_backend(available_slots)
                return self._response("Checking available slots...", CHECKING_SLOTS, action='check_slots')

        # -------------------------------------------------------------------
        # STATE: WAITING_FOR_QUANTITY
        # -------------------------------------------------------------------
        if curr_state == WAITING_FOR_QUANTITY:
            q_val, _, _, _ = parse_quantity(tokens, expected=True)
            if not q_val or q_val <= 0:
                return self._response(self._msg('invalid_quantity'), WAITING_FOR_QUANTITY)

            self.state['quantity_kg'] = clean_number(q_val)
            if extracted_time and not extracted_time.get('is_bare_hour'):
                self.state['preferred_time'] = extracted_time.get('standard_time')
                self.state['preferred_time_label'] = extracted_time.get('label')
                self.state['pending_time_pref'] = extracted_time

            if not self.state.get('preferred_time'):
                self.state['conv_state'] = WAITING_FOR_TIME
                return self._response(self._msg('ask_time'), WAITING_FOR_TIME)
            else:
                self.state['conv_state'] = CHECKING_SLOTS
                if available_slots:
                    return self.check_slots_with_backend(available_slots)
                return self._response("Checking available slots...", CHECKING_SLOTS, action='check_slots')

        # -------------------------------------------------------------------
        # STATE: WAITING_FOR_TIME
        # -------------------------------------------------------------------
        if curr_state == WAITING_FOR_TIME:
            tp = parse_time_preference(raw_text, tokens, expected='time')
            if not tp:
                return self._response(self._msg('ask_time'), WAITING_FOR_TIME)

            if tp.get('is_bare_hour'):
                self.state['pending_hour'] = tp['hour']
                self.state['pending_time_pref'] = tp
                self.state['conv_state'] = CONFIRMING_TIME_AM_PM
                return self._response(self._msg('confirm_bare_time', {'time': f"{tp['hour']} AM"}), CONFIRMING_TIME_AM_PM)

            self.state['preferred_time'] = tp.get('standard_time')
            self.state['preferred_time_label'] = tp.get('label')
            self.state['pending_time_pref'] = tp
            self.state['conv_state'] = CHECKING_SLOTS
            if available_slots:
                return self.check_slots_with_backend(available_slots)
            return self._response("Checking available slots...", CHECKING_SLOTS, action='check_slots')

        # -------------------------------------------------------------------
        # STATE: CONFIRMING_TIME_AM_PM ("Do you mean 9 AM?")
        # -------------------------------------------------------------------
        if curr_state == CONFIRMING_TIME_AM_PM:
            if is_affirmative(raw_text):
                h = self.state.get('pending_hour', 9)
                self.state['preferred_time'] = f"{h:02d}:00 AM"
                self.state['preferred_time_label'] = f"{h} AM"
                self.state['pending_time_pref'] = {'hour': h, 'period': 'AM', 'standard_time': f"{h:02d}:00 AM", 'label': f"{h} AM"}
                self.state['pending_hour'] = None
                self.state['conv_state'] = CHECKING_SLOTS
                if available_slots:
                    return self.check_slots_with_backend(available_slots)
                return self._response("Checking available slots...", CHECKING_SLOTS, action='check_slots')
            elif is_negative(raw_text):
                self.state['pending_hour'] = None
                self.state['conv_state'] = WAITING_FOR_TIME
                return self._response(self._msg('ask_time'), WAITING_FOR_TIME)
            else:
                tp = parse_time_preference(raw_text, tokens, expected='time')
                if tp:
                    self.state['preferred_time'] = tp.get('standard_time')
                    self.state['preferred_time_label'] = tp.get('label')
                    self.state['pending_time_pref'] = tp
                    self.state['conv_state'] = CHECKING_SLOTS
                    if available_slots:
                        return self.check_slots_with_backend(available_slots)
                    return self._response("Checking available slots...", CHECKING_SLOTS, action='check_slots')
                return self._response(self._msg('ask_time'), WAITING_FOR_TIME)

        # -------------------------------------------------------------------
        # STATE: WAITING_FOR_SLOT_CONFIRMATION ("9 AM is available. Book it?")
        # -------------------------------------------------------------------
        if curr_state == WAITING_FOR_SLOT_CONFIRMATION:
            if is_affirmative(raw_text):
                self.state['conv_state'] = WAITING_FOR_CONFIRMATION
                msg = self._msg('final_summary', {
                    'date': human_date(self.state['date']),
                    'crop': self.state['crop'],
                    'quantity': f"{self.state['quantity_kg']} kg",
                    'time': self.state['preferred_time_label'] or ''
                })
                return self._response(msg, WAITING_FOR_CONFIRMATION, action='request_confirmation')
            elif is_negative(raw_text):
                self.state['conv_state'] = WAITING_FOR_TIME
                return self._response(self._msg('booking_cancelled_prompt'), WAITING_FOR_TIME)
            else:
                matched = match_slot_choice(raw_text, self.state.get('available_slots') or available_slots or [])
                if matched:
                    self.state['selected_slot'] = matched
                    self.state['preferred_time_label'] = format_slot_time_label(matched.get('start_time'))
                    self.state['conv_state'] = WAITING_FOR_CONFIRMATION
                    msg = self._msg('final_summary', {
                        'date': human_date(self.state['date']),
                        'crop': self.state['crop'],
                        'quantity': f"{self.state['quantity_kg']} kg",
                        'time': self.state['preferred_time_label']
                    })
                    return self._response(msg, WAITING_FOR_CONFIRMATION, action='request_confirmation')
                return self._response(self._msg('slot_available_confirm', {'time': self.state.get('preferred_time_label')}),
                                      WAITING_FOR_SLOT_CONFIRMATION)

        # -------------------------------------------------------------------
        # STATE: WAITING_FOR_SLOT_SELECTION (Farmer chooses from available slots)
        # -------------------------------------------------------------------
        if curr_state == WAITING_FOR_SLOT_SELECTION:
            slots = self.state.get('available_slots') or available_slots or []
            matched = match_slot_choice(raw_text, slots)
            if matched:
                self.state['selected_slot'] = matched
                self.state['preferred_time_label'] = format_slot_time_label(matched.get('start_time'))
                self.state['conv_state'] = WAITING_FOR_CONFIRMATION
                msg = self._msg('final_summary', {
                    'date': human_date(self.state['date']),
                    'crop': self.state['crop'],
                    'quantity': f"{self.state['quantity_kg']} kg",
                    'time': self.state['preferred_time_label']
                })
                return self._response(msg, WAITING_FOR_CONFIRMATION, action='request_confirmation')
            else:
                slot_labels = format_slot_list_spoken(slots, self.language)
                return self._response(self._msg('choose_available_slot', {'slots': slot_labels}),
                                      WAITING_FOR_SLOT_SELECTION)

        # -------------------------------------------------------------------
        # STATE: WAITING_FOR_CONFIRMATION (Final confirmation)
        # -------------------------------------------------------------------
        if curr_state == WAITING_FOR_CONFIRMATION:
            if is_affirmative(raw_text):
                self.state['conv_state'] = BOOKING
                return self._response(self._msg('booking_in_progress'), BOOKING, action='create_booking')
            elif is_negative(raw_text):
                self.state['conv_state'] = COMPLETED
                return self._response(self._msg('booking_cancelled'), COMPLETED)
            else:
                msg = self._msg('final_summary', {
                    'date': human_date(self.state['date']),
                    'crop': self.state['crop'],
                    'quantity': f"{self.state['quantity_kg']} kg",
                    'time': self.state['preferred_time_label']
                })
                return self._response(msg, WAITING_FOR_CONFIRMATION, action='request_confirmation')

        return self._response(self._msg('repeat'), curr_state)
