"""KisanFlow - farmer procurement slot booking and queue management (SIH26032).

Layers
------
frontend : Flask/Jinja templates + static HTML/CSS/vanilla JS (mobile first)
backend  : this file, a Flask JSON API
database : SQLAlchemy Core, parameterised SQL - SQLite locally, PostgreSQL in production

Farmer journey stays deliberately short:
login -> choose centre/date/slot -> book -> token -> check my turn -> get called -> counter.
"""
import logging
import os
import re
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash

import nlp   # local, pure-Python language understanding (no I/O, no database)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-change-this-secret')
app.config.update(
    SESSION_COOKIE_NAME='session',
    SESSION_COOKIE_PATH='/',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE', '0') == '1',
    SESSION_REFRESH_EACH_REQUEST=True,
    # Browser refresh must reuse the same session. A short permanent lifetime
    # keeps farmer/admin logged in across refresh; default (browser-session
    # cookie) is kept unless SESSION_DAYS is set.
    PERMANENT_SESSION_LIFETIME=timedelta(days=int(os.environ.get('SESSION_DAYS', '7'))),
    MAX_CONTENT_LENGTH=64 * 1024,          # every API payload is small
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
if app.secret_key == 'dev-only-change-this-secret' and os.environ.get('COOKIE_SECURE') == '1':
    app.logger.warning('COOKIE_SECURE=1 but SECRET_KEY is still the development default.')

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///procurement.db')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
IS_SQLITE = DATABASE_URL.startswith('sqlite')
if IS_SQLITE:
    # SQLite + Flask dev server: each request thread must get a FRESH file
    # connection (NullPool) that closes on return. A pooled QueuePool keeps
    # idle connections holding read locks; after 1-2 writes the next
    # request blocks on "database is locked" and the UI looks dead.
    # check_same_thread=False allows the threaded server to use the driver.
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        poolclass=NullPool,
        connect_args={'timeout': 15, 'check_same_thread': False},
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
if IS_SQLITE:
    @event.listens_for(engine, 'connect')
    def _sqlite_connection_pragmas(dbapi_connection, _record):
        """WAL lets readers work while a booking writes; busy_timeout rides out short lock waits."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA busy_timeout=15000')
        finally:
            cursor.close()

# ---------------------------------------------------------------------------
# Domain constants - keep in sync with the farmer and admin frontends
# ---------------------------------------------------------------------------
WAITING, CALLED, SERVING, COMPLETED, CANCELLED = 'Waiting', 'Called', 'Serving', 'Completed', 'Cancelled'
ACTIVE_STATES = (WAITING, CALLED, SERVING)      # bookings that still occupy slot capacity
ACTIVE_SQL = "('Waiting','Called','Serving')"
ALL_STATES = (WAITING, CALLED, SERVING, COMPLETED, CANCELLED)
# Quantity tiers (tons): HIGH >= 5, MID 1..5 (5 excluded -> HIGH), LOW < 1.
TIER_HIGH, TIER_MID, TIER_LOW = 'High', 'Mid', 'Low'
TIER_ORDER = {TIER_HIGH: 0, TIER_MID: 1, TIER_LOW: 2}
TIER_SQL_ORDER = "CASE b.tier WHEN 'High' THEN 0 WHEN 'Mid' THEN 1 ELSE 2 END"


def quantity_tier(quantity):
    """Map a tonnage to its strict priority tier (server-side authority)."""
    try:
        q = float(quantity)
    except (TypeError, ValueError):
        return None
    if q >= 5:
        return TIER_HIGH
    if q >= 1:
        return TIER_MID
    return TIER_LOW
VALID_TRANSITIONS = {                            # PRD section 10 - no invalid state jumps
    WAITING: {CALLED, SERVING, CANCELLED},
    CALLED: {SERVING, WAITING, CANCELLED},
    SERVING: {COMPLETED, CANCELLED},
    COMPLETED: set(),
    CANCELLED: {WAITING},
}
PROCUREMENT_STATUS = {
    WAITING: 'Booked', CALLED: 'Called to counter', SERVING: 'Quality check',
    COMPLETED: 'Procured', CANCELLED: 'Cancelled',
}
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
# Abuse guards for the prototype - override through the environment if needed.
LOGIN_ATTEMPT_LIMIT = int(os.environ.get('LOGIN_ATTEMPT_LIMIT', '12'))
REGISTER_ATTEMPT_LIMIT = int(os.environ.get('REGISTER_ATTEMPT_LIMIT', '20'))


def pk_definition():
    return 'INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY' if engine.dialect.name == 'postgresql' else 'INTEGER PRIMARY KEY AUTOINCREMENT'


def db_init():
    pk = pk_definition()
    with engine.begin() as c:
        c.execute(text(f'''CREATE TABLE IF NOT EXISTS users (
            id {pk}, role VARCHAR(20) NOT NULL, name VARCHAR(200) NOT NULL,
            phone VARCHAR(30) UNIQUE, username VARCHAR(100) UNIQUE,
            password_hash VARCHAR(300) NOT NULL, created_at VARCHAR(50) NOT NULL)'''))
        c.execute(text(f'''CREATE TABLE IF NOT EXISTS centres (
            id {pk}, name VARCHAR(200) NOT NULL, daily_capacity INTEGER NOT NULL DEFAULT 40)'''))
        c.execute(text(f'''CREATE TABLE IF NOT EXISTS slots (
            id {pk}, centre_id INTEGER NOT NULL, slot_date VARCHAR(20) NOT NULL,
            start_time VARCHAR(20) NOT NULL, end_time VARCHAR(20) NOT NULL,
            capacity INTEGER NOT NULL,
            cap_high INTEGER NOT NULL DEFAULT 0, cap_mid INTEGER NOT NULL DEFAULT 0,
            cap_low INTEGER NOT NULL DEFAULT 0,
            UNIQUE(centre_id, slot_date, start_time))'''))
        c.execute(text(f'''CREATE TABLE IF NOT EXISTS bookings (
            id {pk}, user_id INTEGER, slot_id INTEGER NOT NULL, name VARCHAR(200) NOT NULL,
            phone VARCHAR(30) NOT NULL, crop VARCHAR(100) NOT NULL,
            quantity_tons REAL NOT NULL DEFAULT 0, tier VARCHAR(10) NOT NULL DEFAULT 'Low',
            token INTEGER NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'Waiting', procurement_status VARCHAR(50) NOT NULL DEFAULT 'Booked',
            payment_status VARCHAR(50) NOT NULL DEFAULT 'Not started', created_at VARCHAR(50) NOT NULL,
            booking_created_at TIMESTAMP, called_at TIMESTAMP, served_at TIMESTAMP,
            completed_at TIMESTAMP, cancelled_at TIMESTAMP)'''))
        c.execute(text(f'''CREATE TABLE IF NOT EXISTS notifications (
            id {pk}, user_id INTEGER NOT NULL, booking_id INTEGER, message VARCHAR(500) NOT NULL,
            kind VARCHAR(50) NOT NULL DEFAULT 'info', is_read INTEGER NOT NULL DEFAULT 0,
            created_at VARCHAR(50) NOT NULL)'''))
        # Performance indexes - additive and safe on existing databases (no data loss).
        c.execute(text('CREATE INDEX IF NOT EXISTS idx_bookings_user ON bookings(user_id)'))
        c.execute(text('CREATE INDEX IF NOT EXISTS idx_bookings_slot ON bookings(slot_id, status)'))
        c.execute(text('CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, is_read)'))
        c.execute(text('CREATE INDEX IF NOT EXISTS idx_slots_lookup ON slots(centre_id, slot_date)'))
        # Safe migration for databases created by older KisanFlow versions.
        if engine.dialect.name == 'sqlite':
            booking_cols = {r[1] for r in c.execute(text("PRAGMA table_info(bookings)")).fetchall()}
            if 'user_id' not in booking_cols:
                c.execute(text('ALTER TABLE bookings ADD COLUMN user_id INTEGER'))
            if 'quantity_tons' not in booking_cols:
                c.execute(text('ALTER TABLE bookings ADD COLUMN quantity_tons REAL NOT NULL DEFAULT 0'))
            if 'tier' not in booking_cols:
                c.execute(text("ALTER TABLE bookings ADD COLUMN tier VARCHAR(10) NOT NULL DEFAULT 'Low'"))
            for col in ('booking_created_at', 'called_at', 'served_at', 'completed_at', 'cancelled_at'):
                if col not in booking_cols:
                    c.execute(text(f'ALTER TABLE bookings ADD COLUMN {col} TIMESTAMP'))
            slot_cols = {r[1] for r in c.execute(text("PRAGMA table_info(slots)")).fetchall()}
            for col in ('cap_high', 'cap_mid', 'cap_low'):
                if col not in slot_cols:
                    c.execute(text(f'ALTER TABLE slots ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0'))
            # Old slots created before quotas: split capacity 50/30/20 across High/Mid/Low.
            c.execute(text('''UPDATE slots SET
                    cap_high = CAST(capacity * 50 / 100 AS INTEGER),
                    cap_mid = CAST(capacity * 30 / 100 AS INTEGER),
                    cap_low = capacity - CAST(capacity * 50 / 100 AS INTEGER) - CAST(capacity * 30 / 100 AS INTEGER)
                WHERE cap_high + cap_mid + cap_low <= 0 AND capacity > 0'''))
            # Old bookings created before quantity: keep them as Low tier.
            c.execute(text("UPDATE bookings SET tier='Low' WHERE tier IS NULL OR tier=''"))
            # Existing rows have a trustworthy creation time, but no historical
            # transition times. Preserve what is known; leave the rest NULL.
            c.execute(text('UPDATE bookings SET booking_created_at=CAST(created_at AS TIMESTAMP) WHERE booking_created_at IS NULL'))
        else:
            c.execute(text("ALTER TABLE bookings ADD COLUMN IF NOT EXISTS user_id INTEGER"))
            c.execute(text("ALTER TABLE bookings ADD COLUMN IF NOT EXISTS quantity_tons DOUBLE PRECISION NOT NULL DEFAULT 0"))
            c.execute(text("ALTER TABLE bookings ADD COLUMN IF NOT EXISTS tier VARCHAR(10) NOT NULL DEFAULT 'Low'"))
            for col in ('booking_created_at', 'called_at', 'served_at', 'completed_at', 'cancelled_at'):
                c.execute(text(f'ALTER TABLE bookings ADD COLUMN IF NOT EXISTS {col} TIMESTAMP'))
            c.execute(text("ALTER TABLE slots ADD COLUMN IF NOT EXISTS cap_high INTEGER NOT NULL DEFAULT 0"))
            c.execute(text("ALTER TABLE slots ADD COLUMN IF NOT EXISTS cap_mid INTEGER NOT NULL DEFAULT 0"))
            c.execute(text("ALTER TABLE slots ADD COLUMN IF NOT EXISTS cap_low INTEGER NOT NULL DEFAULT 0"))
            c.execute(text('''UPDATE slots SET
                    cap_high = (capacity * 50) / 100,
                    cap_mid = (capacity * 30) / 100,
                    cap_low = capacity - ((capacity * 50) / 100) - ((capacity * 30) / 100)
                WHERE cap_high + cap_mid + cap_low <= 0 AND capacity > 0'''))
            c.execute(text("UPDATE bookings SET tier='Low' WHERE tier IS NULL OR tier=''"))
            c.execute(text('UPDATE bookings SET booking_created_at=created_at WHERE booking_created_at IS NULL'))
        if c.execute(text('SELECT COUNT(*) FROM centres')).scalar() == 0:
            c.execute(text("INSERT INTO centres(name,daily_capacity) VALUES ('Main Procurement Centre',40),('Village Collection Centre',30),('District Procurement Centre',60)"))
        admin = c.execute(text("SELECT id FROM users WHERE role='admin' LIMIT 1")).first()
        if not admin:
            c.execute(text("INSERT INTO users(role,name,username,password_hash,created_at) VALUES ('admin','KisanFlow Centre Admin','admin@kisanflow.com',:ph,:now)"), {
                'ph': generate_password_hash('Admin@123'), 'now': datetime.now().isoformat(timespec='seconds')})


def rows(sql, params=None):
    # Read-only queries must NOT open a write transaction (engine.begin()).
    # On SQLite, holding a BEGIN while polling every 2-3s from two pages can
    # pile up behind a booking/call write and hang every later request.
    with engine.connect() as c:
        return [dict(r._mapping) for r in c.execute(text(sql), params or {}).fetchall()]


def current_user():
    uid = session.get('user_id')
    if not uid:
        return None
    r = rows('SELECT id,role,name,phone,username FROM users WHERE id=:id', {'id': uid})
    return r[0] if r else None


def login_required(role=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # CORS preflight never carries cookies; let after_request answer
            # it (204 + CORS headers) instead of failing it with a 401.
            if request.method == 'OPTIONS':
                return ('', 204)
            u = current_user()
            # API routes must ALWAYS get machine-readable JSON (401/403), never
            # an HTML redirect. fetch() follows a 302 transparently to /login
            # (HTTP 200), so the frontend would see "ok" with garbage data and
            # could neither render status nor redirect — the "stuck" symptom.
            # Only real HTML pages get redirect responses.
            is_api = request.path.startswith('/api/') or request.path == '/health'
            if not u:
                if is_api or request.method != 'GET':
                    return jsonify(error='Login required'), 401
                return redirect(url_for('login'))
            if role and u['role'] != role:
                if is_api or request.method != 'GET':
                    return jsonify(error='Access denied'), 403
                return redirect(url_for('home'))
            return fn(*args, **kwargs)
        return wrapper
    return deco


def one(sql, params=None):
    r = rows(sql, params)
    return r[0] if r else None


def now_iso():
    return datetime.now().isoformat(timespec='seconds')


def lifecycle_now_iso():
    """Higher precision event time for future duration analysis."""
    return datetime.now().isoformat(timespec='microseconds')


def api_error(message, status=400):
    """Uniform error envelope: {"success": false, "error": "..."} (PRD section 19)."""
    return jsonify(success=False, error=message), status


def today_str():
    return datetime.now().strftime('%Y-%m-%d')


def valid_date_str(value):
    if not value or not DATE_RE.match(str(value)):
        return False
    try:
        datetime.strptime(str(value), '%Y-%m-%d')
        return True
    except ValueError:
        return False


# --- login / registration throttling (in-memory, resets on restart) ---------
_attempts = {}
_attempts_lock = threading.Lock()


def too_many_attempts(key, limit=12, window=300):
    now = time.time()
    with _attempts_lock:
        recent = [t for t in _attempts.get(key, []) if now - t < window]
        blocked = len(recent) >= limit
        if not blocked:
            recent.append(now)
        _attempts[key] = recent
    return blocked


# --- booking write serialisation -------------------------------------------
# PostgreSQL serialises bookings with SELECT ... FOR UPDATE on the slot row.
# Local SQLite is one file written by one process, so a single in-process lock
# removes the read-then-write race without locking the whole database.
_sqlite_booking_lock = threading.Lock() if IS_SQLITE else None


def booking_write_guard():
    return _sqlite_booking_lock if _sqlite_booking_lock is not None else nullcontext()


# Extra allowed browser origins for local split-port dev (e.g. Live Server :5500
# calling Flask :5000). Same-origin deployments need none of this.
FRONTEND_ORIGINS = {o.strip().lower() for o in os.environ.get('FRONTEND_ORIGINS', '').replace(',', ' ').split() if o.strip()}


def _is_allowed_origin(origin):
    """True for the request's own host, local loopback on any port, or an
    explicitly configured FRONTEND_ORIGINS entry. A foreign site such as
    https://evil.example is never allowed."""
    try:
        netloc = urlparse(origin).netloc.lower()
    except Exception:
        return False
    if not netloc:
        return False
    host = netloc.split('@')[-1].split(':')[0]
    allowed_hosts = {
        (request.host or '').lower().split(':')[0],
        (request.headers.get('X-Forwarded-Host') or '').lower().split(':')[0],
        (request.environ.get('HTTP_HOST') or '').lower().split(':')[0],
        'localhost', '127.0.0.1',
    }
    allowed_hosts.discard('')
    if host in allowed_hosts:
        return True
    return netloc in FRONTEND_ORIGINS or origin.lower() in FRONTEND_ORIGINS


@app.before_request
def block_cross_site_writes():
    """Light CSRF defence: browsers send Origin on cross-site writes.

    Session cookies are already SameSite=Lax, so this only hardens the
    POST/PUT/PATCH/DELETE surface against an obvious foreign origin.
    """
    if request.method == 'OPTIONS':
        # CORS preflight carries no cookies; answer here so it never hits
        # login_required (which would 401 it) or the origin write-block.
        return ('', 204)
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return None
    origin = request.headers.get('Origin')
    if not origin:
        return None
    if _is_allowed_origin(origin):
        return None
    app.logger.warning('Blocked cross-site %s %s (Origin=%s)', request.method, request.path, origin)
    return api_error('Cross-site request blocked.', 403)


@app.get('/health')
@app.get('/api/health')
def health():
    """Liveness + database readiness for the cloud platform health check."""
    try:
        with engine.connect() as c:
            c.execute(text('SELECT 1'))
        return jsonify(status='ok', database='connected', engine=engine.dialect.name)
    except SQLAlchemyError:
        app.logger.exception('Health check failed: database unreachable')
        return jsonify(status='error', database='unavailable'), 503

@app.route('/')
def home(): return render_template('index.html', user=current_user())

@app.route('/login', methods=['GET','POST'])
@app.route('/api/login', methods=['POST'])
def login():
    """Farmer/admin login. `identifier` = 10-digit mobile number or admin username."""
    if request.method == 'GET': return render_template('login.html')
    data = request.get_json(silent=True) or request.form
    identifier = str(data.get('identifier','')).strip()[:120]
    password = str(data.get('password',''))
    if not identifier or not password:
        return api_error('Enter your mobile number/username and password.', 400)
    if too_many_attempts(f'login:{request.remote_addr}', limit=LOGIN_ATTEMPT_LIMIT):
        return api_error('Too many login attempts. Please wait a few minutes and try again.', 429)
    user = rows("SELECT id,role,name,phone,username,password_hash FROM users WHERE phone=:x OR username=:x LIMIT 1", {'x': identifier})
    if not user or not check_password_hash(user[0]['password_hash'], password):
        app.logger.info('Failed login attempt from %s', request.remote_addr)
        return api_error('Invalid login details.', 401)
    session.clear()
    session['user_id'] = user[0]['id']
    session['role'] = user[0]['role']
    session.permanent = True   # survive browser refresh (see PERMANENT_SESSION_LIFETIME)
    return jsonify(success=True, role=user[0]['role'], redirect='/dashboard' if user[0]['role']=='admin' else '/farmer')

@app.route('/register', methods=['GET','POST'])
@app.route('/api/register', methods=['POST'])
def register():
    """Farmer self-registration. The mobile number is the unique farmer identifier."""
    if request.method == 'GET': return render_template('register.html')
    data = request.get_json(silent=True) or request.form
    name = str(data.get('name','')).strip()[:120]
    phone = str(data.get('phone','')).strip()
    password = str(data.get('password',''))
    if not name or not phone or not password:
        return api_error('Fill all fields.', 400)
    if not phone.isdigit() or len(phone) != 10:
        return api_error('Enter a valid 10-digit mobile number.', 400)
    if len(password) < 6:
        return api_error('Password must be at least 6 characters.', 400)
    if too_many_attempts(f'register:{request.remote_addr}', limit=REGISTER_ATTEMPT_LIMIT):
        return api_error('Too many sign-up attempts. Please try again later.', 429)
    try:
        with booking_write_guard(), engine.begin() as c:
            q = c.execute(text("INSERT INTO users(role,name,phone,password_hash,created_at) VALUES ('farmer',:n,:p,:h,:now) RETURNING id"),
                          {'n':name,'p':phone,'h':generate_password_hash(password),'now':now_iso()})
            uid = int(q.scalar_one())
    except IntegrityError:
        return api_error('This mobile number is already registered.', 409)
    except SQLAlchemyError:
        app.logger.exception('Registration failed for a farmer account')
        return api_error('Could not create account. Please try again.', 500)
    session.clear()
    session['user_id'] = uid
    session['role'] = 'farmer'
    session.permanent = True   # survive browser refresh (see PERMANENT_SESSION_LIFETIME)
    return jsonify(success=True, redirect='/farmer'), 201

@app.post('/logout')
@app.post('/api/logout')
def logout():
    session.clear()
    return jsonify(success=True)

@app.route('/farmer')
@login_required('farmer')
def farmer(): return render_template('farmer.html', user=current_user())

@app.route('/farmer/status')
@login_required('farmer')
def farmer_status():
    """Dedicated independent farmer status page (refresh-safe).

    Server-rendered only for authenticated farmers, so F5 re-authenticates via
    the Flask session cookie and re-renders this page — never the home/login
    page. All status data loads from the API below; no JS memory is trusted.
    """
    return render_template('farmer_status.html', user=current_user())

@app.route('/dashboard')
def dashboard():
    u = current_user()
    if not u or u['role'] != 'admin':
        return redirect('/admin')
    return render_template('dashboard.html', user=u)

@app.route('/admin')
def admin():
    u = current_user()
    if u and u['role'] == 'admin':
        return render_template('admin.html', user=u)
    return render_template('admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """Separate admin entry point (FR-03) - only role='admin' accounts pass."""
    data = request.get_json(silent=True) or request.form
    identifier = str(data.get('identifier','')).strip()[:120]
    password = str(data.get('password',''))
    if too_many_attempts(f'admin-login:{request.remote_addr}', limit=LOGIN_ATTEMPT_LIMIT):
        return api_error('Too many admin login attempts. Please wait a few minutes and try again.', 429)
    user = rows("SELECT id,password_hash FROM users WHERE username=:x AND role='admin' LIMIT 1", {'x': identifier})
    if not user or not check_password_hash(user[0]['password_hash'], password):
        app.logger.info('Failed admin login attempt from %s', request.remote_addr)
        return api_error('Invalid admin login details.', 401)
    session.clear()
    session['user_id'] = user[0]['id']
    session['role'] = 'admin'
    session.permanent = True   # survive browser refresh (see PERMANENT_SESSION_LIFETIME)
    return jsonify(success=True, redirect='/admin')

@app.get('/api/me')
@login_required()
def me(): return jsonify(current_user())

@app.get('/api/auth/status')
def auth_status():
    """Session probe used by protected pages on load/refresh.

    Always HTTP 200 so the frontend can tell 'logged out' apart from
    'network error' without guessing from 401s. Never exposes hashes:
    current_user() only selects id/role/name/phone/username.
    """
    u = current_user()
    if not u:
        return jsonify(authenticated=False, user=None)
    return jsonify(authenticated=True, user={'id': u['id'], 'role': u['role'], 'name': u.get('name')})

@app.get('/api/centres')
@login_required()
def centres(): return jsonify(rows('SELECT * FROM centres ORDER BY id'))

@app.post('/api/generate-slots')
@login_required('admin')
def generate_slots():
    """Create slots for one centre and date with High/Mid/Low quotas (strict, sum = capacity)."""
    data=request.get_json(silent=True) or {}
    raw_date=str(data.get('date','')).strip()
    try:
        cid=int(data.get('centre_id')); cap=int(data.get('capacity',10)); interval=int(data.get('interval',60))
    except (TypeError,ValueError):
        return api_error('Invalid slot settings.',400)
    if not valid_date_str(raw_date): return api_error('Choose a valid date (YYYY-MM-DD).',400)
    if raw_date < today_str(): return api_error('Slots cannot be created for a past date.',400)
    if not 1 <= cap <= 200: return api_error('Capacity must be between 1 and 200 farmers.',400)
    # Per-tier quotas: admin sets High/Mid/Low seats; default split 50/30/20.
    try:
        qh = int(data.get('cap_high', (cap * 50) // 100))
        qm = int(data.get('cap_mid', (cap * 30) // 100))
    except (TypeError, ValueError):
        return api_error('Tier seats must be whole numbers.', 400)
    ql = cap - qh - qm if data.get('cap_low') is None else None
    if ql is None:
        try:
            ql = int(data.get('cap_low'))
        except (TypeError, ValueError):
            return api_error('Tier seats must be whole numbers.', 400)
    if min(qh, qm, ql) < 0: return api_error('Tier seats cannot be negative.', 400)
    if qh + qm + ql != cap:
        return api_error(f'High + Mid + Low seats must equal capacity ({cap}).', 400)
    if interval not in (15,30,45,60,90,120): return api_error('Slot length must be 15, 30, 45, 60, 90 or 120 minutes.',400)
    try:
        cur=datetime.strptime(str(data.get('start','09:00')),'%H:%M')
        finish=datetime.strptime(str(data.get('end','17:00')),'%H:%M')
    except ValueError: return api_error('Use HH:MM for the start and end time.',400)
    if cur>=finish: return api_error('End time must be after start time.',400)
    if not one('SELECT id FROM centres WHERE id=:id',{'id':cid}): return api_error('Centre not found.',404)
    created=0
    with engine.begin() as c:
        while cur<finish:
            nxt=cur+timedelta(minutes=interval)
            if nxt>finish: break
            start_label,end_label=cur.strftime('%I:%M %p'),nxt.strftime('%I:%M %p')
            exists=c.execute(text('SELECT id FROM slots WHERE centre_id=:cid AND slot_date=:d AND start_time=:a'),{'cid':cid,'d':raw_date,'a':start_label}).first()
            if not exists:
                c.execute(text('INSERT INTO slots(centre_id,slot_date,start_time,end_time,capacity,cap_high,cap_mid,cap_low) VALUES(:cid,:d,:a,:b,:cap,:qh,:qm,:ql)'),{'cid':cid,'d':raw_date,'a':start_label,'b':end_label,'cap':cap,'qh':qh,'qm':qm,'ql':ql})
                created+=1
            cur=nxt
    return jsonify(success=True,created=created)

@app.get('/api/slots')
@login_required()
def get_slots():
    """Slot list with per-tier remaining seats for one centre + date (FR-04, FR-06)."""
    raw_date=(request.args.get('date') or '').strip()
    try: cid=int(request.args.get('centre_id'))
    except (TypeError,ValueError): return api_error('Choose a procurement centre.',400)
    if not valid_date_str(raw_date): return api_error('Choose a valid date (YYYY-MM-DD).',400)
    return jsonify(rows(f'''SELECT s.id,s.centre_id,s.slot_date,s.start_time,s.end_time,s.capacity,
            COALESCE(s.cap_high,0) AS cap_high, COALESCE(s.cap_mid,0) AS cap_mid, COALESCE(s.cap_low,0) AS cap_low,
            c.name AS centre_name,
            (SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.status IN {ACTIVE_SQL}) AS booked,
            (s.capacity-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.status IN {ACTIVE_SQL})) AS remaining,
            (SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='High' AND b.status IN {ACTIVE_SQL}) AS booked_high,
            (SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='Mid' AND b.status IN {ACTIVE_SQL}) AS booked_mid,
            (SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='Low' AND b.status IN {ACTIVE_SQL}) AS booked_low,
            (COALESCE(s.cap_high,0)-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='High' AND b.status IN {ACTIVE_SQL})) AS remaining_high,
            (COALESCE(s.cap_mid,0)-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='Mid' AND b.status IN {ACTIVE_SQL})) AS remaining_mid,
            (COALESCE(s.cap_low,0)-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='Low' AND b.status IN {ACTIVE_SQL})) AS remaining_low
        FROM slots s JOIN centres c ON c.id=s.centre_id
        WHERE s.centre_id=:cid AND s.slot_date=:d ORDER BY s.start_time''',{'cid':cid,'d':raw_date}))

def execute_booking(u, crop, sid, quantity):
    """Authoritative booking logic with tier quota and concurrency lock."""
    crop = str(crop or '').strip()[:100]
    if not crop: return None, 'Choose a crop before booking.', 400
    try: sid = int(sid)
    except (TypeError, ValueError): return None, 'Choose an available slot.', 400
    try:
        quantity = float(quantity)
    except (TypeError, ValueError):
        return None, 'Enter your quantity in tons (numbers only).', 400
    if not 0 < quantity <= 100: return None, 'Quantity must be between 0.1 and 100 tons.', 400
    tier = quantity_tier(quantity)
    capacity_col = {'High': 'cap_high', 'Mid': 'cap_mid', 'Low': 'cap_low'}[tier]
    tier_full_msg = {
        'High': 'High-priority seats (5 tons and above) are full for this slot. Choose another slot.',
        'Mid': 'Mid-priority seats (1 to 5 tons) are full for this slot. Choose another slot.',
        'Low': 'Low-priority seats (below 1 ton) are full for this slot. Choose another slot.'}[tier]
    with booking_write_guard(), engine.begin() as c:
        lock_sql = '''SELECT s.id,s.slot_date,s.start_time,s.end_time,s.capacity,
                COALESCE(s.cap_high,0) AS cap_high,COALESCE(s.cap_mid,0) AS cap_mid,COALESCE(s.cap_low,0) AS cap_low,
                c.name AS centre_name
            FROM slots s JOIN centres c ON c.id=s.centre_id WHERE s.id=:id'''
        if not IS_SQLITE: lock_sql += ' FOR UPDATE'      # serialise bookings for this slot row
        slot = c.execute(text(lock_sql), {'id': sid}).mappings().first()
        if not slot: return None, 'That slot is no longer available.', 404
        if not valid_date_str(slot['slot_date']): return None, 'That slot has an invalid date. Please choose another slot.', 400
        if slot['slot_date'] < today_str(): return None, 'That slot is in the past. Please choose a current slot.', 400
        # One active booking per farmer: also blocks accidental duplicate taps.
        dup = c.execute(text(f"SELECT id FROM bookings WHERE user_id=:uid AND status IN {ACTIVE_SQL} LIMIT 1"), {'uid': u['id']}).first()
        if dup: return None, 'You already have an active booking. Open your ticket instead.', 409
        quota = int(slot[capacity_col] or 0)
        taken = int(c.execute(text(f"SELECT COUNT(*) FROM bookings WHERE slot_id=:id AND tier=:tier AND status IN {ACTIVE_SQL}"), {'id': sid, 'tier': tier}).scalar() or 0)
        if taken >= quota: return None, tier_full_msg, 409
        # One conditional INSERT enforces the tier quota + allocates the token, so two
        # simultaneous farmers of the same tier can never overbook the same quota.
        created_at = now_iso()
        booking_created_at = lifecycle_now_iso()
        q = c.execute(text(f'''INSERT INTO bookings(user_id,slot_id,name,phone,crop,quantity_tons,tier,token,status,procurement_status,payment_status,created_at,booking_created_at)
            SELECT CAST(:uid AS INTEGER),CAST(:sid AS INTEGER),CAST(:n AS VARCHAR(200)),CAST(:p AS VARCHAR(30)),CAST(:crop AS VARCHAR(100)),
                   CAST(:qty AS REAL),CAST(:tier AS VARCHAR(10)),
                   (SELECT COALESCE(MAX(token),0)+1 FROM bookings),'Waiting','Booked','Not started',CAST(:now AS VARCHAR(50)),:booking_created_at
            WHERE (SELECT COUNT(*) FROM bookings WHERE slot_id=CAST(:sid AS INTEGER) AND tier=CAST(:tier AS VARCHAR(10)) AND status IN {ACTIVE_SQL})<CAST(:quota AS INTEGER)
            RETURNING id,token'''),
            {'uid': u['id'], 'sid': sid, 'n': u['name'], 'p': u['phone'], 'crop': crop, 'qty': quantity, 'tier': tier,
             'now': created_at, 'booking_created_at': booking_created_at, 'quota': quota})
        row = q.mappings().first()
        if not row and IS_SQLITE:
            taken2 = int(c.execute(text(f"SELECT COUNT(*) FROM bookings WHERE slot_id=:id AND tier=:tier AND status IN {ACTIVE_SQL}"), {'id': sid, 'tier': tier}).scalar() or 0)
            if taken2 >= quota: return None, tier_full_msg, 409
            nxt = int(c.execute(text('SELECT COALESCE(MAX(token),0)+1 FROM bookings')).scalar() or 1)
            c.execute(text('''INSERT INTO bookings(user_id,slot_id,name,phone,crop,quantity_tons,tier,token,status,procurement_status,payment_status,created_at,booking_created_at)
                VALUES(:uid,:sid,:n,:p,:crop,:qty,:tier,:tok,'Waiting','Booked','Not started',:now,:booking_created_at)'''),
                {'uid': u['id'], 'sid': sid, 'n': u['name'], 'p': u['phone'], 'crop': crop, 'qty': quantity,
                 'tier': tier, 'tok': nxt, 'now': created_at, 'booking_created_at': booking_created_at})
            row = {'id': c.execute(text('SELECT id FROM bookings WHERE user_id=:uid AND slot_id=:sid ORDER BY id DESC LIMIT 1'), {'uid': u['id'], 'sid': sid}).scalar(), 'token': nxt}
        if not row: return None, tier_full_msg, 409
        bid, token = int(row['id']), int(row['token'])
        queue_size = int(c.execute(text(f'SELECT COUNT(*) FROM bookings WHERE slot_id=:id AND status IN {ACTIVE_SQL}'), {'id': sid}).scalar() or 1)
    return {
        'success': True, 'booking_id': bid, 'token': token,
        'slot': f"{slot['start_time']} - {slot['end_time']}",
        'centre': slot['centre_name'], 'slot_date': slot['slot_date'],
        'tier': tier, 'quantity_tons': quantity,
        'ahead': queue_size - 1, 'position': queue_size
    }, None, 201


@app.post('/api/book')
@login_required('farmer')
def book():
    """Book with strict High/Mid/Low quota - tier comes from quantity in tons. Same quantity = same tier."""
    data = request.get_json(silent=True) or {}
    u = current_user()
    crop = str(data.get('crop', '')).strip()[:100]
    sid = data.get('slot_id')
    qty = data.get('quantity_tons', data.get('quantity', ''))
    res, err, code = execute_booking(u, crop, sid, qty)
    if err:
        return api_error(err, code)
    return jsonify(res), 201


def query_available_slots(centre_id, slot_date, quantity_tons=None):
    """Retrieve available slots filtered by tier quota for a centre and date."""
    if not valid_date_str(slot_date):
        return []
    try:
        cid = int(centre_id)
    except (TypeError, ValueError):
        return []
    slot_list = rows(f'''SELECT s.id,s.centre_id,s.slot_date,s.start_time,s.end_time,s.capacity,
            COALESCE(s.cap_high,0) AS cap_high, COALESCE(s.cap_mid,0) AS cap_mid, COALESCE(s.cap_low,0) AS cap_low,
            c.name AS centre_name,
            (SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.status IN {ACTIVE_SQL}) AS booked,
            (s.capacity-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.status IN {ACTIVE_SQL})) AS remaining,
            (COALESCE(s.cap_high,0)-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='High' AND b.status IN {ACTIVE_SQL})) AS remaining_high,
            (COALESCE(s.cap_mid,0)-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='Mid' AND b.status IN {ACTIVE_SQL})) AS remaining_mid,
            (COALESCE(s.cap_low,0)-(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.tier='Low' AND b.status IN {ACTIVE_SQL})) AS remaining_low
        FROM slots s JOIN centres c ON c.id=s.centre_id
        WHERE s.centre_id=:cid AND s.slot_date=:d ORDER BY s.start_time''', {'cid': cid, 'd': slot_date})
    if quantity_tons is None:
        return [s for s in slot_list if s.get('remaining', 0) > 0]
    tier = quantity_tier(quantity_tons)
    if not tier:
        return [s for s in slot_list if s.get('remaining', 0) > 0]
    rem_key = {'High': 'remaining_high', 'Mid': 'remaining_mid', 'Low': 'remaining_low'}[tier]
    return [s for s in slot_list if s.get(rem_key, 0) > 0]


@app.post('/api/voice/assistant')
@login_required('farmer')
def voice_assistant():
    """Structured Conversational Booking Assistant endpoint."""
    data = request.get_json(silent=True) or {}
    text_input = str(data.get('text', ''))[:400]
    state = data.get('state') if isinstance(data.get('state'), dict) else {}
    centre_id = data.get('centre_id')
    language = str(data.get('language') or session.get('lang') or 'en')
    reset = bool(data.get('reset'))

    if not centre_id:
        c_row = one("SELECT id FROM centres ORDER BY id LIMIT 1")
        centre_id = c_row['id'] if c_row else 1
    else:
        try:
            centre_id = int(centre_id)
        except (TypeError, ValueError):
            c_row = one("SELECT id FROM centres ORDER BY id LIMIT 1")
            centre_id = c_row['id'] if c_row else 1

    if data.get('action') == 'book_again':
        # Book Again: seed a NEW conversation from this farmer's own most
        # recent cancelled booking. The cancelled row is only read, never
        # modified, and the client never supplies the booking id (no spoofing).
        u = current_user()
        prev = one('''SELECT b.crop, b.quantity_tons, s.slot_date
            FROM bookings b JOIN slots s ON s.id=b.slot_id
            WHERE b.user_id=:uid AND b.status=:st
            ORDER BY b.id DESC LIMIT 1''', {'uid': u['id'], 'st': CANCELLED})
        if not prev:
            return api_error('No recently cancelled booking found to reuse.', 404)
        qty_kg = prev['quantity_tons']
        assistant = nlp.BookingAssistant(state=None,
                                         today=datetime.now().date(),
                                         language=language)
        res = assistant.start_book_again({
            'crop': prev['crop'],
            'quantity_kg': int(round((qty_kg or 0) * 1000)) if qty_kg else None,
            'date': prev['slot_date'],
        })
        return jsonify(res)

    assistant = nlp.BookingAssistant(state=None if reset else state,
                                     today=datetime.now().date(),
                                     language=language)

    if reset or data.get('action') == 'start':
        res = assistant.start()
        return jsonify(res)

    res = assistant.process_turn(text_input)

    # When slot check is requested by the state machine
    if res['conv_state'] == nlp.CHECKING_SLOTS:
        booking_date = assistant.state.get('date')
        qty_kg = assistant.state.get('quantity_kg') or 500
        qty_tons = round(qty_kg / 1000.0, 3)
        avail = query_available_slots(centre_id, booking_date, qty_tons)
        res = assistant.check_slots_with_backend(avail)

    # When booking creation is reached
    elif res['conv_state'] == nlp.BOOKING or res.get('action') == 'create_booking':
        selected_slot = assistant.state.get('selected_slot')
        crop = assistant.state.get('crop')
        qty_kg = assistant.state.get('quantity_kg')
        qty_tons = round((qty_kg or 0) / 1000.0, 3)
        if not selected_slot or not selected_slot.get('id'):
            return api_error("No slot selected for booking.", 400)
        book_res, err, code = execute_booking(current_user(), crop, selected_slot['id'], qty_tons)
        if err:
            res = assistant._response(assistant._msg('booking_failed', {'error': err}),
                                      nlp.WAITING_FOR_SLOT_SELECTION)
            res['error'] = err
            return jsonify(res), 200
        res = assistant.confirm_booking_success(book_res)

    return jsonify(res)


def queue_metrics(item):
    """Queue position inside the farmer's own centre+date+slot queue (FR-09, FR-11).

    Ordering: High tier first, then Mid, then Low; token breaks ties inside a tier.
    Unrelated centres or slots can never leak into a farmer's position.
    """
    sid=item['slot_id']; token=item['token']; status=item.get('status')
    tier=item.get('tier')
    if status in ACTIVE_STATES:
        queue_size=int(one(f'SELECT COUNT(*) n FROM bookings b WHERE b.slot_id=:sid AND b.status IN {ACTIVE_SQL}',{'sid':sid})['n'] or 1)
        serving=rows("SELECT token FROM bookings WHERE slot_id=:sid AND status='Serving' ORDER BY token LIMIT 1",{'sid':sid})
        # A waiting farmer is behind every booking still in this slot's pipeline.
        # A farmer who is already called/serving is at the counter, so nobody is ahead of them.
        if status in (CALLED,SERVING):
            ahead=0
        else:
            ahead=int(one(f'''SELECT COUNT(*) n FROM bookings b WHERE b.slot_id=:sid AND b.status IN {ACTIVE_SQL}
                AND ({TIER_SQL_ORDER} < (CASE :tier WHEN 'High' THEN 0 WHEN 'Mid' THEN 1 ELSE 2 END)
                     OR ({TIER_SQL_ORDER} = (CASE :tier WHEN 'High' THEN 0 WHEN 'Mid' THEN 1 ELSE 2 END) AND b.token<:token))''',
                      {'sid':sid,'token':token,'tier':tier or 'Low'})['n'] or 0)
    else:
        ahead=0; queue_size=1; serving=[]
    item['ahead']=ahead
    item['position']=ahead+1
    item['queue_size']=queue_size
    item['now_serving_token']=serving[0]['token'] if serving else None
    return item


def booking_row_for_user(uid):
    r=rows('''SELECT b.id,b.user_id,b.slot_id,b.name,b.phone,b.crop,b.quantity_tons,b.tier,b.token,b.status,b.procurement_status,b.created_at,
            s.slot_date,s.start_time,s.end_time,s.capacity,s.centre_id,c.name AS centre
        FROM bookings b JOIN slots s ON s.id=b.slot_id JOIN centres c ON c.id=s.centre_id
        WHERE b.user_id=:uid ORDER BY b.id DESC LIMIT 1''',{'uid':uid})
    return queue_metrics(r[0]) if r else None

@app.get('/api/my-booking')
@login_required('farmer')
def my_booking():
    item=booking_row_for_user(current_user()['id']); return jsonify(item or {'booking':None})

@app.get('/api/my-bookings')
@login_required('farmer')
def my_bookings():
    """The farmer's own booking history (newest first), for Book Again + history view.

    Always scoped to the session user; another farmer's rows can never appear.
    The cancelled row stays untouched - a Book Again flow just adds a new row.
    """
    u=current_user()
    items=rows('''SELECT b.id,b.token,b.status,b.crop,b.quantity_tons,b.tier,b.created_at,
            s.slot_date,s.start_time,s.centre_id,c.name AS centre
        FROM bookings b JOIN slots s ON s.id=b.slot_id JOIN centres c ON c.id=s.centre_id
        WHERE b.user_id=:uid ORDER BY b.id DESC LIMIT 20''',{'uid':u['id']})
    return jsonify(success=True, bookings=items)

@app.get('/api/farmer/status')
@login_required('farmer')
def farmer_status_api():
    """Dedicated status API for /farmer/status (refresh-safe, ownership-checked).

    Identifier comes from the server session (current_user), never from JS
    memory. Optional ?booking_id= is honoured ONLY when that booking belongs
    to the logged-in farmer; otherwise 403 — never another farmer's data.
    Reuses the existing booking_row_for_user()/queue_metrics() response shape
    ({id,token,status,...,ahead,position,queue_size}) so no duplicate logic.
    """
    u = current_user()
    raw = (request.args.get('booking_id') or '').strip()
    if not raw:
        item = booking_row_for_user(u['id'])
        return jsonify(success=True, booking=item)
    try:
        bid = int(raw)
    except (TypeError, ValueError):
        return api_error('Invalid booking reference.', 400)
    item = rows('''SELECT b.id,b.user_id,b.slot_id,b.name,b.phone,b.crop,b.quantity_tons,b.tier,b.token,b.status,b.procurement_status,b.created_at,
            s.slot_date,s.start_time,s.end_time,s.capacity,s.centre_id,c.name AS centre
        FROM bookings b JOIN slots s ON s.id=b.slot_id JOIN centres c ON c.id=s.centre_id
        WHERE b.id=:id''', {'id': bid})
    if not item:
        return api_error('Booking not found.', 404)
    if item[0]['user_id'] != u['id']:
        return api_error('Access denied.', 403)
    return jsonify(success=True, booking=queue_metrics(item[0]))

@app.post('/api/nlp/parse')
@login_required('farmer')
def nlp_parse():
    """Understand one farmer utterance: intent, entities and the next dialogue step.

    Pure language understanding (nlp.py). It NEVER decides whether a slot is
    free - the client still asks GET /api/slots and books through POST /api/book.
    The conversation state round-trips through the browser, so this endpoint
    stays stateless; the state itself is language-independent, so the same step
    renders in English, Telugu or Hindi (requirement 9).
    """
    data = request.get_json(silent=True) or {}
    text = str(data.get('text') or '')[:400]
    state = data.get('state') if isinstance(data.get('state'), dict) else {}
    conv = nlp.Conversation(state=state)
    if data.get('reset'):
        conv.reset()
    result = conv.step(text)
    return jsonify(success=True, intent=result['intent'],
                   intent_confidence=result['intent_confidence'],
                   entities=result['entities'], confidence=result['confidence'],
                   corrections=result['corrections'], ambiguous=result['ambiguous'],
                   understood=result['understood'], slots_needed=result['slots_needed'],
                   next=result['next'], state=result['state'])

@app.get('/api/nlp/suggest')
@login_required('farmer')
def nlp_suggest():
    """Context-aware word/phrase prediction for the typed box (requirement 12).

    `expected` is the question being answered and crop/date are what is already
    known, so the completer finishes the sentence, not just the word - and it
    returns nothing when the fragment is too short to guess safely (req. 1).
    """
    context = {}
    for key in ('crop', 'date'):
        value = (request.args.get(key) or '').strip()[:40]
        if value:
            context[key] = value
    expected = (request.args.get('expected') or '').strip() or None
    suggestions = nlp.suggest(request.args.get('q', ''), expected=expected,
                              state=context, limit=8)
    return jsonify(success=True, suggestions=suggestions)

@app.get('/api/notifications')
@app.get('/api/my-notifications')
@login_required('farmer')
def notifications():
    """Only the logged-in farmer's own notifications (FR-09, FR-13)."""
    u=current_user()
    return jsonify(rows('''SELECT n.id,n.booking_id,n.message,n.kind,n.is_read,n.created_at,b.token AS booking_token
        FROM notifications n LEFT JOIN bookings b ON b.id=n.booking_id
        WHERE n.user_id=:uid ORDER BY n.id DESC LIMIT 30''',{'uid':u['id']}))

@app.post('/api/notifications/read')
@app.post('/api/my-notifications/read')
@login_required('farmer')
def notifications_read():
    with engine.begin() as c: c.execute(text('UPDATE notifications SET is_read=1 WHERE user_id=:uid'),{'uid':current_user()['id']})
    return jsonify(success=True)

@app.post('/api/notifications/<int:nid>/read')
@login_required('farmer')
def notification_read(nid):
    """Mark one notification as read - always scoped to the logged-in farmer."""
    with engine.begin() as c:
        res=c.execute(text('UPDATE notifications SET is_read=1 WHERE id=:id AND user_id=:uid'),{'id':nid,'uid':current_user()['id']})
    if res.rowcount==0: return api_error('Notification not found.',404)
    return jsonify(success=True)

@app.get('/api/bookings')
@login_required('admin')
def bookings():
    """Admin queue view (FR-16). Optional ?date=&centre_id=&status=&tier= filters."""
    where=[]; params={}
    raw_date=(request.args.get('date') or '').strip()
    if raw_date:
        if not valid_date_str(raw_date): return api_error('Invalid date filter.',400)
        where.append('s.slot_date=:d'); params['d']=raw_date
    raw_centre=(request.args.get('centre_id') or '').strip()
    if raw_centre:
        try: params['cid']=int(raw_centre)
        except ValueError: return api_error('Invalid centre filter.',400)
        where.append('s.centre_id=:cid')
    raw_status=(request.args.get('status') or '').strip()
    if raw_status:
        if raw_status not in ALL_STATES: return api_error('Invalid status filter.',400)
        where.append('b.status=:st'); params['st']=raw_status
    raw_tier=(request.args.get('tier') or '').strip()
    if raw_tier:
        if raw_tier not in (TIER_HIGH, TIER_MID, TIER_LOW): return api_error('Invalid tier filter.',400)
        where.append('b.tier=:tier'); params['tier']=raw_tier
    sql='''SELECT b.id,b.name,b.phone,b.crop,b.quantity_tons,b.tier,b.token,b.status,b.procurement_status,b.payment_status,b.created_at,
            s.id AS slot_id,s.slot_date,s.start_time,s.end_time,s.capacity,c.name AS centre,
            (s.capacity-(SELECT COUNT(*) FROM bookings b2 WHERE b2.slot_id=s.id AND b2.status IN ('Waiting','Called','Serving'))) AS remaining
        FROM bookings b JOIN slots s ON s.id=b.slot_id JOIN centres c ON c.id=s.centre_id'''
    if where: sql+=' WHERE '+' AND '.join(where)
    sql+=' ORDER BY s.slot_date,s.start_time,' + TIER_SQL_ORDER + ',b.token'
    return jsonify(rows(sql,params))

@app.get('/api/stats')
@login_required('admin')
def stats():
    """Dashboard counters (FR-16) - bookings by status plus slot availability."""
    r=rows("""SELECT COUNT(*) total,
        SUM(CASE WHEN status='Waiting' THEN 1 ELSE 0 END) waiting,
        SUM(CASE WHEN status='Called' THEN 1 ELSE 0 END) called,
        SUM(CASE WHEN status='Serving' THEN 1 ELSE 0 END) serving,
        SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) completed,
        SUM(CASE WHEN status='Cancelled' THEN 1 ELSE 0 END) cancelled
        FROM bookings""")[0]
    payload={k:int(v or 0) for k,v in r.items()}
    tier_counts=rows(f"""SELECT
        SUM(CASE WHEN tier='High' AND status IN {ACTIVE_SQL} THEN 1 ELSE 0 END) high,
        SUM(CASE WHEN tier='Mid' AND status IN {ACTIVE_SQL} THEN 1 ELSE 0 END) mid,
        SUM(CASE WHEN tier='Low' AND status IN {ACTIVE_SQL} THEN 1 ELSE 0 END) low
        FROM bookings""")[0]
    payload.update({('tier_'+k):int(v or 0) for k,v in tier_counts.items()})
    slots=rows(f"""SELECT COUNT(*) slots_total,
        SUM(CASE WHEN active<s.capacity THEN 1 ELSE 0 END) slots_available,
        SUM(CASE WHEN active>=s.capacity THEN 1 ELSE 0 END) slots_full
        FROM (SELECT s.id,s.capacity,(SELECT COUNT(*) FROM bookings b WHERE b.slot_id=s.id AND b.status IN {ACTIVE_SQL}) active
              FROM slots s WHERE s.slot_date>=:today) s""",{'today':today_str()})[0]
    payload.update({k:int(v or 0) for k,v in slots.items()})
    return jsonify(payload)


def booking_time_metrics(item):
    """Return elapsed lifecycle durations in seconds when both events exist."""
    def parse(value):
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except (TypeError, ValueError):
            return None

    def elapsed(start, end):
        start_dt, end_dt = parse(item.get(start)), parse(item.get(end))
        if not start_dt or not end_dt:
            return None
        # Old rows use timezone-naive timestamps; compare naive values as stored.
        if (start_dt.tzinfo is None) != (end_dt.tzinfo is None):
            start_dt, end_dt = start_dt.replace(tzinfo=None), end_dt.replace(tzinfo=None)
        seconds = (end_dt - start_dt).total_seconds()
        return round(seconds, 3) if seconds >= 0 else None

    return {
        'waiting_time_seconds': elapsed('booking_created_at', 'called_at'),
        'service_time_seconds': elapsed('served_at', 'completed_at'),
        'total_processing_time_seconds': elapsed('booking_created_at', 'completed_at'),
    }


ANALYTICS_BOOKING_SELECT = '''SELECT b.id,b.crop,b.status,b.booking_created_at,b.called_at,b.served_at,
        b.completed_at,b.cancelled_at,s.slot_date,s.start_time,s.end_time,c.name AS centre
    FROM bookings b JOIN slots s ON s.id=b.slot_id JOIN centres c ON c.id=s.centre_id'''


@app.get('/api/analytics')
@login_required('admin')
def analytics():
    """Read-only historical aggregates for the admin analytics panel."""
    booking_rows = rows(ANALYTICS_BOOKING_SELECT)
    enriched = [(item, booking_time_metrics(item)) for item in booking_rows]

    def average(metric):
        values = [durations[metric] for _, durations in enriched if durations[metric] is not None]
        return round(sum(values) / len(values), 3) if values else None

    by_crop, by_centre, by_slot = {}, {}, {}
    for item, _ in enriched:
        by_crop[item['crop']] = by_crop.get(item['crop'], 0) + 1
        by_centre[item['centre']] = by_centre.get(item['centre'], 0) + 1
        slot_key = f"{item['centre']} • {item['slot_date']} {item['start_time']}–{item['end_time']}"
        by_slot[slot_key] = by_slot.get(slot_key, 0) + 1
    return jsonify({
        'total_bookings': len(booking_rows),
        'completed_bookings': sum(1 for item in booking_rows if item['status'] == COMPLETED),
        'average_waiting_time_seconds': average('waiting_time_seconds'),
        'average_service_time_seconds': average('service_time_seconds'),
        'average_total_processing_time_seconds': average('total_processing_time_seconds'),
        'waiting_time_sample_count': sum(1 for _, m in enriched if m['waiting_time_seconds'] is not None),
        'service_time_sample_count': sum(1 for _, m in enriched if m['service_time_seconds'] is not None),
        'bookings_by_crop': [{'crop': k, 'count': v} for k, v in sorted(by_crop.items(), key=lambda x: (-x[1], x[0]))],
        'bookings_by_centre': [{'centre': k, 'count': v} for k, v in sorted(by_centre.items(), key=lambda x: (-x[1], x[0]))],
        'bookings_by_slot': [{'slot': k, 'count': v} for k, v in sorted(by_slot.items(), key=lambda x: (-x[1], x[0]))[:100]],
    })


@app.get('/api/analytics/bookings/<int:bid>')
@login_required('admin')
def analytics_booking(bid):
    """Read-only lifecycle timestamps and derived durations for one booking."""
    item = one(ANALYTICS_BOOKING_SELECT + ' WHERE b.id=:id', {'id': bid})
    if not item:
        return api_error('Booking not found.', 404)
    item.update(booking_time_metrics(item))
    for key in ('booking_created_at', 'called_at', 'served_at', 'completed_at', 'cancelled_at'):
        if isinstance(item.get(key), datetime):
            item[key] = item[key].isoformat()
    return jsonify(item)

# --- booking status machine (PRD section 10) --------------------------------
CALL_MESSAGE='Token #{token} is being called. Please proceed to the procurement counter.'
SERVE_MESSAGE='Token #{token} is now being served. Please proceed to the counter.'
COMPLETE_MESSAGE='Token #{token} procurement is completed. Thank you.'
CANCEL_MESSAGE='Your token #{token} has been cancelled.'


def transition_booking(bid, target, message_template=None, kind='info'):
    """Validate and apply one status change, notifying only that booking's farmer.

    Returns (result, error_message, http_status). Invalid transitions are refused
    so the queue can never jump backwards from a finished state.
    """
    try:
        with booking_write_guard(), engine.begin() as c:
            row=c.execute(text('SELECT id,user_id,status,token FROM bookings WHERE id=:id'),{'id':bid}).mappings().first()
            if not row: return None,'Booking not found.',404
            current=row['status']
            if target!=current and target not in VALID_TRANSITIONS.get(current,set()):
                return None,f'A booking that is {current} cannot be moved to {target}.',409
            if target!=current:
                event_column = {CALLED: 'called_at', SERVING: 'served_at',
                                COMPLETED: 'completed_at', CANCELLED: 'cancelled_at'}.get(target)
                event_sql = f', {event_column}=:event_at' if event_column else ''
                params = {'s':target, 'p':PROCUREMENT_STATUS.get(target,'Booked'), 'id':bid}
                if event_column:
                    params['event_at'] = lifecycle_now_iso()
                c.execute(text(f'UPDATE bookings SET status=:s,procurement_status=:p{event_sql} WHERE id=:id'), params)
                if message_template:
                    c.execute(text('INSERT INTO notifications(user_id,booking_id,message,kind,is_read,created_at) VALUES(:uid,:bid,:msg,:kind,0,:now)'),
                              {'uid':row['user_id'],'bid':bid,'msg':message_template.format(token=row['token']),'kind':kind,'now':now_iso()})
    except SQLAlchemyError:
        # engine.begin() already rolled back. Log loudly so "dead after 2 calls"
        # is diagnosable instead of a silent hang on the frontend.
        app.logger.exception('transition_booking failed for booking %s -> %s', bid, target)
        return None,'Database is busy. Please wait a moment and try again.',503
    return {'id':bid,'token':row['token'],'status':target,'previous_status':current},None,200


def transition_response(bid,target,template,kind='info'):
    result,error,code=transition_booking(bid,target,template,kind)
    if error: return api_error(error,code)
    return jsonify(success=True,booking=result,status=result['status'])


@app.post('/api/call/<int:bid>')
@login_required('admin')
def call_farmer(bid):
    """FR-12: call one specific farmer. No other farmer's booking is touched."""
    return transition_response(bid,CALLED,CALL_MESSAGE,'call')

@app.post('/api/serve/<int:bid>')
@login_required('admin')
def serve_farmer(bid):
    """Farmer reached the counter: CALLED -> SERVING."""
    return transition_response(bid,SERVING,SERVE_MESSAGE,'serve')

@app.post('/api/complete/<int:bid>')
@login_required('admin')
def complete_farmer(bid):
    """Procurement finished: SERVING -> COMPLETED."""
    return transition_response(bid,COMPLETED,COMPLETE_MESSAGE,'info')

@app.post('/api/cancel/<int:bid>')
@login_required('admin')
def cancel_booking(bid):
    """Centre-side cancellation (no-show) for a booking that is not finished yet."""
    return transition_response(bid,CANCELLED,CANCEL_MESSAGE,'info')

@app.post('/api/my-booking/cancel')
@login_required('farmer')
def cancel_my_booking():
    """A farmer may cancel only their own booking while it is still WAITING."""
    u=current_user()
    row=one('SELECT id,status,token FROM bookings WHERE user_id=:uid ORDER BY id DESC LIMIT 1',{'uid':u['id']})
    if not row: return api_error('No booking found.',404)
    if row['status']!=WAITING: return api_error('Only a booking that is still waiting can be cancelled.',409)
    return transition_response(row['id'],CANCELLED,'Your token #{token} has been cancelled. You can book a new slot.','info')

@app.post('/api/advance')
@login_required('admin')
def advance():
    """Close the booking currently being served and call the next waiting farmer (High tier first)."""
    completed=called=None
    serving=one("SELECT b.id FROM bookings b JOIN slots s ON s.id=b.slot_id WHERE b.status='Serving' ORDER BY s.slot_date,s.start_time,b.token LIMIT 1")
    if serving:
        result,_,_=transition_booking(serving['id'],COMPLETED,COMPLETE_MESSAGE,'info')
        completed=result['id'] if result else None
    nxt=one("SELECT b.id FROM bookings b JOIN slots s ON s.id=b.slot_id WHERE b.status='Waiting' ORDER BY s.slot_date,s.start_time,"+TIER_SQL_ORDER+",b.token LIMIT 1")
    if nxt:
        result,_,_=transition_booking(nxt['id'],CALLED,CALL_MESSAGE,'call')
        called=result['id'] if result else None
    return jsonify(success=True,completed=completed,called=called)

@app.post('/api/reset')
@login_required('admin')
def reset():
    with engine.begin() as c: c.execute(text('DELETE FROM notifications')); c.execute(text('DELETE FROM bookings')); c.execute(text('DELETE FROM slots'))
    return jsonify(success=True)

@app.get('/api/booking/<int:bid>')
@login_required('farmer')
def booking(bid):
    """Farmer polling endpoint - a farmer can only ever open their own booking (NFR-01)."""
    u=current_user()
    item=rows('''SELECT b.id,b.user_id,b.slot_id,b.name,b.phone,b.crop,b.quantity_tons,b.tier,b.token,b.status,b.procurement_status,b.created_at,
            s.slot_date,s.start_time,s.end_time,s.capacity,s.centre_id,c.name AS centre
        FROM bookings b JOIN slots s ON s.id=b.slot_id JOIN centres c ON c.id=s.centre_id
        WHERE b.id=:id AND b.user_id=:uid''',{'id':bid,'uid':u['id']})
    if not item: return api_error('Booking not found.',404)
    return jsonify(queue_metrics(item[0]))

@app.after_request
def harden_response(response):
    """No caching for live queue data, plus a few baseline security headers."""
    if request.path.startswith('/api/') or request.path == '/health':
        response.headers['Cache-Control'] = 'no-store'
    origin = request.headers.get('Origin')
    if origin and _is_allowed_origin(origin):
        # Exact origin (never '*') + credentials, so the session cookie keeps
        # working for same-site local dev (e.g. Live Server :5500 -> :5000).
        # Same-origin page loads send no Origin on GET and are unaffected.
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Vary'] = 'Origin'
        if request.method == 'OPTIONS':
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Max-Age'] = '600'
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'same-origin')
    return response


@app.errorhandler(HTTPException)
def handle_http_error(error):
    """Keep API failures machine readable instead of returning an HTML error page."""
    if request.path.startswith('/api/') or request.path == '/health':
        return api_error(error.description or error.name, error.code or 500)
    return error


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Last resort: log the real cause, return a safe message to the user."""
    app.logger.exception('Unhandled error on %s %s', request.method, request.path)
    if request.path.startswith('/api/') or request.path == '/health':
        return api_error('Server error. Please try again.', 500)
    return 'Server error. Please try again.', 500

db_init()
if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=os.environ.get('FLASK_DEBUG','0')=='1',threaded=True)
