import sqlite3
import json
import os
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

DATA_DIR = os.environ.get("ACCESS_APP_DATA_DIR", "").strip()
if DATA_DIR:
    os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "registre.db") if DATA_DIR else "registre.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS visites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT,
            prenom TEXT,
            numero_carte TEXT,
            date TEXT,
            heure_entree TEXT,
            heure_sortie TEXT,
            photo_entree TEXT,
            photo_cni TEXT,
            face_encoding TEXT,
            statut TEXT
        )
    ''')

    # Migration légère pour les bases déjà existantes.
    c.execute("PRAGMA table_info(visites)")
    columns = {row[1] for row in c.fetchall()}
    if "face_encoding" not in columns:
        c.execute("ALTER TABLE visites ADD COLUMN face_encoding TEXT")
    if "photo_cni" not in columns:
        c.execute("ALTER TABLE visites ADD COLUMN photo_cni TEXT")

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL DEFAULT 'USER',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
    ''')

    c.execute("PRAGMA table_info(users)")
    user_columns = {row[1] for row in c.fetchall()}
    if "is_active" not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")

    conn.commit()
    conn.close()

def enregistrer_entree(nom, prenom, numero_carte, photo_path, photo_cni_path, date, heure, face_encoding=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    face_encoding_json = json.dumps(face_encoding) if face_encoding is not None else None
    c.execute('''
        INSERT INTO visites (nom, prenom, numero_carte, photo_entree, photo_cni, date, heure_entree, face_encoding, statut)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (nom, prenom, numero_carte, photo_path, photo_cni_path, date, heure, face_encoding_json, 'EN_COURS'))
    visit_id = c.lastrowid
    conn.commit()
    conn.close()
    return visit_id

def enregistrer_sortie(id_visite, heure_sortie, mode='AUTO'):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE visites
        SET statut = ?, heure_sortie = ?
        WHERE id = ?
    ''', (f'SORTI_{mode}', heure_sortie, id_visite))
    conn.commit()
    conn.close()

def get_entrees_du_jour(date):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, nom, prenom, photo_entree, face_encoding
        FROM visites
        WHERE date = ? AND statut = 'EN_COURS'
    ''', (date,))
    result = c.fetchall()
    conn.close()
    return result

def get_visiteurs_du_jour(date):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, nom, prenom, numero_carte, photo_entree, photo_cni, statut, heure_entree, heure_sortie
        FROM visites
        WHERE date = ?
    ''', (date,))
    result = c.fetchall()
    conn.close()
    return result


def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''
        SELECT id, username, password_hash, full_name, role, is_active, created_at
        FROM users
        WHERE username = ?
        ''',
        (username,),
    )
    row = c.fetchone()
    conn.close()
    return row


def create_user(username, password, full_name=None, role="USER"):
    if not username or not password:
        return False
    password_hash = generate_password_hash(password)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            '''
            INSERT INTO users (username, password_hash, full_name, role, is_active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            ''',
            (username.strip(), password_hash, full_name, role, created_at),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user_credentials(username, password):
    row = get_user_by_username((username or "").strip())
    if not row:
        return None
    user_id, db_username, password_hash, full_name, role, is_active, created_at = row
    is_active = bool(is_active)
    if not is_active:
        return None
    if not check_password_hash(password_hash, password or ""):
        return None
    return {
        "id": user_id,
        "username": db_username,
        "full_name": full_name,
        "role": role,
        "created_at": created_at,
        "is_active": is_active,
    }


def ensure_default_user(username="admin", password="admin123", full_name="Administrateur", role="ADMIN"):
    existing = get_user_by_username(username)
    if existing:
        return False
    return create_user(username=username, password=password, full_name=full_name, role=role)


def list_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''
        SELECT id, username, full_name, role, is_active, created_at
        FROM users
        ORDER BY id ASC
        '''
    )
    rows = c.fetchall()
    conn.close()
    return rows


def set_user_active(user_id, is_active):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET is_active = ? WHERE id = ?",
        (1 if is_active else 0, user_id),
    )
    changed = c.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def update_user_password(user_id, new_password):
    if not new_password:
        return False
    password_hash = generate_password_hash(new_password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id),
    )
    changed = c.rowcount
    conn.commit()
    conn.close()
    return changed > 0
