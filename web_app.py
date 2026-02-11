import base64
import logging
import os
from io import BytesIO
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, session, url_for
from PIL import Image

from db_module import (
    create_user,
    ensure_default_user,
    get_visiteurs_du_jour,
    list_users,
    set_user_active,
    update_user_password,
    verify_user_credentials,
)
from main_face_module import FaceRecognitionModule


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("ACCESS_APP_SECRET", "change-me-access-secret")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
system = FaceRecognitionModule()
ensure_default_user(
    username=os.environ.get("ACCESS_APP_ADMIN_USER", "admin"),
    password=os.environ.get("ACCESS_APP_ADMIN_PASS", "admin123"),
)


def _auth_payload():
    return {
        "id": session.get("user_id"),
        "username": session.get("username"),
        "full_name": session.get("full_name"),
        "role": session.get("role"),
    }


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"ok": False, "message": "Authentification requise"}), 401
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"ok": False, "message": "Authentification requise"}), 401
        if (session.get("role") or "").upper() != "ADMIN":
            return jsonify({"ok": False, "message": "Acces admin requis"}), 403
        return fn(*args, **kwargs)
    return wrapper


def data_url_to_bgr(data_url):
    if not data_url or "," not in data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def data_url_to_cv2(data_url):
    if not data_url or "," not in data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(raw)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not username or not password:
        return jsonify({"ok": False, "message": "Identifiants requis"}), 400

    user = verify_user_credentials(username, password)
    if not user:
        return jsonify({"ok": False, "message": "Nom d'utilisateur ou mot de passe invalide"}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["full_name"] = user.get("full_name")
    session["role"] = user.get("role")
    return jsonify({"ok": True, "message": "Connexion réussie", "user": _auth_payload()})


@app.post("/api/logout")
@login_required
def logout():
    session.clear()
    return jsonify({"ok": True, "message": "Déconnexion réussie"})


@app.get("/api/me")
def me():
    if not session.get("user_id"):
        return jsonify({"ok": False, "authenticated": False}), 401
    return jsonify({"ok": True, "authenticated": True, "user": _auth_payload()})


@app.post("/api/admin/users")
@admin_required
def admin_create_user():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    full_name = (payload.get("full_name") or "").strip() or None
    role = (payload.get("role") or "USER").strip().upper()

    if not username or not password:
        return jsonify({"ok": False, "message": "Nom d'utilisateur et mot de passe requis"}), 400
    if role not in {"ADMIN", "USER"}:
        return jsonify({"ok": False, "message": "Role invalide (ADMIN ou USER)"}), 400

    created = create_user(username=username, password=password, full_name=full_name, role=role)
    if not created:
        return jsonify({"ok": False, "message": "Utilisateur deja existant ou donnees invalides"}), 409
    return jsonify({"ok": True, "message": "Utilisateur cree", "user": {"username": username, "role": role, "full_name": full_name}})


@app.get("/api/admin/users")
@admin_required
def admin_list_users():
    rows = list_users()
    users = []
    for user_id, username, full_name, role, is_active, created_at in rows:
        users.append(
            {
                "id": user_id,
                "username": username,
                "full_name": full_name,
                "role": role,
                "is_active": bool(is_active),
                "created_at": created_at,
            }
        )
    return jsonify({"ok": True, "count": len(users), "users": users})


@app.patch("/api/admin/users/<int:user_id>/status")
@admin_required
def admin_set_user_status(user_id):
    payload = request.get_json(silent=True) or {}
    is_active = payload.get("is_active")
    if not isinstance(is_active, bool):
        return jsonify({"ok": False, "message": "Champ is_active requis (true/false)"}), 400
    if user_id == session.get("user_id") and not is_active:
        return jsonify({"ok": False, "message": "Impossible de se desactiver soi-meme"}), 400
    updated = set_user_active(user_id, is_active)
    if not updated:
        return jsonify({"ok": False, "message": "Utilisateur introuvable"}), 404
    return jsonify({"ok": True, "message": "Statut utilisateur mis a jour"})


@app.patch("/api/admin/users/<int:user_id>/password")
@admin_required
def admin_update_user_password(user_id):
    payload = request.get_json(silent=True) or {}
    new_password = payload.get("password") or ""
    if len(new_password) < 4:
        return jsonify({"ok": False, "message": "Mot de passe trop court (min 4 caracteres)"}), 400
    updated = update_user_password(user_id, new_password)
    if not updated:
        return jsonify({"ok": False, "message": "Utilisateur introuvable"}), 404
    return jsonify({"ok": True, "message": "Mot de passe mis a jour"})


@app.post("/api/quit")
@login_required
def quit_app():
    shutdown_fn = request.environ.get("werkzeug.server.shutdown")
    if shutdown_fn is None:
        return jsonify({"ok": False, "message": "Arrêt non disponible"}), 400
    shutdown_fn()
    return jsonify({"ok": True, "message": "Application arrêtée"})


@app.get("/api/files")
@login_required
def list_files():
    date_filter = (request.args.get("date") or "").strip()
    date_prefix = None
    if date_filter:
        try:
            date_prefix = datetime.strptime(date_filter, "%Y-%m-%d").strftime("%Y%m%d")
        except ValueError:
            return jsonify({"ok": False, "message": "Format date invalide (YYYY-MM-DD)"}), 400

    photos_dir = system.photos_dir
    os.makedirs(photos_dir, exist_ok=True)
    files = []
    for name in os.listdir(photos_dir):
        path = os.path.join(photos_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
            continue
        if date_prefix and date_prefix not in name:
            continue
        stat = os.stat(path)
        files.append(
            {
                "name": name,
                "size": stat.st_size,
                "modified": int(stat.st_mtime),
                "url": url_for("get_file", filename=name),
            }
        )

    files.sort(key=lambda item: item["modified"], reverse=True)
    return jsonify({"ok": True, "files": files})


@app.get("/api/files/<path:filename>")
@login_required
def get_file(filename):
    photos_dir = os.path.abspath(system.photos_dir)
    safe_name = os.path.basename(filename)
    return send_from_directory(photos_dir, safe_name, as_attachment=False)


def _photo_url_from_path(photo_path):
    if not photo_path:
        return None
    filename = os.path.basename(photo_path)
    if not filename:
        return None
    return url_for("get_file", filename=filename)


@app.get("/api/visits")
@login_required
def list_visits():
    date_filter = (request.args.get("date") or "").strip()
    if not date_filter:
        return jsonify({"ok": False, "message": "Date requise (YYYY-MM-DD)"}), 400

    try:
        datetime.strptime(date_filter, "%Y-%m-%d")
    except ValueError:
        return jsonify({"ok": False, "message": "Format date invalide (YYYY-MM-DD)"}), 400

    rows = get_visiteurs_du_jour(date_filter)
    visits = []
    for visit_id, nom, prenom, numero_carte, photo_entree, photo_cni, statut, heure_entree, heure_sortie in rows:
        visits.append(
            {
                "id": visit_id,
                "nom": nom,
                "prenom": prenom,
                "numero_carte": numero_carte,
                "photo_entree": photo_entree,
                "photo_entree_url": _photo_url_from_path(photo_entree),
                "photo_cni": photo_cni,
                "photo_cni_url": _photo_url_from_path(photo_cni),
                "statut": statut,
                "heure_entree": heure_entree,
                "heure_sortie": heure_sortie,
            }
        )
    return jsonify({"ok": True, "date": date_filter, "count": len(visits), "visits": visits})


@app.post("/api/entry")
@login_required
def api_entry():
    payload = request.get_json(silent=True) or {}
    nom = (payload.get("nom") or "").strip()
    prenom = (payload.get("prenom") or "").strip()
    cni = (payload.get("cni") or "").strip()
    frame_data = payload.get("frame_data")
    cni_data = payload.get("cni_data")

    if not nom or not prenom or not cni:
        return jsonify({"ok": False, "message": "Nom, prénom et CNI sont requis"}), 400

    frame = data_url_to_bgr(frame_data)
    if frame is None:
        return jsonify({"ok": False, "message": "Image webcam invalide"}), 400

    cni_img = data_url_to_cv2(cni_data) if cni_data else None
    encoding = system.get_face_encoding(frame)
    visit_id = system.register_person(nom, prenom, cni, encoding, cni_img, frame)
    if visit_id is None:
        return jsonify({"ok": False, "message": "Échec enregistrement entrée"}), 500

    if encoding is None:
        return jsonify(
            {
                "ok": True,
                "message": f"Entrée enregistrée sans reconnaissance faciale (visite #{visit_id})",
                "visit_id": visit_id,
                "debug": system.get_last_debug(),
            }
        )
    return jsonify({"ok": True, "message": f"Entrée enregistrée (visite #{visit_id})", "visit_id": visit_id})


@app.post("/api/exit")
@login_required
def api_exit():
    payload = request.get_json(silent=True) or {}
    frame_data = payload.get("frame_data")
    frame = data_url_to_bgr(frame_data)
    if frame is None:
        return jsonify({"ok": False, "message": "Image webcam invalide"}), 400

    encoding = system.get_face_encoding(frame)
    if encoding is None:
        return jsonify({"ok": False, "message": "Visage non détecté", "debug": system.get_last_debug()}), 400

    matched = system.identify_exit(encoding)
    if not matched:
        return jsonify({"ok": False, "message": "Aucune correspondance trouvée"}), 404
    return jsonify({"ok": True, "message": "Sortie enregistrée"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
