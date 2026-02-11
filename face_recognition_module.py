# ==============================
# Module de compatibilité pour main.py
# ==============================

from main_face_module import FaceRecognitionModule  # ton fichier avec la classe principale

# Instance globale (initialisée à la demande)
_system = None


def _get_system():
    global _system
    if _system is None:
        _system = FaceRecognitionModule()
    return _system

# ==============================
# Fonctions globales attendues par main.py
# ==============================

def verifier_sortie():
    """Détecte un visage pour enregistrer une sortie"""
    system = _get_system()
    frame = system.capture_face()
    encoding = system.get_face_encoding(frame)
    if encoding is None:
        print("[ERREUR] Visage non détecté")
        return
    if not system.identify_exit(encoding):
        print("[ECHEC] Aucune correspondance – validation manuelle requise")

def detecter_visage():
    """Capture caméra et affiche le nombre de visages détectés."""
    system = _get_system()
    frame = system.capture_face()
    if frame is None:
        print("[ERREUR] Impossible de capturer l'image caméra")
        return []
    detections = system.detect_faces(frame)
    print(f"[INFO] {len(detections)} visage(s) detecte(s)")
    return detections

def sortie_manuelle():
    """Traitement manuel des sorties (placeholder)"""
    print("[INFO] Sortie manuelle – non implémentée pour le moment")
