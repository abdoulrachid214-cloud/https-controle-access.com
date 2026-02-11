import os
import json
import logging
from datetime import datetime

import cv2
import numpy as np
import face_recognition

from db_module import (
    enregistrer_entree,
    enregistrer_sortie,
    get_entrees_du_jour,
    init_db,
)


class FaceRecognitionModule:
    def __init__(self, photos_dir="photos", tolerance=0.45):
        data_dir = os.environ.get("ACCESS_APP_DATA_DIR", "").strip()
        if data_dir:
            photos_dir = os.path.join(data_dir, "photos")
        self.photos_dir = photos_dir
        self.tolerance = tolerance
        self._active_visits = []
        self.last_debug = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.haar = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        init_db()

    def scan_cni(self, cni_path):
        if not cni_path or not os.path.exists(cni_path):
            return None
        return cv2.imread(cni_path)

    def capture_face(self, camera_index=0, attempts=5):
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            self.logger.error("Caméra inaccessible")
            return None

        frame = None
        try:
            for _ in range(max(1, attempts)):
                ret, candidate = cam.read()
                if ret and candidate is not None:
                    frame = candidate
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()

        if frame is None:
            self.logger.error("Impossible de capturer une image caméra")
        return frame

    def get_face_encoding(self, frame):
        self.last_debug = {}
        if frame is None:
            self.last_debug = {"error": "frame_none"}
            return None

        rgb_frame = self._to_rgb_frame(frame)
        if rgb_frame is None:
            return None

        self.last_debug = {
            "shape": list(rgb_frame.shape),
            "brightness": float(np.mean(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY))),
            "candidates": [],
        }

        for idx, candidate in enumerate(self._frame_candidates(rgb_frame), start=1):
            face_locations, detector = self._find_locations(candidate)
            cand_debug = {
                "candidate_index": idx,
                "shape": list(candidate.shape),
                "detector": detector,
                "faces_found": len(face_locations),
            }
            self.last_debug["candidates"].append(cand_debug)
            if face_locations:
                try:
                    encodings = face_recognition.face_encodings(candidate, face_locations)
                    if encodings:
                        self.last_debug["selected_candidate"] = idx
                        self.last_debug["status"] = "ok"
                        return encodings[0]
                    # Si des boxes existent mais encodage vide, on tente sans boxes.
                    encodings = face_recognition.face_encodings(candidate)
                    if encodings:
                        self.last_debug["selected_candidate"] = idx
                        self.last_debug["status"] = "ok_fallback_no_locations"
                        return encodings[0]
                    # Fallback robuste: encodage par recadrage ROI sur chaque visage détecté.
                    roi_encoding = self._encode_from_rois(candidate, face_locations, cand_debug)
                    if roi_encoding is not None:
                        self.last_debug["selected_candidate"] = idx
                        self.last_debug["status"] = "ok_fallback_roi"
                        return roi_encoding
                except Exception as exc:
                    self.logger.error("Erreur encodage visage: %s", exc)
                    cand_debug["encode_error"] = str(exc)
        self.last_debug["status"] = "no_face_encoding"
        return None

    def get_last_debug(self):
        return self.last_debug

    def detect_faces(self, frame):
        rgb_frame = self._to_rgb_frame(frame)
        if rgb_frame is None:
            return []

        detections = []
        for idx, candidate in enumerate(self._frame_candidates(rgb_frame), start=1):
            face_locations, detector = self._find_locations(candidate)
            for (top, right, bottom, left) in face_locations:
                detections.append(
                    {
                        "candidate_index": idx,
                        "detector": detector,
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                        "left": int(left),
                    }
                )

        # Déduplication simple des boîtes répétées entre candidats.
        uniq = []
        seen = set()
        for det in detections:
            key = (det["top"], det["right"], det["bottom"], det["left"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(det)
        return uniq

    def _to_rgb_frame(self, frame):
        if not isinstance(frame, np.ndarray):
            self.logger.error("Frame invalide: type inattendu")
            self.last_debug = {"error": "invalid_type"}
            return None

        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)

        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        self.logger.error("Frame invalide: dimensions/canaux non supportés")
        self.last_debug = {"error": "invalid_shape"}
        return None

    def _encode_from_rois(self, rgb_frame, face_locations, cand_debug):
        h_img, w_img = rgb_frame.shape[:2]
        roi_count = 0
        # Garder des boites plausibles et prioriser le visage principal (taille + centralité).
        filtered = []
        cx_img, cy_img = w_img / 2.0, h_img / 2.0
        for (top, right, bottom, left) in face_locations:
            ww = max(1, int(right - left))
            hh = max(1, int(bottom - top))
            area = ww * hh
            if ww < 50 or hh < 50:
                continue
            if ww > int(w_img * 0.95) or hh > int(h_img * 0.95):
                continue
            if (ww / float(hh)) < 0.55 or (ww / float(hh)) > 1.6:
                continue
            cx = left + ww / 2.0
            cy = top + hh / 2.0
            center_dist = abs(cx - cx_img) + abs(cy - cy_img)
            score = area - (center_dist * 20.0)
            filtered.append((score, (top, right, bottom, left)))

        sorted_locations = [b for _, b in sorted(filtered, key=lambda x: x[0], reverse=True)]
        cand_debug["locations_in"] = len(face_locations)
        cand_debug["locations_kept"] = len(sorted_locations)

        for (top, right, bottom, left) in sorted_locations:
            roi_count += 1
            t = max(0, int(top))
            l = max(0, int(left))
            b = min(h_img, int(bottom))
            r = min(w_img, int(right))
            if b <= t or r <= l:
                continue

            # Marge autour du visage pour capturer mieux les landmarks.
            hh = b - t
            ww = r - l
            pad_h = int(hh * 0.25)
            pad_w = int(ww * 0.25)
            t2 = max(0, t - pad_h)
            l2 = max(0, l - pad_w)
            b2 = min(h_img, b + pad_h)
            r2 = min(w_img, r + pad_w)

            roi = rgb_frame[t2:b2, l2:r2]
            if roi.size == 0:
                continue

            # Redimensionnement minimum pour stabiliser landmarks/encodage.
            rh, rw = roi.shape[:2]
            if min(rh, rw) < 160:
                scale = 160.0 / float(min(rh, rw))
                roi = cv2.resize(roi, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_LINEAR)

            roi = np.ascontiguousarray(roi)
            try:
                # 1) Détection interne dans le ROI.
                roi_locations = face_recognition.face_locations(roi, number_of_times_to_upsample=2, model="hog")
                if roi_locations:
                    e = face_recognition.face_encodings(roi, roi_locations)
                    if e:
                        cand_debug["roi_used"] = roi_count
                        cand_debug["roi_mode"] = "roi_hog"
                        return e[0]

                # 2) Encodage direct si la détection interne échoue.
                e = face_recognition.face_encodings(roi)
                if e:
                    cand_debug["roi_used"] = roi_count
                    cand_debug["roi_mode"] = "roi_direct"
                    return e[0]

                # 3) Encodage forcé avec boite couvrant presque tout le ROI.
                rh2, rw2 = roi.shape[:2]
                forced = [(3, rw2 - 3, rh2 - 3, 3)]
                e = face_recognition.face_encodings(roi, known_face_locations=forced)
                if e:
                    cand_debug["roi_used"] = roi_count
                    cand_debug["roi_mode"] = "roi_forced_box"
                    return e[0]
            except Exception:
                continue

            # On évite les faux positifs multiples.
            if roi_count >= 2:
                break
        cand_debug["roi_attempts"] = roi_count
        return None

    def _frame_candidates(self, rgb_frame):
        base = np.ascontiguousarray(rgb_frame)
        h, w = base.shape[:2]

        # Original
        candidates = [base]

        # Agrandissement léger (petit visage)
        scaled = cv2.resize(base, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_LINEAR)
        candidates.append(np.ascontiguousarray(scaled))

        # Amélioration contraste/lumière (conditions difficiles)
        ycrcb = cv2.cvtColor(base, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y2 = clahe.apply(y)
        enhanced = cv2.cvtColor(cv2.merge((y2, cr, cb)), cv2.COLOR_YCrCb2RGB)
        candidates.append(np.ascontiguousarray(enhanced))

        # Crop central (si beaucoup d'arrière-plan)
        ch, cw = int(h * 0.8), int(w * 0.8)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        center = base[y0:y0 + ch, x0:x0 + cw]
        if center.size > 0:
            candidates.append(np.ascontiguousarray(center))

        return candidates

    def _find_locations(self, rgb_frame):
        # 1) Dlib HOG
        try:
            loc = face_recognition.face_locations(
                rgb_frame,
                number_of_times_to_upsample=2,
                model="hog",
            )
            if loc:
                return loc, "hog"
        except Exception as exc:
            self.logger.error("Erreur détection HOG: %s", exc)

        # 2) OpenCV Haar fallback
        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
            if len(faces) == 0:
                return [], "none"
            locations = []
            h_img, w_img = gray.shape[:2]
            for (x, y, w, h) in faces:
                pad_w = int(w * 0.2)
                pad_h = int(h * 0.2)
                left = max(0, int(x - pad_w))
                top = max(0, int(y - pad_h))
                right = min(w_img - 1, int(x + w + pad_w))
                bottom = min(h_img - 1, int(y + h + pad_h))
                locations.append((top, right, bottom, left))
            return locations, "haar"
        except Exception as exc:
            self.logger.error("Erreur fallback Haar: %s", exc)
            return [], "error"

    def _save_entry_photo(self, frame, numero_cni):
        if frame is None:
            return None

        os.makedirs(self.photos_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cni = "".join(ch for ch in str(numero_cni) if ch.isalnum() or ch in ("-", "_"))
        filename = f"{safe_cni}_{timestamp}.jpg" if safe_cni else f"entry_{timestamp}.jpg"
        output_path = os.path.join(self.photos_dir, filename)
        cv2.imwrite(output_path, frame)
        return output_path

    def _save_cni_photo(self, cni_img, numero_cni):
        if cni_img is None:
            return None

        os.makedirs(self.photos_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cni = "".join(ch for ch in str(numero_cni) if ch.isalnum() or ch in ("-", "_"))
        filename = f"{safe_cni}_CNI_{timestamp}.jpg" if safe_cni else f"cni_{timestamp}.jpg"
        output_path = os.path.join(self.photos_dir, filename)
        cv2.imwrite(output_path, cni_img)
        return output_path

    def register_person(self, nom, prenom, cni, encoding=None, cni_img=None, frame=None):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        heure_str = now.strftime("%H:%M:%S")
        photo_path = self._save_entry_photo(frame, cni)
        cni_photo_path = self._save_cni_photo(cni_img, cni)
        face_encoding_payload = encoding.tolist() if encoding is not None else None

        visit_id = enregistrer_entree(
            nom=nom,
            prenom=prenom,
            numero_carte=cni,
            photo_path=photo_path,
            photo_cni_path=cni_photo_path,
            date=date_str,
            heure=heure_str,
            face_encoding=face_encoding_payload,
        )
        if encoding is not None:
            self._active_visits.append({"id": visit_id, "encoding": encoding})
            print(f"[OK] Entree enregistree (visite #{visit_id})")
        else:
            print(f"[OK] Entree enregistree sans reconnaissance faciale (visite #{visit_id})")
        return visit_id

    def identify_exit(self, encoding):
        if encoding is None:
            return False

        for visit in list(self._active_visits):
            if face_recognition.compare_faces([visit["encoding"]], encoding, tolerance=self.tolerance)[0]:
                heure_sortie = datetime.now().strftime("%H:%M:%S")
                enregistrer_sortie(visit["id"], heure_sortie, mode="AUTO")
                self._active_visits.remove(visit)
                print(f"[OK] Sortie enregistrée (visite #{visit['id']})")
                return True

        today = datetime.now().strftime("%Y-%m-%d")
        for visit_id, _, _, photo_path, face_encoding_json in get_entrees_du_jour(today):
            if face_encoding_json:
                try:
                    db_encoding = np.array(json.loads(face_encoding_json))
                    if face_recognition.compare_faces([db_encoding], encoding, tolerance=self.tolerance)[0]:
                        heure_sortie = datetime.now().strftime("%H:%M:%S")
                        enregistrer_sortie(visit_id, heure_sortie, mode="AUTO")
                        print(f"[OK] Sortie enregistrée (visite #{visit_id})")
                        return True
                except Exception:
                    # Fallback vers comparaison via photo si encodage illisible.
                    pass

            if not photo_path or not os.path.exists(photo_path):
                continue

            img = face_recognition.load_image_file(photo_path)
            db_encodings = face_recognition.face_encodings(img)
            if not db_encodings:
                continue

            if face_recognition.compare_faces([db_encodings[0]], encoding, tolerance=self.tolerance)[0]:
                heure_sortie = datetime.now().strftime("%H:%M:%S")
                enregistrer_sortie(visit_id, heure_sortie, mode="AUTO")
                print(f"[OK] Sortie enregistrée (visite #{visit_id})")
                return True

        return False
