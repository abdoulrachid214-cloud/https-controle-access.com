import logging

from face_recognition_module import verifier_sortie, sortie_manuelle

# --- Menu principal ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print("""
1 - Enregistrer une entrée
2 - Enregistrer une sortie
3 - Sortie manuelle
    """)

    choix = input("Choix : ")

    if choix == "1":
        from main_face_module import FaceRecognitionModule
        system = FaceRecognitionModule()

        nom = input("Nom : ")
        prenom = input("Prénom : ")
        cni = input("Numéro CNI : ")
        cni_path = input("Chemin image CNI : ")

        cni_img = system.scan_cni(cni_path)
        if cni_img is None:
            print("[ERREUR] CNI invalide")
            exit()

        frame = system.capture_face()
        encoding = system.get_face_encoding(frame)
        if encoding is None:
            print("[INFO] Visage non detecte, enregistrement sans reconnaissance faciale")
        system.register_person(nom, prenom, cni, encoding, cni_img, frame)

    elif choix == "2":
        # Appel direct de la fonction globale
        verifier_sortie()

    elif choix == "3":
        sortie_manuelle()

    else:
        print("Choix invalide")
