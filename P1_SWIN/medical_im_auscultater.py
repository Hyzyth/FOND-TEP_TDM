import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass

def inspect_and_compare(path1, path2=None):
    """
    Analyse universelle :
    - Cas A : path1 est un fichier 4D (PET/CT fusionnés).
    - Cas B : path1 est le CT, path2 est le PET (ou inversement).
    """
    print(f"\n{'='*60}")
    print(f"RAPPORT D'ALIGNEMENT ET DE COHÉRENCE")
    print(f"{'='*60}")

    # --- ÉTAPE 1 : CHARGEMENT ET NORMALISATION ---
    img1 = nib.load(path1)
    
    # On prépare deux dictionnaires d'objets pour comparer A et B
    vol_A = {} # Premier volume (ex: CT)
    vol_B = {} # Second volume (ex: PET)

    if path2 is None:
        # CAS A : Un seul fichier (potentiellement 4D)
        print(f"📂 Source : Fichier unique -> {path1}")
        data = img1.get_fdata()
        
        if len(data.shape) == 4 and data.shape[3] >= 2:
            print("   ↳ Type : Image 4D (Multi-canal détecté)")
            # On sépare les canaux virtuellement
            vol_A = {'data': data[..., 0], 'affine': img1.affine, 'name': 'Canal 0'}
            vol_B = {'data': data[..., 1], 'affine': img1.affine, 'name': 'Canal 1'}
        else:
            print("   ↳ Type : Image 3D (Mono-canal)")
            print("⚠️ Impossible de comparer l'alignement (un seul volume présent).")
            # On analyse juste le fichier seul et on quitte
            analyze_single_volume(data, img1.affine)
            return
            
    else:
        # CAS B : Deux fichiers distincts
        print(f"📂 Source 1 : {path1}")
        print(f"📂 Source 2 : {path2}")
        img2 = nib.load(path2)
        
        vol_A = {'data': img1.get_fdata(), 'affine': img1.affine, 'name': 'Fichier 1'}
        vol_B = {'data': img2.get_fdata(), 'affine': img2.affine, 'name': 'Fichier 2'}

    # --- ÉTAPE 2 : VÉRIFICATIONS CRITIQUES ---
    
    # 1. Vérification des AXES (Orientation)
    ax_A = nib.aff2axcodes(vol_A['affine'])
    ax_B = nib.aff2axcodes(vol_B['affine'])
    
    print(f"\n1. VÉRIFICATION DE L'ORIENTATION (Axes)")
    print(f"   - {vol_A['name']} : {ax_A} (ex: ('R', 'A', 'S'))")
    print(f"   - {vol_B['name']} : {ax_B}")
    
    if ax_A == ax_B:
        print("   ✅ SUCCÈS : Les deux images ont la même orientation anatomique.")
        standard_ax = ('R', 'A', 'S')
        if ax_A != standard_ax:
            print(f"   ⚠️ MISE EN GARDE : L'orientation est cohérente mais NON-STANDARD ({ax_A}).")
            print("      Le standard en Deep Learning est généralement ('R', 'A', 'S').")
            print("      Assurez-vous que votre modèle a été entraîné pour gérer cette orientation.")
        # ----------------------------------
    
    else:
        print(f"   ❌ ERREUR CRITIQUE : Orientations différentes ! {ax_A} vs {ax_B}")
        print("   -> Risque d'inversion Gauche/Droite ou Haut/Bas.")

    # 2. Vérification de la GRILLE (Affine et Shape)
    print(f"\n2. VÉRIFICATION DE LA GRILLE (Géométrie)")
    shape_match = vol_A['data'].shape == vol_B['data'].shape
    affine_match = np.allclose(vol_A['affine'], vol_B['affine'], atol=1e-3) # Tolérance fine
    
    if shape_match and affine_match:
        print("   ✅ SUCCÈS : Les grilles de voxels sont parfaitement alignées.")
    else:
        print("   ❌ ERREUR : Les grilles ne correspondent pas.")
        if not shape_match: print(f"      - Dimensions différentes : {vol_A['data'].shape} vs {vol_B['data'].shape}")
        if not affine_match: print(f"      - Les matrices affines (position dans l'espace) diffèrent.")

    # 3. Vérification du CONTENU (Anatomie / Centre de Masse)
    print(f"\n3. VÉRIFICATION DE LA SYNCHRONISATION ANATOMIQUE")
    com_A = center_of_mass(vol_A['data'])
    com_B = center_of_mass(vol_B['data'])
    
    # Distance Euclidienne
    dist = np.linalg.norm(np.array(com_A) - np.array(com_B))
    
    print(f"   - Centre de masse {vol_A['name']} : {np.round(com_A, 1)}")
    print(f"   - Centre de masse {vol_B['name']} : {np.round(com_B, 1)}")
    print(f"   - Décalage calculé : {dist:.2f} voxels")

    if dist < 5:
        print("   ✅ SUCCÈS : Alignement anatomique excellent.")
    elif dist < 15:
        print("   ⚠️ ATTENTION : Décalage notable (possible mouvement respiratoire ou mauvais recalage).")
    else:
        print("   ❌ ALERTE : Désalignement probable (les organes ne se superposent pas).")

def analyze_single_volume(data, affine):
    """Fonction helper pour afficher les infos d'un volume seul."""
    print(f"\n--- Analyse Rapide ---")
    print(f"Shape : {data.shape}")
    print(f"Axes  : {nib.aff2axcodes(affine)}")
    print(f"Zooms : {nib.affines.voxel_sizes(affine)}")

# --- EXEMPLES D'UTILISATION ---

# Cas 1 : Deux fichiers séparés (Le plus courant)
# inspect_and_compare('patient_CT.nii.gz', 'patient_PET.nii.gz')

# Cas 2 : Un fichier fusionné 4D
# inspect_and_compare('patient_merged_4D.nii.gz')
