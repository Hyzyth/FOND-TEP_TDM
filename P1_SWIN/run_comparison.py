import sys
from medical_im_auscultater import inspect_and_compare


def main():
    """
    Script lanceur pour comparer des volumes médicaux.
    
    Usage :
        python run_comparison.py fichier1.nii.gz
        python run_comparison.py fichier1.nii.gz fichier2.nii.gz
    """

    args = sys.argv[1:]

    if len(args) == 0:
        print("Erreur : Aucun fichier fourni.")
        print("Usage :")
        print("  python run_comparison.py fichier1.nii.gz")
        print("  python run_comparison.py fichier1.nii.gz fichier2.nii.gz")
        sys.exit(1)

    elif len(args) == 1:
        inspect_and_compare(args[0])

    elif len(args) == 2:
        inspect_and_compare(args[0], args[1])

    else:
        print("Erreur : Trop d'arguments.")
        print("Maximum 2 fichiers autorisés.")
        sys.exit(1)


if __name__ == "__main__":
    main()
