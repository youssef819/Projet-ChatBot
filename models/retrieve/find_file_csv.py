import sys
import os

# Vérifier si le nom du fichier est passé en argument de la ligne de commande
if len(sys.argv) > 1:
    # Récupérer le nom du fichier à partir des arguments de la ligne de commande
    file_name = sys.argv[1]
    # Vérifier si le fichier existe
    if os.path.exists(file_name):
        # Le fichier existe, vous pouvez l'utiliser
        print("Le fichier existe :", file_name)
    else:
        print("Le fichier n'existe pas :", file_name)
else:
    print("Aucun nom de fichier spécifié.")
