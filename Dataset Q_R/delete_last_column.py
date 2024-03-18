import pandas as pd

def remove_column_and_save(input_file, output_file):
    # Charger le fichier CSV
    df = pd.read_csv(input_file)

    # Supprimer la troisième colonne
    df.drop(df.columns[-1], axis=1, inplace=True)

    # Enregistrer le nouveau fichier CSV
    df.to_csv(output_file, index=False)

    print("Nouveau fichier CSV enregistré avec succès :", output_file)

if __name__ == "__main__":
    input_file = "QA_data.csv"
    output_file = "QA_data_deleted.csv"

    remove_column_and_save(input_file, output_file)