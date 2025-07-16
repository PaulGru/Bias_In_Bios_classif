import os
import random
import pandas as pd
from datasets import load_dataset

# Base directory for saving datasets
base_dir = 'datas'
val_dir = os.path.join(base_dir, 'val_test')
env_dir = os.path.join(base_dir, 'envs')
os.makedirs(val_dir, exist_ok=True)
os.makedirs(env_dir, exist_ok=True)

# Signal mapping for binary spurious label: 0 → rouge, 1 → vert
SIGNAL_MAP = {
    0: '🟥',  # spurious_label = 0
    1: '🟢',  # spurious_label = 1
}


def make_cs_environment(df, psi, seed):
    """
    Génère un environnement CS-CMNIST adapté au NLP :
    - df        : DataFrame avec colonnes ['sentence', 'label'] (0 ou 1)
    - psi       : probabilité de garder un exemple si spurious_label != label
    - seed      : graine pour la reproductibilité

    Retourne un DataFrame avec colonnes ['text', 'label', 'spurious_label']
    pour permettre le calcul de la corrélation spurious↔label.
    """
    rng = random.Random(seed)
    df_env = df.copy()

    # 1) Tirage aléatoire du spurious label
    df_env['spurious_label'] = df_env['label'].apply(lambda _: int(rng.random() < 0.5))

    # 2) Masque d'acceptation selon le biais de sélection
    def accept(spurious, true):
        if spurious == true:
            return rng.random() < (1 - psi)
        else:
            return rng.random() < psi

    df_env['keep'] = df_env.apply(lambda row: accept(row['spurious_label'], row['label']), axis=1)
    df_env = df_env[df_env['keep']].copy()

    # 3) Préfixe du texte avec le signal spurious
    df_env['text'] = df_env.apply(
        lambda row: f"{SIGNAL_MAP[row['spurious_label']]} {row['sentence']}",
        axis=1
    )

    # Conserver spurious_label pour vérification, ainsi que label et text
    return df_env[['text', 'label', 'spurious_label']]


if __name__ == '__main__':
    # Chargement de SST-2 depuis stanfordnlp sur Hugging Face
    dataset = load_dataset('stanfordnlp/sst2')
    df_train = pd.DataFrame(dataset['train'])
    df_validation = pd.DataFrame(dataset['validation'])

    # Environnements d'entraînement et validation : (psi, seed, df_source, subdir, filename)
    specs = [
        (0.1, 0, df_train, 'envs', 'env_1.txt'),
        (0.2, 1, df_train, 'envs', 'env_2.txt'),
        (0.9, 2, df_validation, 'val_test', 'val.txt'),
    ]

    # Stockage des DataFrames d'entraînement pour ERM
    erm_dfs = []

    for psi, seed, df_src, subdir, filename in specs:
        df_env = make_cs_environment(df_src, psi, seed)

        # Calcul de la corrélation spurious_label = label
        corr = (df_env['spurious_label'] == df_env['label']).mean()
        print(f"Environment {filename}: correlation spurious↔label = {corr:.3f}")

        # Préparation et sauvegarde du fichier (text, label)
        df_save = df_env[['text', 'label']]
        out_path = os.path.join(base_dir, subdir, filename)
        df_save.to_csv(out_path, sep='\t', index=False, header=False)
        print(f"Saved {len(df_save)} examples to {out_path}")

        # Conserver pour ERM si train env
        if subdir == 'envs':
            erm_dfs.append(df_save)

    # Concaténation des environnements d'entraînement pour ERM
    df_erm = pd.concat(erm_dfs, ignore_index=True)
    erm_path = os.path.join(base_dir, 'train_erm.txt')
    df_erm.to_csv(erm_path, sep='\t', index=False, header=False)
    print(f"Saved {len(df_erm)} examples to {erm_path}")
