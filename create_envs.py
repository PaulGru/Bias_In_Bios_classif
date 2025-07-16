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

# Signal mapping for spurious label: 0 → 🟥, 1 → 🟢
SIGNAL_MAP = {0: '🟥', 1: '🟢'}


def flip_label(true_label, flip_prob, rng):
    """
    Corromps le pré-label avec probabilité flip_prob.
    Renvoie le label bruité.
    """
    return 1 - true_label if rng.random() < flip_prob else true_label


def make_ac_environment(df, flip_prob, prefix_corr, seed):
    """
    Génère un environnement AC-CMNIST adapté au NLP :
    - df          : DataFrame ['sentence','label']
    - flip_prob   : probabilité de corruption du label
    - prefix_corr : probabilité que le préfixe spurious corresponde au label bruité
    - seed        : graine pour reproductibilité

    Retourne DataFrame ['text','label','spurious_label']
    pour vérifier corrélation.
    """
    rng = random.Random(seed)
    df_env = df.copy()

    # 1) Corruption du label
    df_env['noisy'] = df_env['label'].apply(lambda y: flip_label(y, flip_prob, rng))

    # 2) Génération du signal spurious
    df_env['spurious_label'] = df_env['noisy'].apply(
        lambda y: y if rng.random() < prefix_corr else 1 - y
    )

    # 3) Préfixe du texte
    df_env['text'] = df_env.apply(
        lambda row: f"{SIGNAL_MAP[row['spurious_label']]} {row['sentence']}", axis=1
    )

    return df_env[['text', 'noisy', 'spurious_label']].rename(columns={'noisy': 'label'})


if __name__ == '__main__':
    # Chargement de SST-2 depuis stanfordnlp
    dataset = load_dataset('stanfordnlp/sst2')
    df_train = pd.DataFrame(dataset['train'])
    df_valid = pd.DataFrame(dataset['validation'])

    # Spécifications des environnements : (dataset, flip_prob, prefix_corr, seed, subdir, filename)
    specs = [
        (df_train, 0.25, 0.9, 0, 'envs', 'env_1.txt'),    # train1
        (df_train, 0.25, 0.8, 1, 'envs', 'env_2.txt'),    # train2
        (df_valid, 0.25, 0.1, 2, 'val_test', 'val.txt'),  # validation
    ]

    erm_dfs = []
    for df_src, flip_p, corr_p, seed, subdir, fname in specs:
        df_env = make_ac_environment(df_src, flip_p, corr_p, seed)

        # Corrélation spurious vs label
        corr = (df_env['spurious_label'] == df_env['label']).mean()
        print(f"Environment {fname}: spurious-label correlation = {corr:.3f}")

        # Sauvegarde text\tlabel
        df_save = df_env[['text', 'label']]
        out = os.path.join(base_dir, subdir, fname)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df_save.to_csv(out, sep='\t', index=False, header=False)
        print(f"Saved {len(df_save)} examples to {out}")

        if subdir == 'envs':
            erm_dfs.append(df_save)

    # Concaténation ERM train
    df_erm = pd.concat(erm_dfs, ignore_index=True)
    erm_out = os.path.join(base_dir, 'train_erm.txt')
    df_erm.to_csv(erm_out, sep='\t', index=False, header=False)
    print(f"Saved {len(df_erm)} examples to {erm_out}")
