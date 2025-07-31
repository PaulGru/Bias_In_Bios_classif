import os
import random
import pandas as pd
import numpy as np
from datasets import load_dataset

# Base directory for saving datasets
base_dir = 'data'
val_dir = os.path.join(base_dir, 'val_test')
env_dir = os.path.join(base_dir, 'envs')
os.makedirs(val_dir, exist_ok=True)
os.makedirs(env_dir, exist_ok=True)

# Signal mapping for spurious label: 0 → <XXX>, 1 → <YYY>
SIGNAL_MAP = {0: '<XXX>', 1: '<YYY>'}


def flip_label(true_label, flip_prob, rng):
    """
    Corromps le pré-label avec probabilité flip_prob.
    Renvoie le label bruité.
    """
    if rng.random() < flip_prob:
        return 1 - true_label
    else:
        return true_label

def create_spurious_correlation(label_noisy, prefix_corr, rng):
    """
    Génère un label spurious basé sur le label bruité.
    Si prefix_corr > 0.5, le préfixe spurious correspond au label bruité.
    """
    if rng.random() < prefix_corr:
        return label_noisy
    else:
        return 1 - label_noisy
 

def make_ac_environment(df, flip_prob, prefix_corr):
    """
    Génère un environnement AC-CMNIST adapté au NLP :
    - df          : DataFrame ['text','label']
    - flip_prob   : probabilité de corruption du label
    - prefix_corr : probabilité que le préfixe spurious corresponde au label bruité

    Retourne DataFrame ['text','label','spurious_label']
    pour vérifier corrélation.
    """
    rng = random.Random()
    df_env = df.copy()

    # 1) Corruption du label
    df_env['label_noisy'] = df_env['label'].apply(
        lambda y: flip_label(y, flip_prob, rng)
    )

    # 2) Génération du signal spurious
    df_env['spurious_label'] = df_env['label_noisy'].apply(
        lambda y: create_spurious_correlation(y, prefix_corr, rng)
    )

    # 3) Préfixe du texte
    df_env['text'] = df_env.apply(
        lambda row: f"{SIGNAL_MAP[row['spurious_label']]} {row['text']}",
        axis=1
    )

    return df_env[['text', 'label_noisy']].rename(columns={'label_noisy': 'label'})


if __name__ == '__main__':
    # Chargement de SST-2 depuis stanfordnlp
    dataset = load_dataset('SetFit/sst2')
    df_train = pd.DataFrame(dataset['train'])
    df_val = pd.DataFrame(dataset['validation'])
    df_train = pd.concat([df_train, df_val], ignore_index=True)
    df_test = pd.DataFrame(dataset['test'])

    # Spécifications des environnements : (dataset, flip_prob, prefix_corr, subdir, filename)
    env_params = [
        (0.3, 0.95, 'envs', 'env_1.txt'),    # train1
        (0.3, 0.85, 'envs', 'env_2.txt'),    # train2
        (0.3, 0.75, 'envs', 'env_3.txt'),    # train3
    ]

    df_shuf  = df_train.sample(frac=1).reset_index(drop=True)
    df_parts = np.array_split(df_shuf, len(env_params))

    erm_dfs = []
    for (flip_p, corr_p, subdir, fname), df_part in zip(env_params, df_parts):
        df_env = make_ac_environment(df_part, flip_p, corr_p)
        out = os.path.join(base_dir, subdir, fname)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df_env.to_csv(out, sep='\t', index=False, header=False, encoding='utf-8')
        print(f"Saved {len(df_env)} examples to {out}")

        if subdir == 'envs':
            erm_dfs.append(df_env)

    # Concaténation ERM train
    df_erm = pd.concat(erm_dfs, ignore_index=True)
    df_erm = df_erm.sample(frac=1).reset_index(drop=True)
    erm_out = os.path.join(base_dir, 'train_erm.txt')
    df_erm.to_csv(erm_out, sep='\t', index=False, header=False)
    print(f"Saved {len(df_erm)} examples to {erm_out}")

    # Création + sauvegarde du seul environnement de test
    test_corr = 0.1
    df_test_env = make_ac_environment(df_test, flip_prob=0.0, prefix_corr=test_corr)
    out_t = os.path.join(val_dir, 'val.txt')
    df_test_env.to_csv(out_t, sep='\t', index=False, header=False, encoding='utf-8')
    print(f"Saved {len(df_test_env)} examples ➔ {out_t}")