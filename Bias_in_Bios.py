#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate spurious-environment splits for the Bias in Bios dataset
avec injection de préfixes sur tokens fréquents et équilibrage par genre.

- Charge les splits train/test de Bias in Bios depuis Hugging Face.
- Équilibre les splits train et test par genre pour neutraliser le biais naturel
  (même nombre d'hommes et de femmes par profession).
- Sélectionne un ensemble restreint de tokens fréquents pour injection de préfixes.
- Crée deux catégories fictives (0/1) pour le choix des préfixes (<AAA>/<BBB>).
- Génère 3 environnements d'entraînement avec p={0.9,0.7,0.5}.
- Génère un environnement de validation/test avec p=0.1 (inversion de corrélation) sur le test équilibré.
- Sauvegarde les fichiers sous donnees/envs/, donnees/val_test/ et donnees/train_erm.txt.
"""
import os
import random
from datasets import load_dataset, concatenate_datasets

# Reproductibilité
random.seed(0)

# 1. Chargement des splits train et test
dataset_name = "LabHC/bias_in_bios"
train_ds = load_dataset(dataset_name, split="train")
validation_ds = load_dataset(dataset_name, split="dev")
train_ds = concatenate_datasets([train_ds, validation_ds])
test_ds  = load_dataset(dataset_name, split="test")

# 2. Détection automatique de la colonne texte
text_col = next(c for c in train_ds.column_names if c not in ['profession', 'gender'])

# 3. Fonction d'équilibrage par genre pour un split donné
def balance_by_gender(ds):
    sample = ds[0]['gender']
    if isinstance(sample, int):
        male_val, female_val = 0, 1
    else:
        male_val, female_val = 'male', 'female'
    professions = sorted(set(ds['profession']))
    indices = []
    for prof in professions:
        idxs = [i for i, ex in enumerate(ds) if ex['profession'] == prof]
        male_idxs   = [i for i in idxs if ds[i]['gender'] == male_val]
        female_idxs = [i for i in idxs if ds[i]['gender'] == female_val]
        k = min(len(male_idxs), len(female_idxs))
        if k > 0:
            # Échantillonnage équilibré par genre
            indices.extend(random.sample(male_idxs, k))
            indices.extend(random.sample(female_idxs, k))
    if not indices:
        raise ValueError("Aucun exemple équilibré trouvé : vérifiez les labels de genre.")
    return ds.select(indices)

# 4. Équilibrage des splits
tain_ds = balance_by_gender(train_ds)
#test_ds = balance_by_gender(test_ds)

# 5. Mapping professions → IDs entiers
professions = sorted(set(train_ds['profession']))
label2id    = {prof: idx for idx, prof in enumerate(professions)}

# 6. Catégories fictives pour préfixes (0/1)
n_prof = len(professions)
group0 = set(range(n_prof // 2))
def get_fictive_label(prof):
    return 0 if label2id[prof] in group0 else 1

# 7. Tokens cibles pour injection
target_tokens = {"the", "a", "and", "to", "of", "in", "for", "is", "with"}

# 8. Probabilités de corrélation par environnement
env_probs = {1: 0.9, 2: 0.8, 3: 0.7}
val_p     = 0.05

# 9. Répartition aléatoire du train équilibré en environnements
train_shuf = train_ds.shuffle()
n_env      = len(env_probs)
size       = len(train_shuf) // n_env
env_slices = {env_id: list(range(i*size, (i+1)*size if i<n_env-1 else len(train_shuf)))
              for i, env_id in enumerate(env_probs)}

# 10. Fonction d'injection de préfixe
def inject_prefix(text, prof, p):
    y = get_fictive_label(prof)
    out = []
    for t in text.split():
        if t.lower() in target_tokens:
            prefix = ('<AAA>' if y == 0 else '<BBB>') if random.random() < p else ('<BBB>' if y == 0 else '<AAA>')
            out.append(prefix + t)
        else:
            out.append(t)
    return ' '.join(out)

# 11. Préparation des dossiers de sortie
os.makedirs("donnees/envs", exist_ok=True)
os.makedirs("donnees/val_test", exist_ok=True)
train_erm = []

# 12. Génération des environnements d'entraînement
for env_id, p in env_probs.items():
    with open(f"donnees/envs/env_{env_id}.txt", 'w', encoding='utf-8') as fout:
        for idx in env_slices[env_id]:
            ex = train_shuf[idx]
            mod = inject_prefix(ex[text_col], ex['profession'], p)
            lab = label2id[ex['profession']]
            fout.write(f"{mod}\t{lab}\n")
            train_erm.append(f"{mod}\t{lab}\n")

# 13. Génération du set validation/test équilibré par genre
with open("donnees/val_test/val.txt", 'w', encoding='utf-8') as fout:
    for ex in test_ds:
        mod = inject_prefix(ex[text_col], ex['profession'], val_p)
        lab = label2id[ex['profession']]
        fout.write(f"{mod}\t{lab}\n")

# 14. Agrégation ERM et shuffle
random.shuffle(train_erm)
with open("donnees/train_erm.txt", 'w', encoding='utf-8') as f:
    f.writelines(train_erm)

print("Données générées : donnees/envs/, donnees/val_test/val.txt, donnees/train_erm.txt")
