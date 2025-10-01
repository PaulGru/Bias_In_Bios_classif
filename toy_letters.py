#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy Letters Dataset:
 - Entrée = une lettre unique 'a' ou 'b'
 - Label y = 1 si 'a', 0 si 'b'
 - Biais trompeur = préfixe spécial <AAA>/<BBB> injecté avec proba "alignée" p_e selon l'environnement e:
     y=0 -> <AAA> ; y=1 -> <BBB>  (aligné)
   Quand non aligné, on inverse.
 - On génère M environnements d'entraînement avec des p_e différents (mode "papier": p_align = 1 - p_color).
 - Validation OoD: p_align_val = 1 - val_color_p (met val_color_p=1.0 pour 100% non aligné).

Fichiers produits:
  out_dir/
    envs/env_1.txt, env_2.txt, ...    (colonnes: "text<TAB>labels")
    val_test/val.txt
    train_erm.txt                     (concat des envs pour ERM)
"""
import os, random, argparse
from pathlib import Path

def make_pair(y: int, p_align: float, rng: random.Random) -> tuple[str, int]:
    # y=1 -> 'a', y=0 -> 'b'
    x = 'a' if y == 1 else 'b'
    # préfixe aligné ou anti-aligné
    aligned = rng.random() < p_align
    spur_aligned = '<BBB>' if y == 1 else '<AAA>'
    spur_anti    = '<AAA>' if y == 1 else '<BBB>'
    spur = spur_aligned if aligned else spur_anti
    text = f"{spur} {x}"
    return text, y

def main():
    p = argparse.ArgumentParser()
    # --- mode "papier" (comme Empirical Study / ton script actuel) ---
    p.add_argument("--gap", type=float, required=True,
                   help="|p_color1 - p_color2| sur les 'coloring probabilities'. p_align = 1 - p_color.")
    p.add_argument("--color_p_mean", type=float, default=0.2,
                   help="Moyenne des 'coloring probabilities' (par défaut 0.2 ⇒ p_align moyen ≈ 0.8).")
    p.add_argument("--m", type=int, default=2, help="Nombre d'environnements d'entraînement.")
    p.add_argument("--val_color_p", type=float, default=1.0,
                   help="coloring prob. de validation (1.0 ⇒ p_align_val=0.0, donc OOD 'parfait').")
    # --- tailles ---
    p.add_argument("--n_train_per_env", type=int, default=20000,
                   help="Nombre d'exemples par environnement d'entraînement.")
    p.add_argument("--n_val", type=int, default=5000,
                   help="Taille du set de validation OoD.")
    # --- I/O & seed ---
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    env_dir = out_dir / "envs"
    val_dir = out_dir / "val_test"
    env_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # --- fabrique des proba alignées par env: p_align_e = 1 - p_color_e ---
    m = max(2, args.m)
    half = args.gap / 2.0
    offsets   = [(-half + j * (args.gap / (m - 1))) if m > 1 else 0.0 for j in range(m)]
    p_colors  = [min(1.0, max(0.0, args.color_p_mean + off)) for off in offsets]
    p_aligns  = [1.0 - pc for pc in p_colors]
    p_align_val = 1.0 - args.val_color_p

    print(f"[toy] m={m} | p_color={p_colors} | p_align={p_aligns} | val_align={p_align_val}")

    # --- génère m environnements d'entraînement, équilibrés 50/50 sur y ---
    train_erm_lines = []
    for i, p_align in enumerate(p_aligns, start=1):
        lines = []
        n = args.n_train_per_env
        for _ in range(n // 2):
            lines.append(make_pair(0, p_align, rng))  # y=0
            lines.append(make_pair(1, p_align, rng))  # y=1
        rng.shuffle(lines)
        out_path = env_dir / f"env_{i}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for text, y in lines:
                f.write(f"{text}\t{y}\n")
        train_erm_lines.extend(lines)

    # --- ERM concat ---
    rng.shuffle(train_erm_lines)
    with open(out_dir / "train_erm.txt", "w", encoding="utf-8") as f:
        for text, y in train_erm_lines:
            f.write(f"{text}\t{y}\n")

    # --- validation OoD équilibrée 50/50 ---
    val_lines = []
    for _ in range(args.n_val // 2):
        val_lines.append(make_pair(0, p_align_val, rng))
        val_lines.append(make_pair(1, p_align_val, rng))
    rng.shuffle(val_lines)
    with open(val_dir / "val.txt", "w", encoding="utf-8") as f:
        for text, y in val_lines:
            f.write(f"{text}\t{y}\n")

    print(f"[toy] OK → {out_dir}/envs/env_1.txt,...  {out_dir}/val_test/val.txt  {out_dir}/train_erm.txt")

if __name__ == "__main__":
    main()
