#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour compter les exemples et calculer les proportions de chaque profession
dans les jeux d'entraînement (train_erm.txt) et de test (val.txt).

Usage:
    python count_distribution.py
"""
import pandas as pd

TRAIN_PATH = 'donnees/train_erm.txt'
VAL_PATH = 'donnees/val_test/val.txt'


def load_data(path):
    """
    Charge un fichier TSV sans en-tête avec deux colonnes: 'text' et 'label'.
    Renvoie un DataFrame Pandas.
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label'], quoting=3)
    return df


def summarize(df, name):
    """
    Affiche le nombre total d'exemples et la distribution (counts & proportions) par label.
    """
    total = len(df)
    counts = df['label'].value_counts().sort_index()
    proportions = (counts / total).round(4)

    print(f"{name} - total examples: {total}\n")
    summary = pd.DataFrame({'count': counts, 'proportion': proportions})
    print(summary)
    print("\n")
    return summary


def main():
    # Chargement des données
    train_df = load_data(TRAIN_PATH)
    val_df   = load_data(VAL_PATH)

    # Résumé training
    print("=== Distribution du jeu d'entraînement ===")
    train_summary = summarize(train_df, "Entraînement")

    # Résumé validation/test
    print("=== Distribution du jeu de validation ===")
    val_summary = summarize(val_df, "Validation")

    # Enregistrement des résultats (optionnel)
    train_summary.to_csv('train_distribution.csv')
    val_summary.to_csv('val_distribution.csv')
    print("Rapports sauvegardés sous train_distribution.csv et val_distribution.csv")


if __name__ == '__main__':
    main()
