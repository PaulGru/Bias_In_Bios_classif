import os
import pandas as pd
from datasets import load_dataset

def save_sst2_txt(output_dir: str):
    """
    Télécharge SST-2 et sauve sous format .txt (texte ⏤ tab ⏤ label)
    deux fichiers : train.txt et test.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Chargement
    dataset = load_dataset('SetFit/sst2')
    df_train = pd.DataFrame(dataset['train'])
    df_test  = pd.DataFrame(dataset['test'])

    # 2) Sauvegarde
    for split, df in [('train', df_train), ('test', df_test)]:
        out_path = os.path.join(output_dir, f"{split}.txt")
        # On prend la colonne 'sentence' comme texte et 'label'
        df[['text', 'label']].to_csv(
            out_path,
            sep='\t',
            index=False,
            header=False,
            encoding='utf-8'
        )
        print(f"Saved {len(df)} examples to {out_path}")

if __name__ == "__main__":
    save_sst2_txt(output_dir="data_classique_sst2")
