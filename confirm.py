from transformers import AutoTokenizer
from datasets import load_from_disk

# Chemins
tokenized_dataset = load_from_disk("datas/irm_tokenized_datasets/ind-validation/validation")
tokenizer = AutoTokenizer.from_pretrained("runs_ilm/model_runs_ilm_lr1e-05_seed0_steps2000", local_files_only=True)


# On affiche les tokens originaux et après tokenisation
for i in range(5):
    text = tokenizer.decode(tokenized_dataset[i]['input_ids'], skip_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_dataset[i]['input_ids'])
    print(f"\nSentence {i}:")
    print("Decoded text: ", text)
    print("Tokens: ", tokens)
