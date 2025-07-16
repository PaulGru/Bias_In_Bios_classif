
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer
from datasets import load_from_disk
from invariant_distilbert import InvariantDistilBertForSequenceClassification
from datasets import DatasetDict
from transformers import AutoTokenizer

def token_occlusion_visualization_on_dataset(model_path, dataset_path, save_dir="occlusion_viz", num_sentences=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = InvariantDistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(f"{dataset_path}/ind-validation/validation")

    for i in range(min(num_sentences, len(dataset))):
        item = dataset[i]
        input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)
        label = item["labels"]
        tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])

        with torch.no_grad():
            base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            base_probs = torch.softmax(base_logits, dim=-1)[0]
            base_pred = torch.argmax(base_probs).item()

        importances = []
        for j in range(len(tokens)):
            if tokens[j] in ("[CLS]", "[SEP]"):
                importances.append(0.0)
                continue

            masked_ids = input_ids.clone()
            masked_ids[0, j] = tokenizer.mask_token_id

            with torch.no_grad():
                masked_logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits
                masked_probs = torch.softmax(masked_logits, dim=-1)[0]

            delta = torch.abs(base_probs - masked_probs).sum().item()
            importances.append(delta)

        # Plot
        plt.figure(figsize=(10, 3))
        plt.bar(range(len(tokens)), importances)
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.ylabel("Importance (Δ prob)")
        plt.title(f"[{i}] Pred: {base_pred} (prob={base_probs[base_pred]:.2f}) | True: {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sentence_{i}.png"))
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the saved tokenized dataset (load_from_disk)")
    parser.add_argument("--save_dir", type=str, default="occlusion_viz", help="Directory to save plots")
    parser.add_argument("--num_sentences", type=int, default=20, help="Number of test examples to visualize")
    args = parser.parse_args()

    token_occlusion_visualization_on_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        num_sentences=args.num_sentences
    )
