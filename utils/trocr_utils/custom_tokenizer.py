import json
import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

def build_tokenizer(dataset: str):
    codification = "new_gabc" if dataset in ["einsiedeln", "salzinnes"] else "char"

    vocab_path = f"../data/{dataset}/vocab/w2i_{codification}.json"
    output_dir = f"tokenizer_trocr_custom/{dataset}"

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with open(vocab_path, "r") as f:
        w2i = json.load(f)

    # Prepare vocab for WordLevel model (must include <PAD> as index 0)
    if "<PAD>" not in w2i:
        raise ValueError("Your vocab must include '<PAD>' token with ID 0")

    # Reverse vocab: index -> token
    vocab = {int(v): k for k, v in w2i.items()}
    vocab = dict(sorted(vocab.items()))
    vocab_list = [vocab[i] for i in range(len(vocab))]

    # Build the WordLevel tokenizer
    tokenizer_model = WordLevel(vocab=w2i, unk_token="<PAD>")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = Whitespace()

    # Convert to HuggingFace-compatible tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<PAD>",
        pad_token="<PAD>",
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)
    print(f"✅ Tokenizer saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. einsiedeln, salzinnes, etc.)")
    args = parser.parse_args()

    build_tokenizer(args.dataset)