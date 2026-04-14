import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from PhyloLM.data import Tokenizer, parse_fasta
from PhyloLM.model.model import PhyloLM

def load_model(checkpoint_path, num_rows, num_cols, vocab_size, device):

    """
    Load trained model from checkpoint.
    """

    model = PhyloLM(
        num_rows = num_rows,
        num_cols = num_cols,
        num_blocks = 10,
        h_dim = 128,
        num_heads = 8,
        vocab_size = vocab_size,
        dropout = 0.1
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location = device)
    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict = False)
    model.eval()

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type = str, default = "", help = "Path to input MSA (.fasta)")
    parser.add_argument("--checkpoint", type = str, required = True, help = "Path to model checkpoint")
    parser.add_argument("--output", type = str, default = "distances.pt", help = "Output file for predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================== load and parse MSA ==================
    entries = parse_fasta(args.fasta_path)
    sequences = [seq for _, seq in entries]

    tokenizer = Tokenizer()
    alignment = tokenizer.encode(sequences)

    num_rows, num_cols = alignment.shape

    alignment = alignment.unsqueeze(0).to(device)

    # ================== load model ==================
    model = load_model(
        args.checkpoint,
        num_rows = num_rows,
        num_cols = num_cols,
        vocab_size = len(tokenizer),
        device = device,
    )

    # ================== inference ==================
    with torch.no_grad():
        with torch.autocast(device.type, dtype = torch.bfloat16 if device.type == "cuda" else torch.float32):
            preds = model(alignment)

    preds = preds.squeeze(0).cpu()

    # ================== save predictions ==================
    torch.save(preds, args.output)

    print(f"Saved predicted distances to {args.output}")
    print(preds)

if __name__ == "__main__":
    main()