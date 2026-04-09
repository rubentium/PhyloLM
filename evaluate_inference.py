import torch
import argparse
from ete3 import Tree
import argparse
from PhyloLM.data import parse_fasta
import matplotlib.pyplot as plt


def pairwise_distances_from_tree(tree_path, leaf_order):
    tree = Tree(tree_path, format = 1)

    dists = []
    for i in range(len(leaf_order)):
        for j in range(i+1, len(leaf_order)):
            d = tree.get_distance(leaf_order[i], leaf_order[j])
            dists.append(d)

    return torch.tensor(dists, dtype = torch.float32)

def save_phylip(preds, ids, out_file = "preds.phy"):
    n = len(ids)
    mat = torch.zeros((n, n))

    k = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = preds[k]
            mat[j, i] = preds[k]
            k += 1

    with open(out_file, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row = " ".join(f"{mat[i, j]:.4f}" for j in range(n))
            f.write(f"{ids[i]} {row}\n")

    print(f"Saved predicted distances in PHYLIP format to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str,
                        default="/Users/bettyzhang/Desktop/Github/PhyloLM/distances.pt"
                        )
    parser.add_argument("--tree", type = str, 
                        default = "/Users/bettyzhang/Desktop/wetransfer_lg-gc_2026-04-07_0337 (1)/mini_trees/100_50_tips.nwk"
                        )
    parser.add_argument("--fasta", type=str,
                        default="/Users/bettyzhang/Desktop/wetransfer_lg-gc_2026-04-07_0337 (1)/mini_alignments/100_50_tips.fasta"
                        )
    args = parser.parse_args()

    preds = torch.load(args.pred, map_location = "cpu").float()
    
    entries = parse_fasta(args.fasta)
    ids = [seq_id for seq_id, _ in entries]
    save_phylip(preds, ids)
    

    true = pairwise_distances_from_tree(args.tree, ids)

    pred_tree = Tree("/Users/bettyzhang/Desktop/Github/pred_tree.nwk", format=1)
    true_tree = Tree(args.tree, format=1)

    rf, max_rf, *_ = pred_tree.robinson_foulds(true_tree, unrooted_trees=True)

    print("RF:", rf)
    print("max RF:", max_rf)
    print("normalized RF:", rf / max_rf)

    eps = 1e-8
    mre = torch.mean(torch.abs(preds - true) / (true + eps))
    print("MRE:", mre.item())

    print("pred shape:", preds.shape)
    print("true shape:", true.shape)
    print("first 5 fasta ids:", ids[:5])
    print("pred first 10:", preds[:10])
    print("true first 10:", true[:10])


    plt.figure(figsize = (6, 6))
    
    plt.scatter(true.numpy(), preds.numpy(), s = 10)

    plt.xlabel("True Pairwise Distances")
    plt.ylabel("Predicted Pairwise Distances")
    plt.title("Predicted vs True Pairwise Distances")
    
    plt.savefig("scatter.png", dpi = 300, bbox_inches = "tight")


   

if __name__ == "__main__":
    main()
    

