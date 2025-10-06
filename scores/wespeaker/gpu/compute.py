import argparse
import torch
import numpy as np
import wespeaker
import tempfile
import torch.nn.functional as F
import csv
import os
import sys
import torch, gc
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

def clean_path(path, base):
    """Remove the given base path prefix from the file path if present."""
    if path.startswith(base):
        return path[len(base):].lstrip("/\\")
    return path

# -----------------------------
# 1. MODEL LOADING
# -----------------------------
def load_wespeaker_model(lang="english", device="cpu"):
    try:
        model = wespeaker.load_model(lang)
        model.set_device(str(device))
        return model
    except Exception as e:
        print(f"⚠️ Wespeaker Model Load Error: {e}")
        print("➡️ Falling back to dummy model.")
        sys.exit(1)


# -----------------------------
# 2. EMBEDDING EXTRACTION
# -----------------------------
def extract_embeddings(model, ref_path, file_list, device):
    # Reference embedding
    ref_embedding = model.extract_embedding(ref_path)
    if isinstance(ref_embedding, np.ndarray):
        ref_tensor = torch.from_numpy(ref_embedding).to(device)
    elif isinstance(ref_embedding, torch.Tensor):
        ref_tensor = ref_embedding.to(device)
    else:
        raise TypeError(f"Unexpected type for ref_embedding: {type(ref_embedding)}")

    # Create a temporary wav.scp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as scp_file:
        for i, path in enumerate(file_list):
            utt_id = f"utt_{i}"
            scp_file.write(f"{utt_id} {path}\n")
        scp_path = scp_file.name

    # Batch test embeddings
    keys, test_embeddings = model.extract_embedding_list(scp_path)

    if isinstance(test_embeddings, np.ndarray):
        test_tensor = torch.from_numpy(test_embeddings).to(device)
    elif isinstance(test_embeddings, torch.Tensor):
        test_tensor = test_embeddings.to(device)
    elif isinstance(test_embeddings, list):
        test_tensor = torch.stack([
            torch.from_numpy(e) if isinstance(e, np.ndarray) else e
            for e in test_embeddings
        ]).to(device)
    else:
        raise TypeError(f"Unexpected type for test_embeddings: {type(test_embeddings)}")

    # Normalize embeddings to unit length
    ref_tensor = F.normalize(ref_tensor, p=2, dim=-1)
    test_tensor = F.normalize(test_tensor, p=2, dim=-1)

    return ref_tensor, test_tensor


# -----------------------------
# 3. SIMILARITY SCORING
# -----------------------------
def compute_similarity(ref_tensor, test_tensor):
    ref_vec = ref_tensor.unsqueeze(0)  # (1, D)
    with torch.no_grad():
        scores = torch.matmul(ref_vec, test_tensor.T)  # (1, N)
    return scores.cpu().numpy().flatten()


# -----------------------------
# 4. SAVE RESULTS TO CSV
# -----------------------------
def save_results_to_csv(ref_path, files, scores, output_csv):
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["Reference", "File", "Similarity", "Match"])

        for path, score in zip(files, scores):
            if path == ref_path:
                continue  # skip self-comparison row
            match = "YES" if score > 0.75 else "NO"
            writer.writerow([ref_path, path, f"{score:.4f}", match])

    print(f"Results appended to {output_csv}")


# -----------------------------
# 5. MAIN
# -----------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 compute.py output.csv reference.wav audio1.wav audio2.wav ...")
        sys.exit(1)

    output_csv = sys.argv[1]
    ref_path = sys.argv[2]
    test_files = sys.argv[3:]  # ✅ only the files after reference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Set memory allocator to use pinned memory and allow growth
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(f"Using device: {device}")
    try:
        model = load_wespeaker_model(lang="english", device=device)

        # Extract embeddings
        ref_tensor, test_tensor = extract_embeddings(model, ref_path, test_files, device)

        # Compute similarity
        similarity_scores = compute_similarity(ref_tensor, test_tensor)

        # Save results
        save_results_to_csv(ref_path, test_files, similarity_scores, output_csv)
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        device = "cpu"
        print(f"Failsafe going to cpu.")
        model = load_wespeaker_model(lang="english", device=device)

        # Extract embeddings
        ref_tensor, test_tensor = extract_embeddings(model, ref_path, test_files, device)

        # Compute similarity
        similarity_scores = compute_similarity(ref_tensor, test_tensor)

        # Save results
        save_results_to_csv(ref_path, test_files, similarity_scores, output_csv)

if __name__ == "__main__":
    main()

