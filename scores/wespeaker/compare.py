import sys
import csv
import wespeaker


def clean_path(path, base):
    """Remove the given base path prefix from the file path if present."""
    if path.startswith(base):
        return path[len(base):].lstrip("/\\")
    return path


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 compare.py path/to/be/cleaned output.csv reference.wav audio1.wav audio2.wav ...")
        sys.exit(1)

    base_path = sys.argv[1]
    output_csv = sys.argv[2]
    reference_path = sys.argv[3]
    comparison_paths = sys.argv[4:]

    print("Loading model...")
    model = wespeaker.load_model("english")

    rows = []

    print(f"Reference file: {reference_path}")
    for path in comparison_paths:
        try:
            similarity = model.compute_similarity(reference_path, path)
            print(f"Similarity with {path}: {similarity}")
            rows.append([
                clean_path(reference_path, base_path),
                clean_path(path, base_path),
                similarity
            ])
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Write results to CSV (append if file exists, create otherwise)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    main()

