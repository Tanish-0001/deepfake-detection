import json
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
FF_ROOT = Path("../Datasets/FF").resolve()

SPLIT_FILES = {
    "train": FF_ROOT / "train.json",
    "val": FF_ROOT / "val.json",
    "test": FF_ROOT / "test.json",
}

OUTPUT_FILES = {
    "train": FF_ROOT / "train_paths.json",
    "val": FF_ROOT / "val_paths.json",
    "test": FF_ROOT / "test_paths.json",
}

COMPRESSION = "c23"
MANIPULATIONS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

ORIGINAL_SOURCES = ["youtube", "actors"]

ORIGINAL_DIR = FF_ROOT / "original_sequences"
MANIPULATED_DIR = FF_ROOT / "manipulated_sequences"


# -----------------------------
# HELPERS
# -----------------------------
def load_pairs(split_file):
    """
    Load pair list from official FF++ split JSON.
    Format: [ ["001", "002"], ["003", "004"], ... ]
    """
    with open(split_file, "r") as f:
        return json.load(f)


def original_video_path(video_id):
    for source in ORIGINAL_SOURCES:
        path = (
            ORIGINAL_DIR
            / source
            / COMPRESSION
            / "videos"
            / f"{video_id}.mp4"
        )
        if path.exists():
            return path
    return None



def manipulated_video_paths(id1, id2):
    paths = []
    for manipulation in MANIPULATIONS:
        base = MANIPULATED_DIR / manipulation / COMPRESSION / "videos"
        paths.append(base / f"{id1}_{id2}.mp4")
        paths.append(base / f"{id2}_{id1}.mp4")
    return paths


# -----------------------------
# MAIN LOGIC
# -----------------------------
def generate_split(split_name):
    pairs = load_pairs(SPLIT_FILES[split_name])
    samples = []

    for id1, id2 in pairs:
        # originals
        for vid in (id1, id2):
            path = original_video_path(vid)
            if path.exists():
                samples.append({
                    "path": str(path),
                    "label": 0
                })

        # manipulated
        for path in manipulated_video_paths(id1, id2):
            if path.exists():
                samples.append({
                    "path": str(path),
                    "label": 1
                })

    return samples


def main():
    for split in ("train", "val", "test"):
        samples = generate_split(split)
        out_file = OUTPUT_FILES[split]

        with open(out_file, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"[✓] {split}: wrote {len(samples)} entries → {out_file}")


if __name__ == "__main__":
    # from collections import Counter

    # counter = Counter()

    # for split in ["train", "val", "test"]:
    #     pairs = load_pairs(SPLIT_FILES[split])
    #     for id1, id2 in pairs:
    #         for m in MANIPULATIONS:
    #             base = (
    #                 FF_ROOT
    #                 / "manipulated_sequences"
    #                 / m
    #                 / COMPRESSION
    #                 / "videos"
    #             )
    #             for name in (f"{id1}_{id2}.mp4", f"{id2}_{id1}.mp4"):
    #                 if (base / name).exists():
    #                     counter[m] += 1

    # print(counter)

    main()
