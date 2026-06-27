import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paz.models.foundation.gemma4.pretrained import shard_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Shard Gemma4 weights for upload")
    add = parser.add_argument
    add("--source_dir", required=True)
    add("--model_name", default="gemma4_2b")
    add("--output_dir", required=True)
    add("--repo", default="oarriaga/altamira-data")
    add("--release", default=None)
    args = parser.parse_args()
    manifest_path = shard_weights(args.source_dir, args.model_name,
                                  args.output_dir)
    print("wrote manifest", manifest_path)
    assets = sorted(Path(args.output_dir).glob("{}*".format(args.model_name)))
    print("{} assets ready in {}".format(len(assets), args.output_dir))
    if args.release is not None:
        command = ["gh", "release", "upload", args.release, "--repo", args.repo]
        command = command + [str(asset) for asset in assets]
        subprocess.run(command, check=True)
        print("uploaded to", args.release)
