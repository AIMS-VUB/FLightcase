import os
import json
import argparse


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def to_json(dict_to_save, output_path):
    with open(output_path, 'w') as f:
        json.dump(dict_to_save, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_paths", nargs="+")
    parser.add_argument("--path_output_dir")
    parser.add_argument("--train_or_val")
    args = parser.parse_args()

    # Read keys from JSON
    random_states = load_json(args.json_paths[0]).keys()

    total_train_val_splits = {rs: [] for rs in random_states}

    for json_path in args.json_paths:
        json_to_add = load_json(json_path)
        for rs in json_to_add.keys():
            total_train_val_splits[rs] += json_to_add[rs]

    to_json(total_train_val_splits, os.path.join(args.path_output_dir, f'{args.train_or_val}_ids.json'))
