import argparse

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jit-fp", type=str)
    args = parser.parse_args()

    model: torch.jit.ScriptModule = torch.jit.load(args.jit_fp, map_location="cpu")

    for n, v in model.named_modules():
        print(n, v.original_name)
