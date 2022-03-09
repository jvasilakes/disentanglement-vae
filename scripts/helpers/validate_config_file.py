import json
import argparse

from vae import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


def main(config_file):
    config = json.load(open(config_file))
    utils.validate_params(config)
    print("Config file is valid")


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
