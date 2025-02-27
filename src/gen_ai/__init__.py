import argparse

from gen_ai.configs import GenAIConfig
from gen_ai.factory import GenAIModelFactory


def train() -> None:
    parser = argparse.ArgumentParser(description="Generative Model Train.")

    parser.add_argument("-c", "--config", type=str, required=True, help="Config File Path.")
    parser.add_argument("-s", "--save_dir", type=str, default="./saved_model/", help="Save Directory.")

    args = parser.parse_args()

    config = GenAIConfig(args.config)

    model = GenAIModelFactory(config=config).make_model()
    model.train()
    model.save(args.save_dir)
