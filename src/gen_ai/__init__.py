import argparse

from gen_ai.configs import ConfigMaker
from gen_ai.factory import GenAIModelFactory


def train() -> None:
    parser = argparse.ArgumentParser(description="Generative Model Train.")

    parser.add_argument(
        "model_type",
        type=str,
        choices=["autoregressive"],
        help="Generative Model Type.",
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="Config File Path.")
    parser.add_argument("-s", "--save_dir", type=str, default="./saved_model/", help="Save Directory.")

    args = parser.parse_args()

    config_maker = ConfigMaker(args.config)
    config = config_maker.make_config()

    model = GenAIModelFactory(model_type=args.model_type, config=config).make_model()
    model.train()
    model.save(args.save_dir)
