import argparse

from gen_ai.configs import ConfigMaker


def train() -> None:
    parser = argparse.ArgumentParser(description="Generative Model Train.")

    parser.add_argument("-c", "--config", type=str, required=True, help="Config File Path.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to Train Model.")
    parser.add_argument("-s", "--save_dir", type=str, default="./saved_model/", help="Save Directory.")

    args = parser.parse_args()

    config_maker = ConfigMaker(args.config)
    config = config_maker.make_config()
