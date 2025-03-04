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


def sample() -> None:
    parser = argparse.ArgumentParser(description="Generative Model Sample.")

    parser.add_argument("-m", "--model_path", type=str, required=True, help="Model Path.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config File Path.")
    parser.add_argument("-s", "--save_dir", type=str, default="./assets/", help="Save Directory.")
    parser.add_argument("-n", "--num_samples", type=int, default=16, help="Number of samples.")

    args = parser.parse_args()

    config = GenAIConfig(args.config)
    model = GenAIModelFactory(config=config).make_model()
    model.load(args.model_path)
    model.sample(save_dir=args.save_dir, num_samples=args.num_samples)
