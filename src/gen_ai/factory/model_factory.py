from gen_ai.configs import GenAIConfig
from gen_ai.enums import ModelType
from gen_ai.models import GenAIModelBase


class GenAIModelFactory:
    """Generative AI model factory."""

    def __init__(self, config: GenAIConfig) -> None:
        """Initializes the model factory.

        Args:
            config (GenAIConfig): configuration
        """
        self.config = config

    def make_model(self) -> GenAIModelBase:
        """creates a model based on the model type.

        Raises:
            ValueError: Unsupported model type

        Returns:
            GenAIModelBase: model
        """
        if self.config.model_type == ModelType.autoregressive:
            from gen_ai.dataset import MNISTDataset
            from gen_ai.models.autoregressive import AutoregressiveModel
            from gen_ai.trainer.autoregressive_model_trainer import AutoregressiveModelTrainer

            torch_module = AutoregressiveModel.torch_module_class(**self.config.module_config)
            dataset = MNISTDataset(self.config.data_config)
            trainer = AutoregressiveModelTrainer(dataset, self.config.train_config)
            sampler = None
            return AutoregressiveModel(torch_module, trainer, sampler)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
