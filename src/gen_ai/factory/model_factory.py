from gen_ai.configs import GenAIConfig
from gen_ai.enums import ModelType
from gen_ai.models import GenAIModelBase
from dataclasses import asdict

class GenAIModelFactory:
    """Generative AI model factory."""

    def __init__(self, model_type: ModelType, config: GenAIConfig) -> None:
        """Initializes the model factory.

        Args:
            model_type (ModelType): model type
            config (GenAIConfig): configuration
        """
        self.model_type = model_type
        self.config = config

    def make_model(self) -> GenAIModelBase:
        """creates a model based on the model type.

        Raises:
            ValueError: Unsupported model type

        Returns:
            GenAIModelBase: model
        """
        if self.model_type == ModelType.autoregressive:
            from gen_ai.dataset import MNISTDataset
            from gen_ai.models.autoregressive import AutoregressiveModel
            from gen_ai.trainer.autoregressive_model_trainer import AutoregressiveModelTrainer

            module_config = asdict(self.config.model_config)
            module_config.pop("model_type")
            
            torch_module = AutoregressiveModel.torch_module_class(**self.config.model_config)
            trainer = AutoregressiveModelTrainer(MNISTDataset(self.config.data_config), self.config.train_config)
            sampler = None
            return AutoregressiveModel(torch_module, trainer, sampler)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
