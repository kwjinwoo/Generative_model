import os

from gen_ai.models.normalizing_flow import StandardLogisticDistribution
from gen_ai.sampler.normalizing_flow_model_sampler import NormalizingFlowModelSampler


def test_sample(test_normalizing_model, test_dataset, tmp_path):
    save_dir = str(tmp_path / "saved_model")
    num_samples = 4
    sampler = NormalizingFlowModelSampler()
    sampler.prior = StandardLogisticDistribution(28 * 28, "cpu")

    sampler.sample(test_normalizing_model, test_dataset, save_dir, num_samples)

    assert os.path.isfile(os.path.join(save_dir, "NICE_generated.png")) is True
    assert os.path.isfile(os.path.join(save_dir, "NICE_interpolate.png")) is True
