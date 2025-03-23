import torch

from gen_ai.models.normalizing_flow import RealNVP


def test_real_nvp():
    real_nvp = RealNVP(4)

    tmp_inp = torch.randn(4, 1, 28, 28)

    z, log_det_jacobian = real_nvp(tmp_inp, reverse=True)

    assert z.size() == torch.Size([4, 28 * 28])
    assert log_det_jacobian.size() == torch.Size([4])

    temp_z = torch.randn(4, 28 * 28)

    x, _ = real_nvp(temp_z, reverse=False)

    assert x.size() == torch.Size([4, 28 * 28])
