import torch

from gen_ai.models.normalizing_flow import NICE


def test_nice():
    nice = NICE(4, 512)

    tmp_inp = torch.randn(4, 1, 28, 28)

    for i in range(4):
        if i % 2 == 0:
            assert nice.coupling_layer[i].mask_type == "odd"
        else:
            assert nice.coupling_layer[i].mask_type == "even"

    z, log_det_jacobian = nice(tmp_inp, reverse=False)

    assert z.size() == torch.Size([4, 28 * 28])
    assert log_det_jacobian.size() == torch.Size([])

    temp_z = torch.randn(4, 28 * 28)

    x, _ = nice(temp_z, reverse=True)

    assert x.size() == torch.Size([4, 28 * 28])
