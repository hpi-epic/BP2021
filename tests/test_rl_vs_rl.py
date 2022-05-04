import pytest

from recommerce.rl.rl_vs_rl_training import train_rl_vs_rl


@pytest.mark.training
@pytest.mark.slow
def test_rl_vs_rl():
    train_rl_vs_rl(4, 230)
