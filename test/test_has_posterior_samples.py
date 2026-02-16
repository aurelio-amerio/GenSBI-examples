import pytest
from gensbi_examples.tasks import has_posterior_samples

@pytest.mark.parametrize(
    "task_name, expected",
    [
        ("two_moons", True),
        ("bernoulli_glm", True),
        ("gaussian_linear", True),
        ("gaussian_linear_uniform", True),
        ("gaussian_mixture", True),
        ("slcp", True),
        ("gravitational_waves", False),
        ("lensing", False),
        ("unknown_task", False),
    ],
)
def test_has_posterior_samples(task_name, expected):
    assert has_posterior_samples(task_name) == expected
