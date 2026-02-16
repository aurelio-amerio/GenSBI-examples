import pytest
from unittest.mock import patch
from gensbi_examples.tasks import GravitationalLensing


def test_lensing_get_reference_raises_not_implemented_error():
    # Mock __init__ to avoid side effects (downloads, file reading) during instantiation
    with patch.object(GravitationalLensing, '__init__', return_value=None):
        # Instantiate the class without running the real __init__
        task = GravitationalLensing()

        # Verify that get_reference raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Reference posterior samples not available for this task."):
            task.get_reference()
