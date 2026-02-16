import pytest
from unittest.mock import patch, MagicMock
from gensbi_examples.tasks import get_task, Task


def test_invalid_kind_error():
    """
    Test that initializing a Task with an invalid kind raises a ValueError.
    We mock the external dependencies to avoid downloading datasets.
    """

    # Mock data to satisfy the initialization before the kind check
    mock_metadata = {
        "two_moons": {
            "dim_cond": 2,
            "dim_obs": 2
        }
    }

    with patch("gensbi_examples.tasks.hf_hub_download") as mock_download, \
         patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("json.load") as mock_json_load, \
         patch("gensbi_examples.tasks.load_dataset") as mock_load_dataset:

        # Setup mocks
        mock_download.return_value = "dummy_path"
        # Mocking open context manager
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_json_load.return_value = mock_metadata

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = MagicMock()  # for ["train"], etc.
        mock_load_dataset.return_value = mock_dataset

        # The actual test
        with pytest.raises(ValueError, match="Unknown kind: invalid_kind"):
            get_task("two_moons", kind="invalid_kind")

        # Verify that the check happens inside Task.__init__
        # We can also test directly instantiating Task if we want, but get_task is the public API
        with pytest.raises(ValueError, match="Unknown kind: invalid_kind"):
            Task("two_moons", kind="invalid_kind")
