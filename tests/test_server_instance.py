from pathlib import Path

import pytest

from run_analysis.cli import _server_instance_lock


def test_only_one_server_instance_can_own_a_database(tmp_path: Path) -> None:
    database = tmp_path / "data" / "coach.sqlite"

    with _server_instance_lock(database):
        with pytest.raises(BlockingIOError):
            with _server_instance_lock(database):
                pass

    # Releasing the first process lock makes an ordinary restart legal.
    with _server_instance_lock(database):
        pass
