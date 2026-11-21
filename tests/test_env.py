import os

from mosaic.env import load_env, parse_env


def test_parse_handles_comments_quotes_and_export():
    text = '# a comment\nA=1\nexport B="two words"\nC=\'x=y\'\n\nbad line\n'
    assert parse_env(text) == {"A": "1", "B": "two words", "C": "x=y"}


def test_load_sets_missing_variables_only(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("MOSAIC_TEST_NEW=from-file\nMOSAIC_TEST_SET=from-file\n")
    monkeypatch.setenv("MOSAIC_TEST_SET", "from-shell")
    monkeypatch.delenv("MOSAIC_TEST_NEW", raising=False)
    assert load_env([f]) == ["MOSAIC_TEST_NEW"]
    assert os.environ["MOSAIC_TEST_NEW"] == "from-file" and os.environ["MOSAIC_TEST_SET"] == "from-shell"
    monkeypatch.delenv("MOSAIC_TEST_NEW")


def test_no_file_is_fine(tmp_path):
    assert load_env([tmp_path / "missing.env"]) == []


def test_env_file_is_ignored_by_git():
    from pathlib import Path
    assert ".env" in (Path(__file__).resolve().parents[1] / ".gitignore").read_text().splitlines()
