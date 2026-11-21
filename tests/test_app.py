"""Smoke test of the Streamlit app (skipped when Streamlit is not installed)."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_file_compiles():
    compile(APP.read_text(), str(APP), "exec")


@pytest.mark.needs_streamlit
def test_demo_run_in_the_app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Acme Robotics" in s.value for s in at.subheader)
