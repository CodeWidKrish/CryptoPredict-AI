"""Tests for Streamlit UI navigation, session state integrity, and error-free rendering."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def app_runner():
    """Initializes and runs the Streamlit application via AppTest."""
    app_path = (Path(__file__).parent.parent / "app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    return at


def test_app_initial_render_no_exceptions(app_runner):
    """Verifies that the Streamlit application starts with zero exceptions."""
    assert len(app_runner.exception) == 0, f"Exceptions on startup: {app_runner.exception}"
    assert app_runner.session_state.selected_page == "Trading Terminal"
    assert app_runner.session_state.show_menu_tab is True


def test_app_navigation_pills(app_runner):
    """Verifies that clicking every top pill transitions pages without StreamlitWidgetAlreadyInstantiatedError."""
    target_pages = [
        ("Leaderboard", "Model Leaderboard & Baseline Audit"),
        ("Walk-Forward", "Walk-Forward & Ablation Lab"),
        ("SHAP Lab", "Explainable AI (SHAP Lab)"),
        ("What-If Lab", "What-If Market Lab"),
        ("Backtest", "Strategy Backtest & Evidence Export"),
        ("Trading", "Trading Terminal"),
    ]

    for short_key, full_title in target_pages:
        pill_btn = app_runner.button(key=f"top_pill_{short_key}")
        assert pill_btn is not None, f"Pill button top_pill_{short_key} not found"
        pill_btn.click().run()
        assert len(app_runner.exception) == 0, f"Exception switching to {short_key}: {app_runner.exception}"
        assert app_runner.session_state.selected_page == full_title


def test_app_menu_toggle(app_runner):
    """Verifies that toggling the menu tab hides and shows without exceptions."""
    # Hide menu using freshly queried button
    app_runner.button(key="main_top_toggle_menu_btn").click().run()
    assert len(app_runner.exception) == 0
    assert app_runner.session_state.show_menu_tab is False

    # Navigate while hidden using freshly queried button
    app_runner.button(key="top_pill_Leaderboard").click().run()
    assert len(app_runner.exception) == 0
    assert app_runner.session_state.selected_page == "Model Leaderboard & Baseline Audit"

    # Show menu again using freshly queried button
    app_runner.button(key="main_top_toggle_menu_btn").click().run()
    assert len(app_runner.exception) == 0
    assert app_runner.session_state.show_menu_tab is True
    assert app_runner.session_state.selected_page == "Model Leaderboard & Baseline Audit"


def test_app_sidebar_buttons(app_runner):
    """Verifies that clicking sidebar navigation buttons transitions pages cleanly."""
    for page in ["Trading Terminal", "Walk-Forward & Ablation Lab", "Explainable AI (SHAP Lab)"]:
        btn = app_runner.button(key=f"sb_nav_btn_{page}")
        assert btn is not None
        btn.click().run()
        assert len(app_runner.exception) == 0
        assert app_runner.session_state.selected_page == page
