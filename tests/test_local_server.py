"""Tests for the local static-file server helper used to give fixture pages a real http://
origin (instead of file://) so the domain-based allowlist has something meaningful to check."""

import urllib.error
import urllib.request

from capability_forge.utils.local_server import serve_directory


def test_serves_a_file_from_the_given_directory(tmp_path):
    (tmp_path / "hello.html").write_text("<h1>Hello</h1>")
    with serve_directory(tmp_path) as base_url:
        assert base_url.startswith("http://127.0.0.1:")
        response = urllib.request.urlopen(f"{base_url}/hello.html")
        assert response.status == 200
        assert response.read() == b"<h1>Hello</h1>"


def test_missing_file_returns_404(tmp_path):
    with serve_directory(tmp_path) as base_url:
        try:
            urllib.request.urlopen(f"{base_url}/does_not_exist.html")
            assert False, "expected a 404"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404


def test_serves_the_real_fixture_directory():
    from pathlib import Path

    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    with serve_directory(fixtures_dir) as base_url:
        response = urllib.request.urlopen(f"{base_url}/hostile_legacy_page.html")
        body = response.read().decode()
        assert "First Fidelity" in body


def test_concurrent_servers_get_different_ports(tmp_path):
    with serve_directory(tmp_path) as first_url, serve_directory(tmp_path) as second_url:
        assert first_url != second_url
