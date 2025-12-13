import os
import requests
from src.crawler.origin_proxy import enable_origin_proxy
from src.crawler.utils import mask_proxy_url


class DummyResponse:
    def __init__(self):
        self.status_code = 200


def test_enable_origin_proxy_masks_response_proxy_url(monkeypatch):
    monkeypatch.setenv("ORIGIN_PROXY_URL", "http://user:pass@proxy.example.com:1234")
    monkeypatch.setenv("PROXY_USERNAME", "user")
    monkeypatch.setenv("PROXY_PASSWORD", "pass")
    monkeypatch.setenv("USE_ORIGIN_PROXY", "true")

    session = requests.Session()
    enable_origin_proxy(session)

    # Replace original request with a dummy that returns DummyResponse
    session._origin_original_request = lambda method, url, *args, **kwargs: DummyResponse()

    resp = session.get("https://example.com/test")

    # Ensure proxy metadata attached and URL is masked
    assert getattr(resp, "_proxy_used", False) is True
    assert resp._proxy_url == mask_proxy_url("http://user:pass@proxy.example.com:1234")
    assert resp._proxy_authenticated is True
