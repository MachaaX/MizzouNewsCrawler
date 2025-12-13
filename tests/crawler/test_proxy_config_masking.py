import os

from src.crawler.proxy_config import ProxyManager
from src.crawler.utils import mask_proxy_url


def test_proxy_manager_list_providers_masks_urls(monkeypatch):
    # Setup an environment with a standard proxy containing credentials
    env = {
        "STANDARD_PROXY_URL": "http://user:pass@standard.proxy:8080",
        "PROXY_PROVIDER": "standard",
    }
    monkeypatch.setenv("STANDARD_PROXY_URL", env["STANDARD_PROXY_URL"])
    monkeypatch.setenv("PROXY_PROVIDER", env["PROXY_PROVIDER"])

    manager = ProxyManager()
    providers = manager.list_providers()

    # Ensure the provider entry is masked
    assert "standard" in providers
    assert providers["standard"]["url"] == "http://user:***@standard.proxy:8080"
