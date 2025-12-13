from src.crawler.utils import mask_proxy_url


def test_mask_proxy_url_masks_password():
    url = "https://user:pass@unblock.decodo.com:60000"
    assert mask_proxy_url(url) == "https://user:***@unblock.decodo.com:60000"


def test_mask_proxy_url_no_credentials():
    url = "https://unblock.decodo.com:60000"
    assert mask_proxy_url(url) == "https://unblock.decodo.com:60000"


def test_mask_proxy_url_none_returns_none():
    assert mask_proxy_url(None) is None
