import socket

import pytest


def test_guard_blocks_outbound_and_records_it(no_network_attempts):
    with pytest.raises(OSError):
        socket.create_connection(("api.openai.com", 443), timeout=2)
    assert no_network_attempts == ["getaddrinfo api.openai.com"]
    no_network_attempts.clear()  # expected attempt; keep the teardown assertion for real leaks


def test_guard_allows_loopback():
    a, b = socket.socketpair()
    a.close()
    b.close()
