"""Basic tests for DDP-PBT package"""

import ddp_pbt


def test_version():
    """Test that version is defined"""
    assert hasattr(ddp_pbt, "__version__")
    assert isinstance(ddp_pbt.__version__, str)


def test_import():
    """Test that package can be imported"""
    assert ddp_pbt is not None
