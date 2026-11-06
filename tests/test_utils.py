import torch

from vial_classifier.utils import get_device, print_section_header


def test_get_device_returns_torch_device():
    assert isinstance(get_device(), torch.device)


def test_print_section_header_prints_title(capsys):
    print_section_header("TEST")
    assert "TEST" in capsys.readouterr().out