"""Explicit model fixture for local operator tests; CLI processes select their own model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crouch'))
import classconfig as cc
from models import FlowModel


@pytest.fixture(autouse=True)
def sa_model(monkeypatch):
  monkeypatch.setattr(cc, 'flow_model', FlowModel.SA)
