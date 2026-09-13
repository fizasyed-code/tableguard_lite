"""Contracts only: deliberately no fake SO-101 or VLA implementation."""
from typing import Protocol
from .supervisor import Decision

class ApprovedRobotAdapter(Protocol):
    def at_supported_boundary(self) -> bool: ...
    def dispatch_valid_joint_repair(self, decision: Decision) -> str: ...
    def request_supported_hold(self) -> None: ...

class UnconfiguredRobotAdapter:
    def at_supported_boundary(self) -> bool:
        return False
    def dispatch_valid_joint_repair(self, decision: Decision) -> str:
        raise NotImplementedError('Connect the organizer-approved dual-SO-101 policy/action adapter first.')
    def request_supported_hold(self) -> None:
        raise NotImplementedError('No robot controller is connected; no hold/stop has been executed.')
