"""NIST RMF-aligned deployment modes for graduated autonomy.

This module implements three deployment modes that support the graduated
autonomy principle from NIST AI Risk Management Framework:

- SHADOW: Log-only mode for validation and trust building
- GUARDED: Human-in-the-loop confirmation required
- EXPANDED: Autonomous execution of approved actions

Each mode enforces different levels of human oversight, enabling
organizations to progressively increase autonomy as confidence grows.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from anstkit.schemas import ControlAction, PlantState


class DeploymentMode(str, Enum):
    """Graduated autonomy levels per NIST AI RMF.

    These modes support the "graduated autonomy" principle where
    AI systems start with maximum oversight and progressively
    gain more autonomous capabilities as trust is established.
    """

    SHADOW = "shadow"      # Log only, no execution
    GUARDED = "guarded"    # Execute with human confirmation
    EXPANDED = "expanded"  # Auto-execute approved actions


@dataclass
class ModeResult:
    """Result of mode controller processing.

    Attributes:
        would_execute: Whether the action would be executed if allowed.
        actually_executed: Whether the action was actually executed.
        awaiting_confirmation: Whether waiting for human confirmation.
        rejection_reason: Reason for rejection if not approved.
    """

    would_execute: bool
    actually_executed: bool
    awaiting_confirmation: bool = False
    rejection_reason: Optional[str] = None


class ShadowResult(ModeResult):
    """Shadow mode result - always sets actually_executed=False.

    Shadow mode is used during initial deployment to validate
    the system's decision-making without executing any actions.
    All proposed actions are logged but never executed.
    """
    pass


@dataclass
class ModeController:
    """Controls action execution based on deployment mode.

    The ModeController is the final gatekeeper before action execution.
    Even if the triad approves an action, the ModeController determines
    whether it should actually be executed based on the current
    deployment mode and human confirmation status.

    Attributes:
        mode: The current deployment mode.
    """

    mode: DeploymentMode

    def process(
        self,
        state: PlantState,
        actions: List[ControlAction],
        approved: bool,
        human_confirmed: bool = False,
    ) -> ModeResult:
        """Process actions according to current deployment mode.

        Args:
            state: Current plant state.
            actions: Proposed actions from the triad.
            approved: Whether triad approved the actions.
            human_confirmed: Whether human operator confirmed (for GUARDED mode).

        Returns:
            ModeResult with execution status.

        Raises:
            ValueError: If an unknown deployment mode is encountered.
        """
        if self.mode == DeploymentMode.SHADOW:
            return ShadowResult(
                would_execute=approved,
                actually_executed=False,
            )

        elif self.mode == DeploymentMode.GUARDED:
            if not approved:
                return ModeResult(
                    would_execute=False,
                    actually_executed=False,
                    rejection_reason="Triad rejected actions",
                )
            if not human_confirmed:
                return ModeResult(
                    would_execute=True,
                    actually_executed=False,
                    awaiting_confirmation=True,
                )
            return ModeResult(
                would_execute=True,
                actually_executed=True,
            )

        elif self.mode == DeploymentMode.EXPANDED:
            if not approved:
                return ModeResult(
                    would_execute=False,
                    actually_executed=False,
                    rejection_reason="Triad rejected actions",
                )
            return ModeResult(
                would_execute=True,
                actually_executed=True,
            )

        raise ValueError(f"Unknown mode: {self.mode}")
