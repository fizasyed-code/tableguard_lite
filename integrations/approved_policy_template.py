"""Connection point, NOT a supplied or working learned policy.

Fill this file only after inspecting the actual approved policy API. Implement
normalization, camera mapping, chunk/state reset and action decoding exactly as
that policy requires. Do not turn zero actions or replayed data into a fake VLA.

After implementing, factory would be:
  integrations.approved_policy_template:create_policy
Factories execute local Python. Audit any external code before importing it.
"""
from tableguard.phase2.bridge import Action, Observation


class ApprovedPolicy:
    def __init__(self, settings: dict):
        raise NotImplementedError(
            'No approved checkpoint/controller is connected. Supply the actual policy and documented action interface; '
            'this template does not fabricate robot behavior.'
        )

    def reset(self, instruction: str) -> None:
        """Reset model memory, cached action chunks and task progress per episode."""
        raise NotImplementedError

    def select_action(self, observation: Observation) -> Action:
        """Return decoded native MuJoCo controls, or explicit unscored stop.

        observation.rgb is RGB, NOT OpenCV BGR.
        Only selected robot-joint positions/velocities are supplied, never full
        simulator qpos, object body positions or evaluator disturbance labels.
        """
        raise NotImplementedError


def create_policy(settings: dict) -> ApprovedPolicy:
    return ApprovedPolicy(settings)
