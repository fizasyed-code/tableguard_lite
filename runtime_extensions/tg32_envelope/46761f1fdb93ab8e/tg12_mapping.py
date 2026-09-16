"""Unmodified scalar actuator validation from the supplied 07C worker. No planner."""
import math
import numpy as np
JOINT_SUFFIXES=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")

def validate_mapping(model, data, mj):
    """Reject incompatible actuators rather than assume index/unit semantics."""
    expected = {f"{side}_{joint}" for side in ("left", "right")
                for joint in JOINT_SUFFIXES}
    nactuator = len(model.actuator_trntype)
    if nactuator != 12 or int(model.nu) != 12 or len(data.ctrl) != 12:
        raise ValueError("This test requires the inspected 12 scalar actuators.")
    items = []
    used_controls, used_joints, names = set(), set(), set()
    for aid in range(nactuator):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, aid)
        if name not in expected or name in names:
            raise ValueError(f"Unexpected/duplicate actuator name: {name}")
        names.add(name)
        if int(model.actuator_trntype[aid]) != int(mj.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"Not a joint transmission: {name}")
        jid = int(model.actuator_trnid[aid, 0])
        if not 0 <= jid < int(model.njnt):
            raise ValueError(f"Invalid joint ID: {name}")
        jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        if jname != name or jid in used_joints:
            raise ValueError(f"Unexpected actuator-to-joint mapping: {name} -> {jname}")
        used_joints.add(jid)
        if int(model.jnt_type[jid]) != int(mj.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected a hinge joint: {name}")
        # Newer MuJoCo versions expose separate control addresses/counts.
        cid = int(model.actuator_ctrladr[aid]) if hasattr(model, "actuator_ctrladr") else aid
        num = int(model.actuator_ctrlnum[aid]) if hasattr(model, "actuator_ctrlnum") else 1
        if num != 1 or not 0 <= cid < len(data.ctrl) or cid in used_controls:
            raise ValueError(f"Non-scalar/duplicate control address: {name}")
        used_controls.add(cid)
        if not np.allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0], atol=1e-9, rtol=0):
            raise ValueError(f"Non-unit gear requires a different control mapping: {name}")
        if int(model.actuator_dyntype[aid]) != int(mj.mjtDyn.mjDYN_NONE):
            raise ValueError(f"Filtered/dynamic actuators are outside this test: {name}")
        if (int(model.actuator_gaintype[aid]) != int(mj.mjtGain.mjGAIN_FIXED)
                or int(model.actuator_biastype[aid]) != int(mj.mjtBias.mjBIAS_AFFINE)):
            raise ValueError(f"Not an unfiltered fixed-gain position servo: {name}")
        kp = float(model.actuator_gainprm[aid, 0])
        bp = np.asarray(model.actuator_biasprm[aid])
        if (not math.isfinite(kp) or kp <= 0 or not np.isfinite(bp).all()
                or not np.isclose(bp[0], 0, atol=1e-9)
                or not np.isclose(bp[1], -kp, atol=1e-6)
                or bp[2] > 1e-9 or not np.allclose(bp[3:], 0)):
            raise ValueError(f"Unexpected position-servo coefficients: {name}")
        if not np.allclose(model.actuator_gainprm[aid, 1:], 0):
            raise ValueError(f"Unexpected gain parameters: {name}")
        if not model.actuator_ctrllimited[aid] or not model.jnt_limited[jid]:
            raise ValueError(f"Explicit joint and control ranges required: {name}")
        cr = np.asarray(model.actuator_ctrlrange[cid], dtype=float)
        jr = np.asarray(model.jnt_range[jid], dtype=float)
        if not np.isfinite(cr).all() or not np.isfinite(jr).all():
            raise ValueError(f"Non-finite limits: {name}")
        lo, hi = max(float(cr[0]), float(jr[0])), min(float(cr[1]), float(jr[1]))
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        q0 = float(data.qpos[qadr])
        if not math.isfinite(q0) or not lo + 0.01 < q0 < hi - 0.01:
            raise ValueError(f"Initial position is too close to/outside limits: {name}")
        items.append({"name": name, "aid": aid, "cid": cid, "jid": jid,
                      "qadr": qadr, "vadr": vadr, "q0": q0,
                      "low": lo, "high": hi, "kp": kp,
                      "kv": float(-bp[2]), "control_range": cr.tolist(),
                      "joint_range": jr.tolist()})
    if names != expected or used_controls != set(range(12)):
        raise ValueError("Incomplete actuator/control mapping.")
    return items
