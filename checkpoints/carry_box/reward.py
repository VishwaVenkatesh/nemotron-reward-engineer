import jax
import jax.numpy as jp

# Reward terms for dual-hand box-carrying walk.
# Ideas from research:
# - bilateral_contact gating (Reward Design Agent, arxiv:2606.01672): both hands must be in "contact" (together) to earn full credit.
# - stage-specific rewards (Sim-to-Real Box Loco-Manipulation, arxiv:2310.03191): transport phase rewards forward progress only when embrace is maintained.
# - anti-phase arm-leg coordination (Gait-Conditioned RL, UCL 2025): arm swing opposite to leg swing stabilizes angular momentum.

SCALES = {
    # Override base reach term to avoid conflict with new symmetric target
    "hand_reach": 0.0,
    # Symmetric hand targets in pelvis frame (x forward, y left, z up)
    "left_hand_target": 1.5,
    "right_hand_target": 1.5,
    # Hands close together (embrace)
    "hands_together": 2.0,
    # Left/right symmetry around midline
    "hands_symmetry": 1.0,
    # Chest-height maintenance (dense shaping from rest pose)
    "hands_chest_height": 1.2,
    # Forward progress gated by embrace quality
    "gated_velocity": 1.0,
    # Anti-phase arm swing coordination
    "arm_leg_coordination": 0.8,
    # Upright torso while carrying (prevent excessive lean)
    "carry_torso_upright": 0.7,
}

def extra_rewards(env, data, action, info):
    # Pelvis frame
    pelvis = data.site_xpos[env._pelvis_imu_site_id]
    R = data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3)
    fwd = R[:, 0]
    left = R[:, 1]
    up = R[:, 2]

    # Hand positions in pelvis frame
    lh_w = data.site_xpos[env._hands_site_id[0]]
    rh_w = data.site_xpos[env._hands_site_id[1]]
    lh = R.T @ (lh_w - pelvis)
    rh = R.T @ (rh_w - pelvis)

    # Target positions for "carrying a box in front of chest"
    # Based on measured geometry: chest ~0.38m above pelvis, 0.3m forward, hands ~0.1m apart laterally
    target_x = 0.30
    target_z = 0.38
    target_y_abs = 0.10

    lh_target = jp.array([target_x,  target_y_abs, target_z])
    rh_target = jp.array([target_x, -target_y_abs, target_z])

    # Tracking errors
    lh_err = lh - lh_target
    rh_err = rh - rh_target
    lh_err_sq = jp.sum(lh_err ** 2)
    rh_err_sq = jp.sum(rh_err ** 2)

    # Wide sigma so reward is non-zero from rest pose (rest error ~0.4-0.5m)
    sigma_pos = 0.35
    left_hand_target = jp.exp(-lh_err_sq / (2 * sigma_pos ** 2))
    right_hand_target = jp.exp(-rh_err_sq / (2 * sigma_pos ** 2))

    # Hands together: distance between hands (target ~0.2m apart)
    hands_vec = lh - rh
    hands_dist = jp.linalg.norm(hands_vec)
    target_dist = 0.20
    hands_together = jp.exp(-(hands_dist - target_dist) ** 2 / (2 * 0.15 ** 2))

    # Symmetry: y-coordinates opposite, x and z matched
    sym_err = (lh[0] - rh[0]) ** 2 + (lh[1] + rh[1]) ** 2 + (lh[2] - rh[2]) ** 2
    hands_symmetry = jp.exp(-sym_err / (2 * 0.1 ** 2))

    # Dense chest-height shaping: grows steadily as hands rise from rest (~ -0.02, 0.02) to target (0.38)
    # Use mean hand height
    mean_hand_z = 0.5 * (lh[2] + rh[2])
    # Sigmoid from rest (~0.0) to target (0.38), midpoint ~0.2, scale ~0.15
    hands_chest_height = jax.nn.sigmoid((mean_hand_z - 0.20) / 0.15)

    # Embrace quality gate (product of together + symmetry + height)
    embrace = hands_together * hands_symmetry * hands_chest_height

    # Forward velocity tracking gated by embrace
    # Base env already has velocity command; we add a bonus when embrace is good
    vel_cmd = info.get("vel_cmd", jp.array([1.0, 0.0, 0.0]))  # default forward 1 m/s
    base_vel = data.qvel[0:3]
    vel_err = base_vel - vel_cmd
    vel_err_sq = jp.sum(vel_err ** 2)
    vel_track = jp.exp(-vel_err_sq / (2 * 0.5 ** 2))
    gated_velocity = embrace * vel_track

    # Anti-phase arm-leg coordination
    # Leg phase from base env info or approximate from pelvis velocity
    # Use pelvis forward velocity as phase proxy (sinusoidal at walk frequency ~1Hz)
    walk_freq = 1.0
    phase = 2 * jp.pi * walk_freq * data.time
    # Desired arm swing: opposite to leg (anti-phase). When right leg forward (pelvis vel +), left arm forward, right arm back.
    # In pelvis frame, arm forward/back is x. We want left hand x - right hand x ~ -sin(phase) (anti-phase)
    arm_swing = lh[0] - rh[0]
    desired_swing = -0.15 * jp.sin(phase)  # amplitude ~0.15m
    arm_leg_coordination = jp.exp(-(arm_swing - desired_swing) ** 2 / (2 * 0.1 ** 2))

    # Torso upright while carrying: pelvis up vector aligned with world up
    torso_upright = jp.clip(up[2], 0.0, 1.0)  # cos(tilt)
    carry_torso_upright = torso_upright ** 2

    return {
        "left_hand_target": left_hand_target,
        "right_hand_target": right_hand_target,
        "hands_together": hands_together,
        "hands_symmetry": hands_symmetry,
        "hands_chest_height": hands_chest_height,
        "gated_velocity": gated_velocity,
        "arm_leg_coordination": arm_leg_coordination,
        "carry_torso_upright": carry_torso_upright,
    }
