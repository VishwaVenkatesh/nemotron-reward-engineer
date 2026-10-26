import jax
import jax.numpy as jp

import jax
import jax.numpy as jp

SCALES = {
    "hand_reach": 0.0,
    "hand_height": 3.0,
    "hand_rise": 1.5,
    "hand_wave_y": 4.0,
    "hand_above_head": 2.5,
    "elbow_up": 1.0,
}

def extra_rewards(env, data, action, info):
    pelvis = data.site_xpos[env._pelvis_imu_site_id]
    R = data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3)
    rh_world = data.site_xpos[env._hands_site_id[1]]
    rh_pelvis = R.T @ (rh_world - pelvis)
    hand_z = rh_pelvis[2]
    hand_y = rh_pelvis[1]

    # Estimated elbow height in pelvis frame (coupled to hand height)
    elbow_z_est = 0.57 + 0.3 * (hand_z - 0.7)

    z_rest = 0.022
    head_height = 0.648
    target_height = 0.70          # above head (0.648), below max (0.792)

    # 1) Precision height tracking: wider sigma (0.25) so gradient reaches all the way to target
    height_err = hand_z - target_height
    height_reward = jp.exp(-height_err**2 / (0.25**2))

    # 2) Linear rise shaping: 0 at rest, 1.0 at target_height
    rise_reward = jp.clip((hand_z - z_rest) / (target_height - z_rest), 0.0, 1.0)

    # 3) Height gate: waving only counts when hand is above 0.55m (lowered from 0.6)
    gate_x = 12.0 * (hand_z - 0.55)
    height_gate = 1.0 / (1.0 + jp.exp(-gate_x))

    # 4) Waving target: 1.0 Hz, center at shoulder line (-0.1), amplitude 0.25
    freq = 1.0
    y_center = -0.1
    amp = 0.25
    y_target = y_center + amp * jp.sin(2.0 * jp.pi * freq * data.time)
    wave_err = hand_y - y_target
    wave_reward = jp.exp(-wave_err**2 / (0.12**2))
    wave_reward_gated = wave_reward * height_gate

    # 5) Above-head bonus: sigmoid at 0.62m (lowered from 0.65) to trigger before head top
    above_x = 20.0 * (hand_z - 0.62)
    above_head_reward = 1.0 / (1.0 + jp.exp(-above_x))

    # 6) Elbow up encouragement: target elbow_z > 0.55 when hand is high
    elbow_target = 0.55
    elbow_err = elbow_z_est - elbow_target
    elbow_reward = jp.exp(-elbow_err**2 / (0.1**2)) * height_gate

    return {
        "hand_height": height_reward,
        "hand_rise": rise_reward,
        "hand_wave_y": wave_reward_gated,
        "hand_above_head": above_head_reward,
        "elbow_up": elbow_reward,
    }
