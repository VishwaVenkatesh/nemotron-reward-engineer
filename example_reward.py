import jax
import jax.numpy as jp

# Hand-written example of the format Nemotron must produce.
# Raises the right hand above the head and waves it side to side while walking.
SCALES = {"hand_reach": 0.0, "wave_height": 1.5, "wave_motion": 1.0}


def extra_rewards(env, data, action, info):
    pelvis = data.site_xpos[env._pelvis_imu_site_id]
    R = data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3)
    hand = R.T @ (data.site_xpos[env._hands_site_id[1]] - pelvis)  # right hand, pelvis frame
    height_err = hand[2] - 0.55
    target_y = -0.25 + 0.15 * jp.sin(2 * jp.pi * 1.0 * data.time)
    return {
        "wave_height": jp.exp(-height_err ** 2 / 0.02),
        "wave_motion": jp.exp(-(hand[1] - target_y) ** 2 / 0.01),
    }
