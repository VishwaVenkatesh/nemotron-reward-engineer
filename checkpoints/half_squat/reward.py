import jax
import jax.numpy as jp

SCALES = {
    "pelvis_height": 2.0,
    "pelvis_vertical_vel": -0.5,
    "arm_height_symmetry": 0.5,
    "hand_reach": 0.0,  # disable conflicting reach term
}

def extra_rewards(env, data, action, info):
    pelvis = data.site_xpos[env._pelvis_imu_site_id]
    R = data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3)
    
    # 1. Pelvis height target (15 cm below nominal 0.672)
    target_height = 0.522
    height_err = pelvis[2] - target_height
    pelvis_height_reward = jp.exp(-height_err**2 / 0.1)
    
    # 2. Penalize vertical velocity of pelvis (use base linear vel z as proxy)
    vel_z = data.qvel[2]
    pelvis_vertical_vel_penalty = vel_z**2
    
    # 3. Arm height symmetry in pelvis frame
    left_hand = data.site_xpos[env._hands_site_id[0]]
    right_hand = data.site_xpos[env._hands_site_id[1]]
    left_local = R.T @ (left_hand - pelvis)
    right_local = R.T @ (right_hand - pelvis)
    height_diff = right_local[2] - left_local[2]
    arm_height_symmetry_reward = jp.exp(-height_diff**2 / 0.05)
    
    return {
        "pelvis_height": pelvis_height_reward,
        "pelvis_vertical_vel": pelvis_vertical_vel_penalty,
        "arm_height_symmetry": arm_height_symmetry_reward,
    }
