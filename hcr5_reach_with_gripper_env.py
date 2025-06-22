from typing import Optional

import numpy as np
import pybullet as p

from gymnasium import spaces
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from hcr_5 import HCR5
from reach_with_gripper import ReachWithGripper


class HCR5ReachWithGripperEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        # 1) assign these *before* super().__init__
        self.render_width  = render_width
        self.render_height = render_height
        # if the user passed None, give a default
        self.render_target_position = (
            np.array(render_target_position)
            if render_target_position is not None
            else np.array([0.0, 0.0, 0.0])
        )
        self.render_distance = render_distance
        self.render_yaw      = render_yaw
        self.render_pitch    = render_pitch
        self.render_roll     = render_roll
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = HCR5(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type
        )
        task = ReachWithGripper(
            sim,
            reward_type=reward_type,
            get_ee_position=robot.get_ee_position,
            get_fingers_width=robot.get_fingers_width,
        )
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
        self.observation_space['observation'] = spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8)

    def _get_obs(self) -> Optional[np.ndarray]:
        # 1) grab the RGB image H×W×3
        self.sim.physics_client.connection_mode = p.DIRECT
        img = self.sim.render(
            width=96,
            height=96,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )
        self.sim.physics_client.connection_mode = p.GUI

        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return {
            "observation": img.astype(np.uint8),
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal().astype(np.float32),
        }
