from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class ReachWithGripper(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        get_fingers_width,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
        radius_range=0.02
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.get_fingers_width = get_fingers_width
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.radius_range_low = 0.02
        self.radius_range_high = self.radius_range_low + radius_range
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        fingers_width = np.array([self.get_fingers_width()])
        return np.concatenate([ee_position, fingers_width])

    def reset(self) -> None:
        self.goal = self._sample_goal()
        if "target" in self.sim._bodies_idx:  # body registered earlier
            uid = self.sim._bodies_idx.pop("target")  # remove entry from dict
            self.sim.physics_client.removeBody(uid)  # destroy the body in Bulle
        self.sim.create_sphere(
            body_name="target",
            radius=self.goal[-1],
            mass=0.0,
            ghost=False,
            position=self.goal[:3],
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        #self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.concatenate([
            self.np_random.uniform(self.goal_range_low, self.goal_range_high),
            np.array([self.np_random.uniform(self.radius_range_low, self.radius_range_high)])
        ])
        goal[2] = goal[-1] / 2
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        elif self.reward_type == "combined":
            return -d.astype(np.float32) + 100 * np.array(d < self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)