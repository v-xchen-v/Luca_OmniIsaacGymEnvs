from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omni.isaac.gym.vec_env import VecEnvBase

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage

from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

import torch
import numpy as np

class FrankaFollowTargetTask(RLTask):
    def __init__(
        self, 
        name, 
        sim_config: SimConfig,
        env: VecEnvBase, 
        offset=None
    ) -> None:
        # settings from config files
        # call to init _cfg and _task_cfg by config fand should go before the super init
        self.update_config(sim_config)

        # settings in scripts
        # 7 actions for 7 DOFs in arm, ignore the 2 DOFs in gripper for the following target task
        self._num_actions = 7

        self._num_observations = 32
        # 7: dofbot joints position (action space)
        # 7: dofbot joints velocity
        # 3: goal position
        # 4: goal rotation
        # 4: goal relative rotation
        # 7: previous action

        #???
        self._num_states = 0

        # calling the initialization, combine the defaults and config settings, no as default.
        # call the parent class contructor to initialize key RL variables
        super().__init__(name, env, offset)


    def update_config(self, sim_config: SimConfig):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # configs ifrom task configs
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]


    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()

        # first create a single environment
        self.get_franka()

        # call the parent class to clone the single environment
        super().set_up_scene(scene)

        # construct an ArticulationView object to hold our collection of environments
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        # register the ArticulationView object to the world, so that it can be initialized
        scene.add(self._frankas)
        return

    def get_observations(self) -> dict:
        # dummy observation
        self.obs_buf = torch.zeros(
            (self._num_envs, self._num_observations), device=self.device, dtype=torch.float)
            # self._num_observations, device=self.device, dtype=torch.float)
        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        pass

    def is_done(self):
        pass

    def calculate_metrics(self):
        reward=np.random.random(1)[0]
        self.rew_buf[:] = reward

    #Ref: omniisaacgymenvs/tasks/franka_deformable.py
    def get_franka(self):
        # add a single robot to the stage
        franka = Franka(
            prim_path=self.default_zero_env_path + "/franka", 
            name="franka", 
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            translation=torch.tensor([0.0, 0.0, 0.0]),
        )

        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )
        franka.set_franka_properties(stage=self._stage, prim=franka.prim) 