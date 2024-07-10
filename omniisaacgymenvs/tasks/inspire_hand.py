# Ref: omniisaacgymenvs/tasks/shadow_hand.py

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omni.isaac.gym.vec_env import VecEnvBase

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage

from omniisaacgymenvs.robots.articulations.inspire_hand import InspireHandR
from omniisaacgymenvs.robots.articulations.views.inspire_hand_view import InspireHandView

from omniisaacgymenvs.tasks.shared.in_hand_manipulation import InHandManipulationTask

from omni.isaac.core.utils.torch import *

import torch
import numpy as np

class InspireHandRotateCubeTask(InHandManipulationTask):
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
        self._num_actions = 6

        self.obs_type = self._task_cfg["env"]["observationType"]
        if self.obs_type == "full_no_vel":
            self._num_observations = 36
        else:
            self._num_observations = 42
        # 6: inspire hand joints position (action space)
        # 6: inspire hand joints velocity
        # 3: goal position
        # 4: goal rotation
        # 4: goal relative rotation
        # 6: previous action

        #???
        self._num_states = 0
        
        InHandManipulationTask.__init__(self, name=name, env=env)


        # calling the initialization, combine the defaults and config settings, no as default.
        # call the parent class contructor to initialize key RL variables
        # super().__init__(name, env, offset)


    def update_config(self, sim_config: SimConfig):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # configs ifrom task configs
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        
        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["full_no_vel", "full"]):
            raise Exception("Unknown type of observations!\nobservationType should be one of: [full_no_vel, full]")
        print("Obs type:", self.obs_type)
        
        # self.object_scale = torch.tensor([1.1, 1.1, 1.1])
        # self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        # smaller cube is easier to rotate for inspire hand
        # self.object_scale = torch.tensor([0.9, 0.9, 0.9])
        # self.object_scale = torch.tensor([0.85, 0.85, 0.85])
        # smaller cube is easier to roll by gravity, bigger is easier to be pushed by hand links
        # self.object_scale = torch.tensor([0.80, 0.80, 0.80])
        # self.object_scale = torch.tensor([0.70, 0.70, 0.70])
        # self.object_scale = torch.tensor([0.60, 0.60, 0.60])
        self.object_scale = torch.tensor([0.50, 0.50, 0.50])
        
        InHandManipulationTask.update_config(self)


    # def set_up_scene(self, scene) -> None:
    #     self._stage = get_current_stage()

    #     # first create a single environment
    #     self.get_inspire_hand()

    #     # call the parent class to clone the single environment
    #     super().set_up_scene(scene)

    #     # construct an ArticulationView object to hold our collection of environments
    #     self._inspire_hand_R = InspireHandView(prim_paths_expr="/World/envs/.*/inspire_L", name="inspire_hand_view")

    #     # register the ArticulationView object to the world, so that it can be initialized
    #     scene.add(self._inspire_hand_R)
    #     return

    # def get_observations(self) -> dict:
    #     # dummy observation
    #     self.obs_buf = torch.zeros(
    #         (self._num_envs, self._num_observations), device=self.device, dtype=torch.float)
    #         # self._num_observations, device=self.device, dtype=torch.float)
    #     observations = {self._inspire_hand_R.name: {"obs_buf": self.obs_buf}}
    #     return observations

    # def get_starting_positions(self):
    #     self.hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
    #     self.hand_start_orientation = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=self.device)
    #     self.pose_dy, self.pose_dz = -0.39, 0.10
    def get_starting_positions(self):
        self.hand_start_translation = torch.tensor([0.01, 0.01, 0.5], device=self.device)
        # self.hand_start_orientation = torch.tensor([0.257551, 0.283045, 0.683330, -0.621782], device=self.device)
        # 90 degree rotation in Y
        # self.hand_start_orientation = torch.tensor([0.7071, -0.7072, 0.0, 0.0], device=self.device)
        # 90 degree rotation in Y, 5 degree in X
        # self.hand_start_orientation = torch.tensor([0.7064, -0.7065, 0.0309, 0.0308], device=self.device)
        # 90 degree rotation in Y, 5 degree in X, and more 10 degree in Y
        # self.hand_start_orientation = torch.tensor([0.6421, -0.7654, 0.028, 0.0334], device=self.device)
        # # 90 degree rotation in Y, 5 degree in X, and more 20 degree in Y
        # self.hand_start_orientation = torch.tensor([0.573, -0.8184, 0.025, 0.0357], device=self.device)
        
        #  # 90 degree rotation in Y, -5 degree in X, and more 20 degree in Y
        # self.hand_start_orientation = torch.tensor([0.573, -0.8184, -0.025, 0.0357], device=self.device)
        # 90 degree rotation in Y, -15 degree in X, and more 20 degree in Y
        # self.hand_start_orientation = torch.tensor([0.5686, -0.8122, -0.0749, -0.107], device=self.device)
        # 90 degree rotation in Y, -20 degree in X, and more 20 degree in Y
        # self.hand_start_orientation = torch.tensor([0.5648, -0.8086, -0.0997, -0.1423], device=self.device)
        
        # 90 degree rotation in Y, -20 degree in X, and more 30 degree in Y
        self.hand_start_orientation = torch.tensor([0.4924, -0.8529, -0.0869, -0.1504], device=self.device)
        
        # self.hand_start_orientation = torch.tensor([0.6532, -0.6533, 0.2706, 0.2705], device=self.device)
        # 75 degree rotation in Y
        # self.hand_start_orientation = torch.tensor([0.7933, -0.6088, 0.0, 0.0], device=self.device)
        self.pose_dy, self.pose_dz = 0.15, 0.01
        # self.pose_dx = -0.05 # object scale 1.0
        self.pose_dx = -0.01 # object scale 0.9, close to thumb for a little bit    

    #Ref: omniisaacgymenvs/tasks/franka_deformable.py
    def get_hand(self):
        # add a single robot to the stage
        inpsire_hand = InspireHandR(
            prim_path=self.default_zero_env_path + "/inspire_L", 
            name="inspire_L", 
            # orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # translation=torch.tensor([0.0, 0.0, 0.0]),
            orientation=self.hand_start_orientation,
            translation=self.hand_start_translation,
        )

        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "inspire_hand", get_prim_at_path(inpsire_hand.prim_path), self._sim_config.parse_actor_config("inspire_hand")
        )
        # inpsire_hand.setins(stage=self._stage, prim=inpsire_hand.prim)
        
    def get_hand_view(self, scene):
        return InspireHandView(prim_paths_expr="/World/envs/.*/inspire_L", name="inspire_hand_view")
    
    def get_observations(self):
        self.get_object_goal_observations()

        self.hand_dof_pos = self._hands.get_joint_positions(clone=False)
        self.hand_dof_vel = self._hands.get_joint_velocities(clone=False)

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unkown observations type!")

        observations = {self._hands.name: {"obs_buf": self.obs_buf}}
        return observations

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            # self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
            #     self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            # )

            # self.obs_buf[:, 16:19] = self.object_pos
            # self.obs_buf[:, 19:23] = self.object_rot
            # self.obs_buf[:, 23:26] = self.goal_pos
            # self.obs_buf[:, 26:30] = self.goal_rot
            # self.obs_buf[:, 30:34] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            # self.obs_buf[:, 34:50] = self.actions
            
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.obs_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel

            self.obs_buf[:, 12:15] = self.object_pos
            self.obs_buf[:, 15:19] = self.object_rot
            self.obs_buf[:, 19:22] = self.goal_pos
            self.obs_buf[:, 22:26] = self.goal_rot
            self.obs_buf[:, 26:30] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 30:36] = self.actions
            # 6: inspire hand joints position (action space)
            # 6: inspire hand joints velocity
            # 3: object position
            # 4: object rotation
            # 3: goal position
            # 4: goal rotation
            # 4: goal relative rotation
            # 6: previous action
        else:
            # 6: inspire hand joints position (action space)
            # 6: inspire hand joints velocity
            # 3: object position
            # 4: object rotation
            # 3: scaled linear velocity
            # 3: scaled angle velocity
            # 3: goal position
            # 4: goal rotation
            # 4: goal relative rotation
            # 6: previous action

            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.obs_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel

            self.obs_buf[:, 12:15] = self.object_pos
            self.obs_buf[:, 15:19] = self.object_rot
            self.obs_buf[:, 19:22] = self.object_linvel
            self.obs_buf[:, 22:25] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 25:28] = self.goal_pos
            self.obs_buf[:, 28:32] = self.goal_rot
            self.obs_buf[:, 32:36] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 36:42] = self.actions