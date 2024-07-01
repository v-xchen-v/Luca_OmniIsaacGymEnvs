"""
Step 1: Setup scene, adds franka and target object to scene, randomly set goal object's position and rotation, apply random action  to the franka arm
Step 2: Observation
Step 3: Reward
Step 4: Apply action by RL policy to the franka arm
Step 5: Finetune
"""

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omni.isaac.gym.vec_env import VecEnvBase

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.core.prims import RigidPrimView, XFormPrim

from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

import torch
import numpy as np


class FrankaFollowTargetTask(RLTask):
    def __init__(
        self,
        # three parameters will be parsed and processed by the task_utils.py script, which is called when a training run
        # is launched from command line.
        name,  # the name of the class, which will be parsed from the task config file.
        sim_config: SimConfig,  # contains task and physics parameters from task config file.
        env: VecEnvBase,  # the environment object, defined by rlgames_train.py.
        offset=None,
    ) -> None:
        # initialize few key variables in our contructor method
        # init _cfg , _task_cfg, _sim_config by config files and this should go before the calling the parent class contructor
        self.update_config(sim_config)

        # these two must be defined in the task class
        self._num_actions = 7  # 7 actions for 7 DOFs in arm, ignore the 2 DOFs in gripper for the following target task
        self._num_observations = 32
        # 7: dofbot joints position (action space)
        # 7: dofbot joints velocity
        # 3: goal position
        # 4: goal rotation
        # 4: goal relative rotation
        # 7: previous action

        # call the parent class contructor which needs _cfg, _task_cfg prepared to initialize key RL variables,
        # include setups for domain randomization, defining action and observation spaces if not specified by the task class,
        # initializing the Cloner for environment set up, and defining the reward, obervation, and reset buffers for our task
        super().__init__(name, env, offset)

        # buf record which env's goal should reset, should be called after super.init()
        self.reset_goal_buf = self.reset_buf.clone()

        # settings
        self._assets_root_path = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1"
        self.step = 0

    def update_config(self, sim_config: SimConfig):
        """Optionally implemented by individual task classes to update values parsed from config file."""
        # extract task config from main config dictionary
        
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # required variables in task class
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        # more optional configs can be put here, e.g., reset and actions related variables

    def set_up_scene(self, scene) -> None:
        """Clones environments based on value provided in task config"""
        print("=============Setup Scene=============")
        """Step 2: Setup Scene
        add three objects to the scene: Franka+Object(Stands for End Effector)+Goal
        """
        self._stage = get_current_stage()

        # first create a single environment
        self.get_franka()

        self.get_object()

        self.get_goal()

        # call the parent class to clone the single environment
        super().set_up_scene(scene)

        # construct an ArticulationView object to hold our collection of environments
        self._frankas = FrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="franka_view"
        )

        # register the ArticulationView object to the world, so that it can be initialized
        scene.add(self._frankas)

        # adds object to scene
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
        )
        self._objects._non_root_link = True  # hack to ignore kinematics
        scene.add(self._objects)
        
        # adds goal object to follow to the scene
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object",
            name="goal_view",
            reset_xform_properties=False,
        )
        self._goals._non_root_link = True  # hack to ignore kinematics
        scene.add(self._goals)

    def post_reset(self):
        """Calls while doing a .reset() on the world."""
        print("=============POST RESET==============")
        self.end_effectors_init_pos, self.end_effectors_init_rot = (
            self._frankas._hands.get_world_poses()
        )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_index(indices)
        return

    def reset_index(self, env_ids):
        # num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)
        
        # randomize DOF positions
        dof_pos = self.get_randomized_DOF_positions(indices)
        
        # randomize DOF velocities
        dof_vel = self.get_randomized_DOF_velocities(indices)
        
        # apply randomized joint positions and velocities to environments
        # use set_joint_positions() to move joint to target pos immediately, and use set_joint_position_targets doesn't move to target immediately by  
        # following PD controller.
        self._frankas.set_joint_positions(positions=dof_pos, indices=indices)
        self._frankas.set_joint_velocities(velocities=dof_vel, indices=indices)
        
        # reset the reset buffer after applying reset
        self.reset_buf[env_ids] = 0
        
    def get_randomized_DOF_positions(self, indices):
        # got franka dot limites
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device) # (9,)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device) # (9,)
        self.franka_dof_default_pos = self._frankas._default_joints_state.positions
        
        delta_max = (self.franka_dof_upper_limits - self.franka_dof_default_pos)[:, :self.num_actions] # skip 2 DOFs for gripper
        delta_min = (self.franka_dof_lower_limits - self.franka_dof_default_pos)[:, :self.num_actions] # skip 2 DOFs for gripper
        rand_floats = torch_rand_float(0, 1.0, (self.num_envs, self.num_actions), device=self._device)
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats
        
        rand_dof_pos = self.franka_dof_default_pos[:, :self.num_actions] + rand_delta
        dof_pos = torch.zeros((self._num_envs, self._frankas._num_dof), device=self._device)
        dof_pos[indices, :self.num_actions] = rand_dof_pos
        return dof_pos # (num_envs, 9)
    
    def get_randomized_DOF_velocities(self, indices):
        # got franka default velocity
        self.franka_dof_default_vel = self._frankas._default_joints_state.velocities
        
        rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs, self.num_actions), device=self._device)
        rand_dof_vel = self.franka_dof_default_vel[:, :self.num_actions] + rand_floats
        dof_vel = torch.zeros((self._num_envs, self._frankas._num_dof), device=self._device)
        dof_vel[indices, :self.num_actions] = rand_dof_vel
        return dof_vel
        
        
    def pre_physics_step(self, actions) -> None:
        """Optionally implemented by individual task classes to process actions.

        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        """
        1. Puts object on Franka's End Effector's place(loc and rot)
        2. Random Set Target Pos of Goal
        """
        # print("===============Pre Physics Step===============")
        # make sure simulation has not been stopped from the UI
        if not self._env._world.is_playing():
            return

        # extract environment indices that need reset and reset them
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_index(reset_env_ids)
        
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        
        self.step += 1
        
        # apply action generated by RL policy
        # print(actions)
        self.actions = torch.zeros((self._num_envs, self._frankas._num_dof), dtype=torch.float, device=self.device)
        self.actions[:, :self._num_actions] = actions
        self.franka_dof_target = torch.zeros((self._num_envs, self._frankas._num_dof), dtype=torch.float, device=self.device)
        self.franka_dof_target[:] = tensor_clamp(self.actions, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        self._frankas.set_joint_position_targets(self.franka_dof_target, indices=env_ids_int32)

        # got franka dot limites
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        end_effectors_pos, end_effectors_rot = self._frankas._hands.get_world_poses()

        # Reverse the default rotation and rotate the displacement tensor according to the current rotation
        self.object_pos = end_effectors_pos + quat_rotate(
            end_effectors_rot,
            quat_rotate_inverse(
                self.end_effectors_init_rot, self.get_object_displacement_tensor()
            ),
        )
        self.object_pos -= self._env_pos  # subtract world env pos
        self.object_rot = end_effectors_rot
        object_pos = self.object_pos + self._env_pos
        object_rot = self.object_rot
        self._objects.set_world_poses(object_pos, object_rot)
        # print(f"object pos: {object_pos}\nobject rot: {object_rot}\n")

        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # # if only goals need reset, then call set API
        # if len(goal_env_ids) > 0 and len(env_ids) == 0:
        #     self.reset_target_pose(goal_env_ids)
        # elif len(goal_env_ids) > 0:
        #     self.reset_target_pose(goal_env_ids)

        # self.actuated_dof_indices = [0, 1, 2, 3, 4, 5, 6]
        # rand_joint_pos = np.random.uniform(
        #     self.franka_dof_lower_limits.cpu(),
        #     self.franka_dof_upper_limits.cpu(),
        #     size=(2, 9),
        # )
        # rand_actions = torch.tensor(
        #     [rand_joint_pos[:, :-2]],
        #     dtype=torch.float32,
        #     device=self._cfg["sim_device"],
        # )
        # self._frankas.set_joint_position_targets(
        #     rand_actions, None, self.actuated_dof_indices
        # )

    def get_object(self):
        self.object_scale = torch.tensor([0.1] * 3)
        self.object_start_translation = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device
        )
        self.object_start_orientation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )

        """A simply model for robot to calculate relative rot?"""
        self.object_usd_path = (
            f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        )
        add_reference_to_stage(
            self.object_usd_path, self.default_zero_env_path + "/object"
        )
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale,
        )
        self._sim_config.apply_articulation_settings(
            "object",
            get_prim_at_path(obj.prim_path),
            self._sim_config.parse_actor_config("object"),
        )

    def get_goal(self):
        self.goal_displacement_tensor = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device
        )
        self.goal_start_translation = (
            torch.tensor([0.0, 0.0, 0.0], device=self.device)
            + self.goal_displacement_tensor
        )
        self.goal_start_orientation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )
        self.goal_scale = torch.tensor([0.5] * 3)

        # to goal object to reach
        self.goal_usd_path = (
            f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        )
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal/object",
            name="goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale,
        )
        self._sim_config.apply_articulation_settings(
            "goal",
            get_prim_at_path(goal.prim_path),
            self._sim_config.parse_actor_config("goal_object"),
        )

    def get_observations(self) -> dict:
        # # dummy observation
        # self.obs_buf = torch.zeros(
        #     (self._num_envs, self._num_observations),
        #     device=self.device,
        #     dtype=torch.float,
        # )
        # # self._num_observations, device=self.device, dtype=torch.float)
        # observations = {self._frankas.name: {"obs_buf": self.obs_buf}}

        # real observations
        arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        # 7: franka joints position (action space)
        self.arm_dof_pos = self._frankas.get_joint_positions(joint_indices=arm_joint_indices)
        # 7: franka joints velocity (action space)
        self.arm_dof_vel = self._frankas.get_joint_velocities(joint_indices=arm_joint_indices)
        self.compute_full_observations()
        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}

        return observations


    def compute_full_observations(self, no_vel=False):
        # 7: dofbot joints position (action space)
        # 7: dofbot joints velocity
        # 3: goal position
        # 4: goal rotation
        # 4: goal relative rotation
        # 7: previous action
        
        # There are many redundant information for the simple Reacher task, but we'll keep them for now.
        self.obs_buf[:, 0 : self._num_actions] = unscale(
            self.arm_dof_pos[:, : self._num_actions],
            self.franka_dof_lower_limits[:self._num_actions],
            self.franka_dof_upper_limits[:self._num_actions],
        )
        self.vel_obs_scale = 1
        self.obs_buf[:, self._num_actions : 2 * self._num_actions] = (
            self.vel_obs_scale * self.arm_dof_vel[:, : self._num_actions]
        )
        base = 2 * self._num_actions
        self.obs_buf[:, base + 0 : base + 3] = self.goal_pos
        self.obs_buf[:, base + 3 : base + 7] = self.goal_rot
        self.object_rot = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self.device)
        self.obs_buf[:, base + 7 : base + 11] = quat_mul(
            self.object_rot, quat_conjugate(self.goal_rot)
        )
        self.obs_buf[:, base + 11 : base + 18] = self.actions[:, :self._num_actions]

    def get_object_displacement_tensor(self):
        return torch.tensor([0.0, 0.015, 0.1], device=self.device).repeat(
            (self.num_envs, 1)
        )

    def reset_target_pose(self, env_ids):
        self.x_unit_tensor = torch.tensor(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        # reset goal
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_pos = self.get_reset_target_new_pos(len(env_ids))
        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )

        self.goal_pos, self.goal_rot = self._goals.get_world_poses()
        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot

        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = (
            self.goal_pos[env_ids] + self._env_pos[env_ids]
        )  # add world env pos

        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def get_reset_target_new_pos(self, n_reset_envs):
        # Randomly generate goal positions, although the resulting goal may still not be reachable.
        new_pos = torch_rand_float(-1, 1, (n_reset_envs, 3), device=self.device)
        new_pos[:, 0] = new_pos[:, 0] * 0.15 + 0.45 * torch.sign(new_pos[:, 0])
        new_pos[:, 1] = new_pos[:, 1] * 0.15 + 0.45 * torch.sign(new_pos[:, 1])
        new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.6) + 0.45
        return new_pos

    def is_done(self):
        if self.step % 360 == 0:
            print("============IS Done and RESET============")
            self.reset_buf = torch.ones((self._num_envs), device=self._device)
            self.reset_goal_buf = torch.ones((self._num_envs), device=self._device)
        else:
            self.reset_buf = torch.zeros((self._num_envs), device=self._device)
            self.reset_goal_buf = torch.zeros((self._num_envs), device=self._device)

    def calculate_metrics(self):
        goal_dist = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        distRewardScale = -10.0
        reward = distRewardScale*goal_dist#np.random.random(1)[0]
        # print(f"reward: {reward}")

        self.rew_buf[:] = reward

    # Ref: omniisaacgymenvs/tasks/franka_deformable.py
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
            "franka",
            get_prim_at_path(franka.prim_path),
            self._sim_config.parse_actor_config("franka"),
        )
        franka.set_franka_properties(stage=self._stage, prim=franka.prim)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )