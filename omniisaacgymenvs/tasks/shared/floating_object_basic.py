# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
from abc import abstractmethod

import numpy as np
import torch
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.nucleus import get_assets_root_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask


class FloatingObjectBasicTask(RLTask):
    def __init__(self, name, env, offset=None) -> None:

        FloatingObjectBasicTask.update_config(self)

        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.randomization_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0
        self.debug_dof = 0

    def update_config(self):
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
        self.fall_dist = self._task_cfg["env"]["fallDistance"]
        self.fall_penalty = self._task_cfg["env"]["fallPenalty"]
        self.rot_eps = self._task_cfg["env"]["rotEps"]
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self._task_cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]

        self.hand_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)
        
        self.weights = {}
        # self.weights['delta_init_qpos_value'] = -0.1
        # self.weights['delta_init_qpos_value'] = -0.01
        self.weights['delta_init_qpos_value'] = 0.0
        # self.weights['right_hand_dist'] = -1.0
        self.weights['right_hand_dist'] = -10.0
        self.weights['right_hand_finger_dist'] = -1.0
        self.weights['right_hand_joint_dist'] = 0.0
        self.weights['right_hand_body_dist'] = 0.0
        # self.weights['max_finger_dist'] = 0.3
        self.weights['max_finger_dist'] = 0.05
        # self.weights['max_hand_dist'] = 0.06
        # self.weights['max_hand_dist'] = 0.05
        self.weights['max_hand_dist'] = 0.02
        self.weights['max_goal_dist'] = 0.05
        self.weights['delta_target_hand_pca'] = 0.0
        self.weights['right_hand_exploration_dist'] = 0.0
        self.weights['goal_dist'] = -0.5
        self.weights['goal_rew'] = 1.0
        self.weights['hand_up'] = 2.0
        self.weights['bonus'] = 1.0
        self.weights['hand_up_goal_dist'] = 1.0
        

        self.dt = 1.0 / 60
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = get_assets_root_path()

        self.get_starting_positions()
        self.get_hand()

        self.goal_displacement_tensor = torch.tensor([0, 0, 0.1], device=self.device)
        self.goal_start_translation = self.object_start_translation + self.goal_displacement_tensor
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.get_object(self.hand_start_translation, self.pose_dy, self.pose_dz)
        # self.object_init_pc = np.load('/home/wenbo/Obj_Asset/pc/Bottle/coacd/pc_fps1024_008.npy',allow_pickle=True)
        self.object_init_pc = np.load('/home/wenbo/Car/pc_fps1024_008.npy',allow_pickle=True)
        theta = torch.tensor(torch.pi / 2)  # change rotation representation to usd format

        # object_pc_init_rotation_matrix = torch.tensor([
        #     [1, 0, 0],
        #     [0, torch.cos(theta), -torch.sin(theta)],
        #     [0, torch.sin(theta), torch.cos(theta)]
        # ]).float()
        self.object_init_pc = torch.tensor(self.object_init_pc).float()
        self.object_init_pc *= 0.7 # object scale
        # self.object_init_pc = torch.matmul(self.object_init_pc, object_pc_init_rotation_matrix.T)
        self.object_init_pc = self.object_init_pc.to(self._device).unsqueeze(0).repeat([self.num_envs, 1, 1])
        # self.object_init_pc = torch.tensor(self.object_init_pc, device=self._device).float().unsqueeze(0).repeat([self.num_envs, 1, 1])
        
        self.get_goal()
        # self.get_table(self.hand_start_translation, self.pose_dy, self.pose_dz)
        from pxr import UsdGeom, Gf, Usd
        point_cloud_prim_path = "/World/PointCloud"
        self.point_cloud_prim = UsdGeom.Points.Define(self._stage, point_cloud_prim_path)

        super().set_up_scene(scene, filter_collisions=False)

        self._hands = self.get_hand_view(scene)
        scene.add(self._hands)
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
            # masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
            masses=torch.tensor([0.10] * self._num_envs, device=self.device),
        )
        scene.add(self._objects)
        # self._objects._non_root_link = True
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object", name="goal_view", reset_xform_properties=False
        )
        self._goals._non_root_link = True  # hack to ignore kinematics
        scene.add(self._goals)
        self.finger_move_direction = 0.0

        
        # self._tables  = RigidPrimView(
        #     # prim_paths_expr="/World/envs/env_.*/table01/table01_inst", name="table_view", reset_xform_properties=False,
        #     prim_paths_expr="/World/envs/env_.*/table01/table01", name="table_view", reset_xform_properties=False,
        #     masses=torch.tensor([1000.0] * self._num_envs, device=self.device)
        # )
        # # # self._tables._non_root_link = True
        # scene.add(self._tables)
        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)
            

    # def initialize_views(self, scene): # not used
    #     RLTask.initialize_views(self, scene)

    #     if scene.object_exists("shadow_hand_view"):
    #         scene.remove_object("shadow_hand_view", registry_only=True)
    #     if scene.object_exists("finger_view"):
    #         scene.remove_object("finger_view", registry_only=True)
    #     if scene.object_exists("allegro_hand_view"):
    #         scene.remove_object("allegro_hand_view", registry_only=True)
    #     if scene.object_exists("goal_view"):
    #         scene.remove_object("goal_view", registry_only=True)
    #     if scene.object_exists("object_view"):
    #         scene.remove_object("object_view", registry_only=True)

    #     self.get_starting_positions()
    #     self.object_start_translation = self.hand_start_translation.clone()
    #     self.object_start_translation[1] += self.pose_dy
    #     self.object_start_translation[2] += self.pose_dz
    #     self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

    #     self.goal_displacement_tensor = torch.tensor([-0.2, -0.06, 0.12], device=self.device)
    #     self.goal_start_translation = self.object_start_translation + self.goal_displacement_tensor
    #     self.goal_start_translation[2] -= 0.04
    #     self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

    #     self._hands = self.get_hand_view(scene)
    #     scene.add(self._hands)
    #     self._objects = RigidPrimView(
    #         prim_paths_expr="/World/envs/env_.*/object/object",
    #         name="object_view",
    #         reset_xform_properties=False,
    #         masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
    #     )
    #     scene.add(self._objects)
    #     self._goals = RigidPrimView(
    #         prim_paths_expr="/World/envs/env_.*/goal/object", name="goal_view", reset_xform_properties=False
    #     )
    #     self._goals._non_root_link = True  # hack to ignore kinematics
    #     scene.add(self._goals)

    #     if self._dr_randomizer.randomize:
    #         self._dr_randomizer.apply_on_startup_domain_randomization(self)

    @abstractmethod
    def get_hand(self):
        pass

    @abstractmethod
    def get_hand_view(self):
        pass

    @abstractmethod
    def get_observations(self):
        pass

    def get_object(self, hand_start_translation, pose_dy, pose_dz):
        # self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        # self.object_usd_path = f"/home/wenbo/Obj_Asset/Bottle_col.usd"
        self.object_usd_path = f"/home/wenbo/Car/Car_col.usd"
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        self.obj = XFormPrim(
            # prim_path=self.default_zero_env_path + "/object/object",
            prim_path=self.default_zero_env_path + "/object",
            name="object",
            translation=self.object_start_translation,
            # translation= self.table_start_translation + torch.tensor([0.,0.,0.7]).cuda(),
            orientation=self.object_start_orientation,
            # scale=self.object_scale,
            scale=self.object_scale * 0.7,
        )
        self._sim_config.apply_articulation_settings(
            "object", get_prim_at_path(self.obj.prim_path), self._sim_config.parse_actor_config("object")
        )

    def get_table(self, hand_start_translation, pose_dy, pose_dz):
        # self.table_usd_path = f"{self._assets_root_path}/Isaac/Props/Mounts/table.usd"
        # self.table_usd_path = f"{self._assets_root_path}/Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/common_assets/props_general/table01/table01.usd"
        self.table_usd_path = f"/home/wenbo/Obj_Asset/table01_col.usd"
        add_reference_to_stage(self.table_usd_path, self.default_zero_env_path + "/table01")
        table = XFormPrim(
            # prim_path=self.default_zero_env_path + "/table01/table01_inst",
            prim_path=self.default_zero_env_path + "/table01/table01",
            name="table",
            # translation=self.object_start_translation,
            translation=self.table_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale,
        )
        # self._sim_config.apply_articulation_settings(
        #     "table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table")
        # )
        self._sim_config.apply_rigid_body_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"),is_articulation=False)
    def get_goal(self):
        self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd" # zl
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal",
            name="goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.object_scale,
        )
        self._sim_config.apply_articulation_settings(
            "goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object")
        )

    def post_reset(self):
                
        self.num_hand_dofs = self._hands.num_dof
        self.hand_dof_default_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        self.actuated_dof_indices = self._hands.actuated_dof_indices
        self.finger_dof_indices = self.actuated_dof_indices[6:]
        self.base_trans_dof_indices = self.actuated_dof_indices[:3]
        self.base_rot_dof_indices = self.actuated_dof_indices[3:6]
        
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        
        dof_limits = self._hands.get_dof_limits()
        self.hand_dof_lower_limits, self.hand_dof_upper_limits = torch.t(dof_limits[0].to(self.device))
                
        self.hand_dof_pregrasp_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        self.hand_dof_pregrasp_pos[6:10] = torch.pi / 6 # index, middle, pinky, ring
        # self.hand_dof_pregrasp_pos[6:10] = torch.pi / 6 # index, middle, pinky, ring
        
        
        self.object_init_pos, self.object_init_rot = self._objects.get_world_poses() # strange value, z around 0.7
        self.object_init_pos -= self._env_pos 
        # self.object_init_pos = self.object_start_translation.unsqueeze(0).repeat([self.num_envs,1])
        self.object_init_velocities = torch.zeros_like(
            self._objects.get_velocities(), dtype=torch.float, device=self.device
        )
        self.hand_init_velocities = torch.zeros_like(
            self._hands.get_velocities(), dtype=torch.float, device=self.device
        )
        self.hand_dof_init_velocities = torch.zeros_like(
            self._hands.get_joint_velocities(), dtype=torch.float, device=self.device
        )
        self.goal_pos = self.object_init_pos.clone()
        # self.goal_pos[:, 2] += 0.1
        # self.goal_pos[:, 1] += 0.1
        self.goal_pos[:, 2] += 0.1
        self.goal_rot = self.object_init_rot.clone()

        self.goal_init_pos = self.goal_pos.clone()
        self.goal_init_rot = self.goal_rot.clone()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def get_object_goal_observations(self):
        self.object_pos, self.object_rot = self._objects.get_world_poses(clone=False)
        self.object_pos -= self._env_pos
        self.object_velocities = self._objects.get_velocities(clone=False)
        self.object_linvel = self.object_velocities[:, 0:3]
        self.object_angvel = self.object_velocities[:, 3:6]

    def calculate_metrics(self):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        # ) = compute_hand_reward(
        # ) = compute_hand_reward_stage_1(
        # ) = compute_obj_reward(
        ) = compute_supgrasp_reward(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            # self.weights,
            torch.tensor(self.finger_dof_indices),
            # self.hand_obj_dist,
            # self.hand_obj_dist_xy,
            self.object_pos,
            self.object_rot,
            self.goal_pos,
            self.goal_rot,
            self.hand_dof_pos, # current joint states
            self.hand_dof_pregrasp_pos, # default joint states
            # self.right_hand_pos,
            self.right_hand_palm_pos,
            self.right_hand_pc_dist,
            self.right_hand_finger_pc_dist,
            self.object_points,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
        )

        self.extras["consecutive_successes"] = self.consecutive_successes.mean()
        self.randomization_buf += 1

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets)
                )

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        reset_buf = self.reset_buf.clone()

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        obj_local_pos = self._objects.get_local_poses()[0]
        obj_world_pos = self._objects.get_world_poses()[0]
        hand_world_pos = self._hands.get_world_poses()[0]
        hand_local_pos = hand_world_pos - self._env_pos

        
        # # move base root pos
        # lower_limit = torch.tensor([-1,-1,0.05]).cuda()
        # upper_limit = torch.tensor([1,1,1.5]).cuda()
        # target_pos =  hand_local_pos + actions *0.01
        # target_pos = torch.clamp(target_pos, lower_limit, upper_limit)
        # target_pos += self._env_pos
        # self._hands.set_joint_positions(self.hand_dof_default_pos[0])
        
        
        hand_dof = self._hands.get_joint_positions()
        lower_limit = torch.tensor([-1,-1,-0.13]).cuda()
        upper_limit = torch.tensor([1,1,0.5]).cuda()
        # self._hands.set_joint_positions(self.hand_dof_default_pos[0])
        joint_indices = torch.tensor(self.base_trans_dof_indices + self.finger_dof_indices)
        # joint_indices = np.array(self.base_trans_dof_indices)
        
        target_hand_dof = hand_dof[:, joint_indices]
        target_hand_dof[:,:3] += actions[:,:3] * 0.05
        target_hand_dof[:,:3] = torch.clamp(target_hand_dof[:,:3], lower_limit, upper_limit)
        
        # target_hand_dof[:,:1] -=0.01 
        # target_hand_dof[:, 3:] = self.hand_dof_default_pos[self.finger_dof_indices] #static finger
        # target_hand_dof [:, 3:] += self.finger_move_direction * 0.1
        
        
        
        # target_hand_dof[:,3:] += actions[:,3:]* 0.1 #dynamic finger
        target_hand_dof[:,3:7] += actions[:,3:7]* 0.1 #slower move for other 4 fingers
        target_hand_dof[:,7:9] += actions[:,7:9]* 0.3 #faster for thumb
        target_hand_dof[:,3:] = torch.clamp(target_hand_dof[:,3:], self.hand_dof_lower_limits[self.finger_dof_indices], self.hand_dof_upper_limits[self.finger_dof_indices])
        
        # # self.hand_dof_pregrasp_pos[self.finger_dof_indices]
        # target_hand_dof[:,:] = self.hand_dof_pregrasp_pos[joint_indices] 
        # # target_hand_dof[:,-1] += self.progress_buf * 0.01
        # self.hand_dof_upper_limits[8] = 1.3 # pinky
        # np.array(self._hands._dof_names)[joint_indices]
        # if self.progress_buf[0] <60:
        #     target_hand_dof[:,0] -= self.progress_buf * 0.01
        #     target_hand_dof[:,0] = torch.clamp(target_hand_dof[:,0],torch.tensor([-0.120]).cuda(),torch.tensor([1.05]).cuda())
        # elif self.progress_buf[0] <160:
        #     target_hand_dof[:,0] = -0.120
        #     target_hand_dof[:, 3:] = (self.progress_buf-60).unsqueeze(-1) * 0.012 *self.hand_dof_upper_limits[self.finger_dof_indices]
        #     target_hand_dof[:, 3:5] = 0
        #     target_hand_dof[:, 6] = 0
        #     # print(self.progress_buf)
        #     # target_hand_dof[:,3:-1] += (self.progress_buf * 0.005).unsqueeze(-1)
        #     target_hand_dof[:,3:] = torch.clamp(target_hand_dof[:,3:], self.hand_dof_lower_limits[self.finger_dof_indices], self.hand_dof_upper_limits[self.finger_dof_indices])
        # elif self.progress_buf[0] >=160:
        #     target_hand_dof[:, 3:] = self.hand_dof_upper_limits[self.finger_dof_indices] * 1.0
        #     target_hand_dof[:, 3:5] = 0
        #     target_hand_dof[:, 6] = 0
        #     target_hand_dof[:,0] += (self.progress_buf -160) * 0.005 - 0.120
        #     target_hand_dof[:,3:] = torch.clamp(target_hand_dof[:,3:], self.hand_dof_lower_limits[self.finger_dof_indices], self.hand_dof_upper_limits[self.finger_dof_indices])
        # self._hands.set_joint_position_targets(target_hand_dof,joint_indices = joint_indices) # direct position control, wo smoothing
        # print(self.progress_buf)
        
        self.cur_targets[:, joint_indices] = 0.2 * target_hand_dof + 0.8 * self.prev_targets[:, joint_indices]
        self._hands.set_joint_position_targets(self.cur_targets[:, joint_indices],joint_indices = joint_indices)
        
        self.prev_targets = self.cur_targets
      
      

    def is_done(self):
        pass

    def reset_target_pose(self, env_ids):
        # reset goal
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        rand_floats_trans = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device) *0.05

        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # self.goal_pos[env_ids] = self.goal_init_pos[env_ids, 0:3] + rand_floats_trans
        self.goal_pos[env_ids] = self.goal_init_pos[env_ids, 0:3]
        # self.goal_rot[env_ids] = new_rot
        self.goal_rot[env_ids] = self.goal_init_rot[env_ids]

        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = (
            # self.goal_pos[env_ids] + self.goal_displacement_tensor + self._env_pos[env_ids]
            self.goal_pos[env_ids] + self._env_pos[env_ids]
        )  # add world env pos
        goal_pos[:,2]+=1.0 # avoid collision

        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        self.reset_target_pose(env_ids) # set reset_goal_buf to 0 

        # reset object
        # new_object_pos = (
        #     self.object_init_pos[env_ids] + self.reset_position_noise * rand_floats[:, 0:3] + self._env_pos[env_ids]
        # )  # add world env pos

        # new_object_rot = randomize_rotation(
        #     rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        new_object_rot = self.object_start_orientation.unsqueeze(0).repeat([len(env_ids),1])
        new_object_pos = self.object_start_translation.unsqueeze(0).repeat([len(env_ids),1]) + self._env_pos[env_ids]
        object_velocities = torch.zeros_like(self.object_init_velocities, dtype=torch.float, device=self.device)
        self._objects.set_velocities(object_velocities[env_ids], indices)
        self._objects.set_world_poses(new_object_pos, new_object_rot, indices)
        

        hand_velocities = torch.zeros_like(self.hand_init_velocities[env_ids], dtype=torch.float, device=self.device)
        hand_init_pos = self.hand_start_translation.repeat([len(env_ids),1]) + self._env_pos[env_ids]
        hand_init_rot = self.hand_start_orientation.repeat([len(env_ids),1])
        
        # a = self.hand_start_orientation.unsqueeze(0).repeat([self.num_envs,1])
        self._hands.set_world_poses(positions=hand_init_pos,orientations=hand_init_rot,indices= indices)
        self._hands.set_velocities(hand_velocities,indices= indices)
        # self._hands.set_joint_position_targets(
        #         self.hand_dof_default_pos[env_ids],indices = indices
        #     )
        # self._hands.set_joint_positions(self.hand_dof_default_pos[env_ids],indices = indices)
        pos = self.hand_dof_default_pos 
 
        self.prev_targets[env_ids, : self.num_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_hand_dofs] = pos
        
        hand_dof_init_velocities = self.hand_dof_init_velocities[env_ids]
        self._hands.set_joint_positions(self.hand_dof_default_pos, indices = indices)
        self._hands.set_joint_velocities(hand_dof_init_velocities, indices = indices)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def compute_supgrasp_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    # weights,
    finger_dof_indices,
    # hand_obj_dist,
    # hand_obj_dist_xy,
    object_pos,
    object_rot,
    goal_pos,
    goal_rot, # 
    hand_dof_pos,
    hand_dof_pregrasp_pos,
    # right_hand_pos,
    right_hand_palm_pos,
    right_hand_pc_dist,
    right_hand_finger_pc_dist,
    object_points,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
):
 
    # goal_hand_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
    # goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
    # goal_hand_dist = torch.norm(object_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
    # goal_hand_dist = right_hand_pc_dist # palm to object
    # goal_hand_dist = right_hand_finger_pc_dist /5
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]
    max_finger_dist = 0.2 # 0.15
    max_hand_dist = 0.045
    max_goal_dist = 0.05
    hand_up_goal_dist = 1.0
    
    # Assign target initial hand pose in the midair
    # target_init_pose = torch.tensor([0.1, 0., 0.6, 0., 0., 0., 0.6, 0., -0.1, 0., 0.6, 0., 0., -0.2, 0., 0.6, 0., 0., 1.2, 0., -0.2, 0.], dtype=hand_dof_pos.dtype, device=hand_dof_pos.device)
    # delta_init_qpos_value = torch.norm(hand_dof_pos - target_init_pose, p=1, dim=-1)

        

    
    # goal_distance = 0.045
    # reward = -1 * right_hand_pc_dist - 1 * right_hand_finger_pc_dist
    # palm_hold = right_hand_pc_dist < goal_distance 
    # finger_hold = (right_hand_finger_pc_dist / 5) < (goal_distance -0.015)
    # hold_flag = palm_hold *1 + finger_hold*1
    # reward = torch.where(hold_flag == 2, reward +10, reward)
    
    
    delta_init_qpos_value = torch.norm(hand_dof_pos[:,finger_dof_indices] - hand_dof_pregrasp_pos[finger_dof_indices], p=1, dim=-1) # pass
    right_hand_dist = right_hand_pc_dist
    right_hand_finger_dist = right_hand_finger_pc_dist
    goal_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
    goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1)
    
    hold_value = 2
    hold_flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
    
        # # ---------------------- Reward After Holding ---------------------- # #
    # Distanc from object pos to goal target pos
    goal_rew = torch.zeros_like(goal_dist)
    goal_rew = torch.where(hold_flag == hold_value, 1.0 * (0.9 - 2.0 * goal_dist), goal_rew)
    # Distance from hand pos to goal target pos
    hand_up = torch.zeros_like(goal_dist)
    hand_up = torch.where(lowest >= 0.01, torch.where(hold_flag == hold_value, 0.1 + 0.1 * actions[:, 0], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.20, torch.where(hold_flag == hold_value, 0.2 - goal_hand_dist * 0 + hand_up_goal_dist * (0.2 - goal_dist), hand_up), hand_up)
    # Already hold the object and Already reach the goal
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(hold_flag == hold_value, torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
        
    
    
    init_reward = -0.1* delta_init_qpos_value  + -1.0 * right_hand_dist + -0.5 * goal_dist

    grasp_reward = -1.0 * right_hand_finger_dist + -2.0 * right_hand_dist + -0.5 * goal_dist + 1.0 * goal_rew + 2.0 * hand_up + 1.0 * bonus
    reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)
    
        # Init reset_buff
    resets = reset_buf
    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    # Reset goal also
    goal_resets = resets
    # Compute successes: reach the goal during running
    successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), successes)
    # Compute final_successes: reach the goal at the end
    final_successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), torch.zeros_like(successes))
    # Compute current_successes: reach the episode length and reach the goal
    # current_successes = torch.where(resets == 1, successes, current_successes)
    # Compute cons_successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes, consecutive_successes)

    
    
    # lift_rew = lowest - 0.1
    # reward = torch.where(hold_flag == 2, reward + 10* lift_rew, reward)
    # hand_up = torch.zeros_like(reward)
    # hand_up = torch.where(lowest >= 0.0, torch.where(hold_flag == 2, 0.1 + 1.0 * actions[:, 0], hand_up), hand_up) # zl +x of action is world + z
    # reward += hand_up
    
    # goal_standard = (hold_flag == 2) & (lowest >= 0.1)
    # reward = torch.where(goal_standard == 1, reward +20, reward)
    
    # goal_resets = torch.where(goal_standard == 1, torch.ones_like(reset_goal_buf), reset_goal_buf)
    
    # successes = successes + goal_resets
    # resets = goal_resets # *
    # # print('palm \n')
    # # print(right_hand_pc_dist)
    # print(reward)

    # if max_consecutive_successes > 0:
    #     # Reset progress buffer on goal envs if max_consecutive_successes > 0
    #     progress_buf = torch.where(
    #         goal_standard == 1, torch.zeros_like(progress_buf), progress_buf
    #     )
    #     resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    # resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # num_resets = torch.sum(resets)
    # finished_cons_successes = torch.sum(successes * resets.float())

    # cons_successes = torch.where(
    #     num_resets > 0,
    #     av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
    #     consecutive_successes,
    # )
    
    return reward, resets, goal_resets, progress_buf, successes, cons_successes


# @torch.jit.script
def compute_obj_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    weights,
    # hand_obj_dist,
    # hand_obj_dist_xy,
    object_pos,
    object_rot,
    goal_pos,
    goal_rot, # 
    # hand_dof_pos,
    # hand_dof_pregrasp_pos,
    # right_hand_pos,
    right_hand_palm_pos,
    right_hand_pc_dist,
    right_hand_finger_pc_dist,
    object_points,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
):
    # exp_type = 'single_reward_approaching'
    # exp_type = 'double_reward_approaching'
    # exp_type = 'double_reward_approaching_lift'
    # exp_type = 'single_reward_approaching_lift'
    exp_type = 'unidexgrasp'
    
    if exp_type == 'single_reward_approaching':
        # goal_hand_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(object_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = right_hand_pc_dist # palm to object
        goal_hand_dist = right_hand_finger_pc_dist /5
        goal_distance = 0.04
        reward = -1 * goal_hand_dist
        reward = torch.where(goal_hand_dist < goal_distance, reward +20, reward)
        goal_resets = torch.where(goal_hand_dist < goal_distance, torch.ones_like(reset_goal_buf), reset_goal_buf)
        successes = successes + goal_resets
        resets = goal_resets # *
        # print('palm \n')
        # print(right_hand_pc_dist)
        print(reward)

        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(
                goal_hand_dist < 0.02, torch.zeros_like(progress_buf), progress_buf
            )
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(
            num_resets > 0,
            av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
            consecutive_successes,
        )
    elif exp_type == 'double_reward_approaching':
        # goal_hand_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(object_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = right_hand_pc_dist # palm to object
        # goal_hand_dist = right_hand_finger_pc_dist /5
        lowest = torch.min(object_points[:, :, -1], dim=1)[0]
        goal_distance = 0.04
        reward = -1 * right_hand_pc_dist -  1 * right_hand_finger_pc_dist
        palm_hold = right_hand_pc_dist < goal_distance
        finger_hold = right_hand_finger_pc_dist / 5 < goal_distance
        hold_flag = palm_hold *1 + finger_hold*1
        
        reward = torch.where(hold_flag == 2, reward +20, reward)
        goal_resets = torch.where(hold_flag == 2, torch.ones_like(reset_goal_buf), reset_goal_buf)
        successes = successes + goal_resets
        resets = goal_resets # *
        # print('palm \n')
        # print(right_hand_pc_dist)
        print(reward)

        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(
                hold_flag == 2, torch.zeros_like(progress_buf), progress_buf
            )
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(
            num_resets > 0,
            av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
            consecutive_successes,
        )
        
    elif exp_type == 'double_reward_approaching_lift':
        # goal_hand_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(object_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = right_hand_pc_dist # palm to object
        # goal_hand_dist = right_hand_finger_pc_dist /5
        lowest = torch.min(object_points[:, :, -1], dim=1)[0]
        goal_distance = 0.045
        reward = -1 * right_hand_pc_dist -  1 * right_hand_finger_pc_dist
        palm_hold = right_hand_pc_dist < goal_distance 
        finger_hold = right_hand_finger_pc_dist / 5 < goal_distance -0.01
        hold_flag = palm_hold *1 + finger_hold*1
        
        # reward = torch.where(hold_flag == 2, reward +20, reward)
        reward = torch.where(hold_flag == 2, reward +10, reward)
        
        
        lift_rew = lowest - 0.1
        reward = torch.where(hold_flag == 2, reward + 10* lift_rew, reward)
        
        goal_standard = (hold_flag == 2) & (lowest >= 0.1)
        reward = torch.where(goal_standard == 1, reward +20, reward)
        
        goal_resets = torch.where(goal_standard == 1, torch.ones_like(reset_goal_buf), reset_goal_buf)
        
        successes = successes + goal_resets
        resets = goal_resets # *
        # print('palm \n')
        # print(right_hand_pc_dist)
        print(reward)
    
        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(
                goal_standard == 1, torch.zeros_like(progress_buf), progress_buf
            )
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(
            num_resets > 0,
            av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
            consecutive_successes,
        )

    elif exp_type == 'single_reward_approaching_lift':
        # goal_hand_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = torch.norm(object_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
        # goal_hand_dist = right_hand_pc_dist # palm to object
        # goal_hand_dist = right_hand_finger_pc_dist /5
        lowest = torch.min(object_points[:, :, -1], dim=1)[0]
        goal_distance = 0.045
        reward = -1 * right_hand_pc_dist - 1 * right_hand_finger_pc_dist
        palm_hold = right_hand_pc_dist < goal_distance 
        finger_hold = (right_hand_finger_pc_dist / 5) < (goal_distance -0.015)
        hold_flag = palm_hold *1 + finger_hold*1
        
        # reward = torch.where(hold_flag == 2, reward +20, reward)
        reward = torch.where(hold_flag == 2, reward +10, reward)
        
        
        lift_rew = lowest - 0.1
        reward = torch.where(hold_flag == 2, reward + 10* lift_rew, reward)
        hand_up = torch.zeros_like(reward)
        hand_up = torch.where(lowest >= 0.0, torch.where(hold_flag == 2, 0.1 + 1.0 * actions[:, 0], hand_up), hand_up) # zl +x of action is world + z
        reward += hand_up
        
        goal_standard = (hold_flag == 2) & (lowest >= 0.1)
        reward = torch.where(goal_standard == 1, reward +20, reward)
        
        goal_resets = torch.where(goal_standard == 1, torch.ones_like(reset_goal_buf), reset_goal_buf)
        
        successes = successes + goal_resets
        resets = goal_resets # *
        # print('palm \n')
        # print(right_hand_pc_dist)
        print(reward)
    
        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(
                goal_standard == 1, torch.zeros_like(progress_buf), progress_buf
            )
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(
            num_resets > 0,
            av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
            consecutive_successes,
        )
        
    return reward, resets, goal_resets, progress_buf, successes, cons_successes


# @torch.jit.script
def compute_hand_reward_stage_1(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    weights,
    hand_obj_dist,
    hand_obj_dist_xy,
    object_pos,
    object_rot,
    goal_pos,
    target_rot, # 
    hand_dof_pos,
    hand_dof_pregrasp_pos,
    right_hand_pos,
    right_hand_palm_pos,
    right_hand_pc_dist,
    right_hand_finger_pc_dist,
    object_points,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
):

    heighest = torch.max(object_points[:, :, -1], dim=1)[0]
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]
    
    
    max_finger_dist, max_hand_dist, max_goal_dist = weights['max_finger_dist'], weights['max_hand_dist'], weights['max_goal_dist']


    delta_init_qpos_value = torch.norm(hand_dof_pos[:,6:] - hand_dof_pregrasp_pos[6:], p=1, dim=-1) # pass
    goal_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
    
    right_hand_dist = right_hand_pc_dist # palm
    # right_hand_body_dist = right_hand_body_pc_dist
    # right_hand_joint_dist = right_hand_joint_pc_dist
    right_hand_finger_dist = right_hand_finger_pc_dist
    # right_hand_discrepancy_z = 1 - right_hand_alignment_z
    # print('reward')
    # print(right_hand_dist) #
    # print(right_hand_finger_dist/5)
    # # ---------------------- Hold Detection / Reward Before Hold ---------------------- # #
    # hold_flag: hand pos and finger reach object region
    # hold_value =2
    hold_value = 1
    hold_flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
    # print((hold_flag==2).sum())
    # print(hold_flag)
    if ((hold_flag==2).sum()) !=0:
        print('123')
    
    init_reward = weights['delta_init_qpos_value'] * delta_init_qpos_value  
    init_reward += weights['right_hand_dist'] * right_hand_dist  # 
    # init_reward += weights['right_hand_finger_dist'] * right_hand_finger_dist #  new added
    # init_reward += weights['delta_target_hand_pca'] * delta_target_hand_pca 
    # init_reward += weights['right_hand_exploration_dist'] * right_hand_exploration_dist 
  


    env_lower_bound = torch.tensor([-0.5,-0.5, -0.02]).cuda()
    env_upper_bound = torch.tensor([0.5,0.5, 0.5]).cuda()
    hand_exceed_lower = right_hand_palm_pos < env_lower_bound
    hand_exceed_upper = right_hand_palm_pos > env_upper_bound
    hand_out_of_box = hand_exceed_lower + hand_exceed_upper
    hand_reset_flag = hand_out_of_box.sum(1)
    obj_exceed_lower = object_pos < env_lower_bound
    obj_exceed_upper = object_pos > env_upper_bound
    obj_out_of_box = obj_exceed_lower + obj_exceed_upper
    obj_reset_flag = obj_out_of_box.sum(1)
    reward = init_reward
    
    reward = torch.where(hold_flag == hold_value, reward +20, reward)
    goal_resets = torch.where(goal_dist <= max_goal_dist, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    resets = goal_resets # *
    print(reward)
    
    
    # reward = torch.where(hand_reset_flag != 0, reward + fall_penalty, reward) # fall out penalty for hand
    # resets = torch.where(hand_reset_flag != 0, torch.ones_like(resets), resets)
    # reward = torch.where(obj_reset_flag != 0, reward + fall_penalty, reward) # fall out penalty for obj
    # resets = torch.where(obj_reset_flag != 0, torch.ones_like(resets), resets)
    
    
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(
            goal_dist <= max_goal_dist, torch.zeros_like(progress_buf), progress_buf
        )
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    # if max_consecutive_successes > 0:
    #     reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes




    # hack pipeline

    goal_dist = hand_obj_dist_xy 
    reward = -1 * hand_obj_dist_xy # distance reward

    goal_resets = torch.where(goal_dist <= max_goal_dist, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward) # goal reach reward
    
    
    reward = torch.where(hand_reset_flag != 0, reward + fall_penalty, reward) # fall out penalty
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(
            goal_dist <= max_goal_dist, torch.zeros_like(progress_buf), progress_buf
        )
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

    # Init reset_buff
    # resets = reset_buf
    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    # Reset goal also
    # Compute successes: reach the goal during running
    successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), successes)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    # Compute final_successes: reach the goal at the end
    # final_successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), torch.zeros_like(successes))
    # Compute current_successes: reach the episode length and reach the goal
    # current_successes = torch.where(resets == 1, successes, current_successes)
    # Compute cons_successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes, consecutive_successes)


    return reward, resets, goal_resets, progress_buf, successes, cons_successes



    right_hand_finger_dist = right_hand_finger_pc_dist


    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(
    #     torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    # )  # changed quat convention

    # dist_rew = goal_dist * dist_reward_scale
    # rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    dist_rew = hand_obj_dist * dist_reward_scale
    finger_dist_rew = right_hand_finger_dist * dist_reward_scale
    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = dist_rew + rot_rew + action_penalty * action_penalty_scale
    reward = dist_rew + finger_dist_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    # goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    goal_resets = torch.where(hand_obj_dist <= 0.05, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    # reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)
    reward = torch.where(object_pos[:,2] <=0.5, reward + fall_penalty, reward)
    # print(object_pos[:,2])
    # Check env termination conditions, including maximum success number
    # resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(object_pos[:,2] <=0.5, torch.ones_like(reset_buf), reset_buf)
    
    env_lower_bound = torch.tensor([-0.5,-0.5, 0.5]).cuda()
    env_upper_bound = torch.tensor([0.5,0.5, 1.0]).cuda()
    exceed_lower = hand_pos < env_lower_bound
    exceed_upper = hand_pos > env_upper_bound
    hand_out_of_box = exceed_lower + exceed_upper
    hand_reset_flag = hand_out_of_box.sum(1)
    
    reward = torch.where(hand_reset_flag !=0, reward + fall_penalty, reward) # hand out of box
    resets = torch.where(hand_reset_flag !=0, torch.ones_like(resets), resets)
    
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        # progress_buf = torch.where(
        #     torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf
        # )
        
        progress_buf = torch.where(
            hand_obj_dist <= 0.05, torch.zeros_like(progress_buf), progress_buf
        )
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets) # zl 600 in cfg

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    if num_resets >0:
        # print('123')
        pass
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes
