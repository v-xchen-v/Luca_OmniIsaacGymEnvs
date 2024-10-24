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
import shutil
import yaml


class VerticalGraspBasicTask(RLTask):
    def __init__(self, name, env, offset=None) -> None:

        VerticalGraspBasicTask.update_config(self)

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
        


        self.dt = 1.0 / 120
        # self.dt = 0.166
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        self.test = self._cfg['test']

        self.code_backup_path = self._cfg['code_backup_path']
        self.asset_path = self._cfg['asset_path']
        self.exp_name= self._cfg['experiment']
        self.experiment_code_dir = os.path.join(self.code_backup_path, self.exp_name)
       
        if not self.test:
            with open('./omniisaacgymenvs/cfg/training_params/params.yaml', 'r') as file:
                self.training_params = yaml.safe_load(file)  # Use safe_load for security reasons
        else:
            assert os.path.exists(self.experiment_code_dir)
            with open(os.path.join(self.experiment_code_dir, 'cfg/training_params/params.yaml'), 'r') as file:
                self.training_params = yaml.safe_load(file)  # Use safe_load for security reasons
    
        self.object_name = self.training_params['object']
        self.object_suffix = self.training_params['object_suffix']
        self.hand_usd = self.training_params.get("hand_usd", "R_inspire_sh_property_11.usd")
        
        self.hand_dof_default_pos = self.training_params['hand_dof_default_pos']
        self.random_init = self.training_params['random_init']
        # torch.tensor(self.training_params['reset_lower'], dtype=torch.float, device=self.device)
        self.enable_trajectory_recording = self._cfg['enable_trajectory_recording']
        self.trajectory_recording = False
        if self.test:
            if self.enable_trajectory_recording:
                self.trajectory_recording = True
                self.trajectory_dir = self._cfg['trajectory_dir']
                os.makedirs(self.trajectory_dir,exist_ok=True)
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
        # self.object_init_pc = np.load('/home/wenbo/Obj_Asset/011_banana/banana.npy',allow_pickle=True)
        self.object_init_pc = np.load(os.path.join(self.asset_path, 'Obj_Asset', self.object_name, f'{self.object_name}.npy'),allow_pickle=True)
        # self.object_init_pc = np.load('/home/wenbo/Obj_Asset/cube_055/cube_055.npy',allow_pickle=True)
        # self.object_init_pc = np.load('/home/wenbo/Obj_Asset/orange/orange.npy',allow_pickle=True)
        # self.object_init_pc = np.load('/home/wenbo/Car/pc_fps1024_008.npy',allow_pickle=True)
        theta = torch.tensor(torch.pi / 2)  # change rotation representation to usd format

        object_pc_init_rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(theta), -torch.sin(theta)],
            [0, torch.sin(theta), torch.cos(theta)]
        ]).float()
        self.object_init_pc = torch.tensor(self.object_init_pc).float()
        self.object_init_pc *= 1.0 # Bottle 0.8
        # self.object_init_pc = torch.matmul(self.object_init_pc, object_pc_init_rotation_matrix.T) # rot for bottle pc
        self.object_init_pc = self.object_init_pc.to(self._device).unsqueeze(0).repeat([self.num_envs, 1, 1])
        # self.object_init_pc = torch.tensor(self.object_init_pc, device=self._device).float().unsqueeze(0).repeat([self.num_envs, 1, 1])
        
        # self.get_goal()
        # self.get_table(self.hand_start_translation, self.pose_dy, self.pose_dz)
        from pxr import UsdGeom, Gf, Usd
        point_cloud_prim_path = "/World/PointCloud"
        self.point_cloud_prim = UsdGeom.Points.Define(self._stage, point_cloud_prim_path)

        super().set_up_scene(scene, filter_collisions=False)

        self._hands = self.get_hand_view(scene)
        scene.add(self._hands)
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object", # bottle & car
            # prim_paths_expr="/World/envs/env_.*/object", # coke_can
            name="object_view",
            reset_xform_properties=False,
            # masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
            # masses=torch.tensor([0.10] * self._num_envs, device=self.device), # bottle
            # masses=torch.tensor([0.39] * self._num_envs, device=self.device), # coke_can
            masses=torch.tensor([self.training_params['mass']] * self._num_envs, device=self.device), # banana
            # masses=torch.tensor([1.0] * self._num_envs, device=self.device), # banana
        )
        scene.add(self._objects)
        # self._objects._non_root_link = True
        # self._goals = RigidPrimView(
        #     prim_paths_expr="/World/envs/env_.*/goal/object", name="goal_view", reset_xform_properties=False
        # )
        # self._goals._non_root_link = True  # hack to ignore kinematics
        # scene.add(self._goals)
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
        # self.object_usd_path = f"/home/wenbo/Obj_Asset/011_banana/Banana_col_material.usd"
        # self.object_usd_path = f"/home/wenbo/Obj_Asset/realsense_box/realsense_box_material.usd"
        # self.object_usd_path = f"/home/wenbo/Obj_Asset/realsense_box/realsense_box_03.usd"
        # self.object_usd_path = f"/home/wenbo/Obj_Asset/cube_055/cube_055_col.usd"
        # self.object_usd_path = f"/home/wenbo/Obj_Asset/cube_055/cube_055_col_material.usd"
        # self.object_usd_path = f"/home/wenbo/Obj_Asset/orange/orange_material.usd"
        # self.object_usd_path = f"/home/wenbo/Car/Car_col.usd"
        if self.object_suffix is not None:
            # self.object_usd_path = f"/home/wenbo/Obj_Asset/{self.object_name}/{self.object_name}_{self.object_suffix}.usd"
            self.object_usd_path = os.path.join(self.asset_path, 'Obj_Asset', self.object_name, f'{self.object_name}_{self.object_suffix}.usd')
        else:
            # self.object_usd_path = f"/home/wenbo/Obj_Asset/{self.object_name}/{self.object_name}.usd"
            self.object_usd_path = os.path.join(self.asset_path, 'Obj_Asset', self.object_name, f'{self.object_name}.usd')
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        self.obj = XFormPrim(
            # prim_path=self.default_zero_env_path + "/object/object",
            prim_path=self.default_zero_env_path + "/object",
            name="object",
            translation=self.object_start_translation,
            # translation= self.table_start_translation + torch.tensor([0.,0.,0.7]).cuda(),
            orientation=self.object_start_orientation,
            # scale=self.object_scale,
            # scale=self.object_scale * 0.8, # bottle
            scale=self.object_scale * 1.0,
        )
        self._sim_config.apply_articulation_settings(
            "object", get_prim_at_path(self.obj.prim_path), self._sim_config.parse_actor_config("object")
        )

    def post_reset(self):
        source_file_list = ['']
        self.num_hand_dofs = self._hands.num_dof
        cfg_folder = './omniisaacgymenvs/cfg'
        task_folder = './omniisaacgymenvs/tasks'
        if not self.test:
            os.makedirs(self.experiment_code_dir, exist_ok= True)
            shutil.copytree(cfg_folder,self.experiment_code_dir+'/cfg', dirs_exist_ok=True)
            shutil.copytree(task_folder,self.experiment_code_dir+'/tasks', dirs_exist_ok=True)
        self.reset_lower = torch.tensor(self.training_params['reset_lower'], dtype=torch.float, device=self.device)
        self.reset_upper = torch.tensor(self.training_params['reset_upper'], dtype=torch.float, device=self.device)
        
        self.hand_dof_default_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        self.hand_dof_default_pos[:] = torch.tensor(self.training_params['hand_dof_default_pos'], dtype=torch.float, device=self.device)
        # self.hand_dof_default_pos[2] = -0.023 # rule base
        # self.hand_dof_default_pos[2] = +0.05 # cube and orange
        # self.hand_dof_default_pos[2] = +0.04 # banana
        # self.hand_dof_default_pos[2] = +0.08 # realsense_box
        # self.hand_dof_default_pos[4] = -3.1416 / 2
        # self.hand_dof_default_pos[5] = -3.1416 / 4 # rule base tilt
        # self.hand_dof_default_pos[10] = 1.3
        # self.hand_dof_default_pos[15] = -0.1
        self.actuated_dof_indices = self._hands.actuated_dof_indices
        self.finger_dof_indices = self.actuated_dof_indices[6:]
        self.base_trans_dof_indices = self.actuated_dof_indices[:3]
        self.base_rot_dof_indices = self.actuated_dof_indices[3:6]
        
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets[:,:] = self.hand_dof_default_pos
        self.cur_targets[:,:] = self.hand_dof_default_pos
        
        
        self.action_buf = []
        dof_limits = self._hands.get_dof_limits()
        self.hand_dof_lower_limits, self.hand_dof_upper_limits = torch.t(dof_limits[0].to(self.device))
        self.hand_dof_lower_limits[5] = -3.1416 / 3 
        self.hand_dof_upper_limits[5] = 0
        # self.hand_dof_pregrasp_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        # self.hand_dof_pregrasp_pos[6:10] = torch.pi / 6 # index, middle, pinky, ring
        # self.hand_dof_pregrasp_pos[6:10] = torch.pi / 6 # index, middle, pinky, ring
        self.hand_dof_pregrasp_pos = self.hand_dof_default_pos
        self.hand_dof_pos = self._hands.get_joint_positions()
        
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
        self.goal_pos[:, 2] += 0.15
        self.goal_rot = self.object_init_rot.clone()

        self.goal_init_pos = self.goal_pos.clone()
        self.goal_init_rot = self.goal_rot.clone()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        if self.trajectory_recording:
            self.record_buf = []
            self.record_traj_id = 0
            self.reset_record_buf()
        self.init_reset = False
        self.reset_idx(indices)
        self.init_reset = True
        
        self.get_object_goal_observations()
        self.object_pos_static = torch.zeros_like(self.object_pos)
        self.object_pos_cache = torch.zeros_like(self.object_pos).unsqueeze(0).repeat([2,1,1])
        self.object_pos_cache[0] = self.object_pos.clone()
        self.object_pos_cache[1] = self.object_pos.clone()
        self.prev_targets = self.cur_targets
        # dof_limits = self._hands._physics_view.get_dof_limits()
        max_forces = self._hands._physics_view.get_dof_max_forces()
        # dof_limits[:,-2:,0] = -0.1
        # dof_limits[:,-2:,1] = 1.7
        # # max_forces[:,6] = 1.0 # constrained
        # # max_forces[:,:6] *= 1000 # constrained
        # # max_forces[:,6:] *= 30 # constrained
        
        # max_forces[:,6:10] = 0.1
        # max_forces[:,15:] = 100000  # 
        # max_forces[:,15:] = 1e8  # 
        max_forces[:,15] *=3 # 
        max_forces[:,10] *=3 # 
        max_forces[:,6:] *=2
        # max_forces[:,:6] = 1000
        masses = self._hands._physics_view.get_masses()
        masses[:,7:] *=10
        
        self._hands._physics_view.set_masses(masses,indices = indices.detach().cpu())
        self._hands._physics_view.set_dof_max_forces(max_forces,indices = indices.detach().cpu())
        # self._hands._physics_view.set_dof_limits(dof_limits,indices = indices.detach().cpu())
        max_vel = self._hands._physics_view.get_dof_max_velocities()
        # max_vel[:,11:15] /=100
        # max_vel[:,16:] = 0
        
        # max_vel[:,6:] /= 100
        
        # thumb_scale = 1.5
        # max_vel[:,-3] *= 1 * thumb_scale
        # max_vel[:,-2] *= 1.6 * thumb_scale
        # max_vel[:,-1] *= 1.6*2.4 * thumb_scale
        # max_vel[:,-3:] = 1.7453e4
        # max_vel[:,6:15] *= 0.01
        self._hands._physics_view.set_dof_max_velocities(max_vel,indices = indices.detach().cpu())
        # gravity = self._hands.get_body_disable_gravity()
        # gravity*=0
        # gravity = 1-gravity
        # self._hands.set_body_disable_gravity(gravity)
        stiffness = self._hands._physics_view.get_dof_stiffnesses()
        # stiffness[:,:6] *= 100
        # stiffness[:,6:] =stiffness[:,6]
        
        # stiffness[:,6:] *= 10
        # stiffness[:,15] *= 3
        
        # stiffness[:,16:] = stiffness[:,15]
        # stiffness[:,15] *=10  # 
        # stiffness[:,-3:] *= 100
        # stiffness[:,6:] *= 100
        # stiffness *= 100
        # stiffness[:11] = 5.7296e+8
        # stiffness[:12] = 5.7296e+8
        # stiffness[:13] = 5.7296e+8
        # stiffness[:14] = 5.7296e+8
        # stiffness[:16] = 5.7296e+8
        # stiffness[:17] = 5.7296e+8
        self._hands._physics_view.set_dof_stiffnesses(stiffness,indices = indices.detach().cpu())
        damping = self._hands._physics_view.get_dof_dampings()
        # stiffness[:,6:] *= 0.1
        # damping[:,6:]  = stiffness[:,6:] * 0.1
        damping[:,6:] *=1
        # damping[:,6:] *= 0.1
        self._hands._physics_view.set_dof_dampings(damping, indices = indices.detach().cpu())
        armatures = self._hands._physics_view.get_dof_armatures()
        armatures[:,6:] = 0.01
        self._hands._physics_view.set_dof_armatures(armatures, indices = indices.detach().cpu())
        # max_vel = self._hands._physics_view.get_dof_max_velocities()
        # max_vel[0,6]*=10
        # max_vel[0,7]*=10
        # max_vel[0,8]*=10
        # max_vel[0,9]*=10
        # max_vel[0,15]*=10
        # self._hands._physics_view.set_dof_max_velocities(max_vel,indices = indices.detach().cpu())
        

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def update_object_pos_static(self):
        update_flag = self.progress_buf==10 # update at step 10
        self.object_pos_static[update_flag==True] = self.object_pos[update_flag==True].detach().clone()
        
        
    def update_object_pos_cache(self, object_pos_cache, object_pos):
        # Move current X[1] (current step) to X[0] (previous step)
        object_pos_cache[0] = object_pos_cache[1].clone()
        
        # Update X[1] with the new data (current step)
        object_pos_cache[1] = object_pos.clone()

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
            torch.tensor(self.finger_dof_indices),
            self.object_pos,
            self.object_rot,
            self.object_pos_cache,
            self.object_pos_static,
            self.goal_pos,
            self.goal_rot,
            self.hand_dof_pos, # current joint states
            self.hand_dof_pregrasp_pos, # default joint states
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
            torch.tensor(self.test),
            self.hand_dof_upper_limits,
            self.training_params,
            self.reset_lower,
            self.reset_upper
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
        
        actions[env_ids] = 0
                
       
        # # relative control
        # self.actions = actions.clone().to(self.device)
        # joint_indices = torch.tensor(self.base_trans_dof_indices + self.base_rot_dof_indices + self.finger_dof_indices)
        # hand_dof = self.hand_dof_pos
        # target_hand_dof = hand_dof[:, joint_indices]
        
        # # lower_limit = torch.tensor([-1, -1, -0.05]).cuda()
        # lower_limit = torch.tensor([-1, -1, -0.02]).cuda()
        # upper_limit = torch.tensor([1, 1, 1]).cuda()

        # target_hand_dof[:,:3] += actions[:,:3] * 0.02 # 0.05 # 0.015
        # # target_hand_dof[:,:3] += actions[:,:3] * 0.005 # 0.05 # 0.015
        # target_hand_dof[:,3:6]  = self.hand_dof_default_pos[3:6] # rotation set to default pose
        # target_hand_dof[:,6:10] += actions[:,6:10]* 0.2 # 0.5 # 0.15
        # target_hand_dof[:,10:12] += actions[:,10:12]* 0.2 # 0.5 # 0.15

        # self.cur_targets[:, joint_indices] = 1.0 * target_hand_dof + 0. * self.prev_targets[:, joint_indices] # 0.2 / 0.8
        # self.cur_targets[:,self.base_trans_dof_indices] = torch.clamp(self.cur_targets[:,self.base_trans_dof_indices], lower_limit, upper_limit) # base action clamp
        # self.cur_targets[:,self.base_rot_dof_indices] = torch.clamp(self.cur_targets[:,self.base_rot_dof_indices], self.hand_dof_lower_limits[self.base_rot_dof_indices], self.hand_dof_upper_limits[self.base_rot_dof_indices])
        # self.cur_targets[:,self.finger_dof_indices] = torch.clamp(self.cur_targets[:,self.finger_dof_indices], 
        #                                                 self.hand_dof_lower_limits[self.finger_dof_indices], 
        #                                                 self.hand_dof_upper_limits[self.finger_dof_indices] * 1.0) # 1.0 to 0.8, to avoid finger break issue
        # self._hands.set_joint_position_targets(self.cur_targets[:, joint_indices],joint_indices = joint_indices)
        # self.prev_targets = self.cur_targets
        
        
           
        # # finger absolute position control
        self.actions = actions.clone().to(self.device)
        joint_indices = torch.tensor(self.base_trans_dof_indices + self.base_rot_dof_indices + self.finger_dof_indices)
        hand_dof = self.hand_dof_pos
        target_hand_dof = hand_dof[:, joint_indices]
        
        # lower_limit = torch.tensor([-1, -1, -0.05]).cuda()
        lower_limit = torch.tensor([-1, -1, -0.02]).cuda()
        upper_limit = torch.tensor([1, 1, 1]).cuda()

        target_hand_dof[:,:3] += actions[:,:3] * 0.02 # 0.05 # 0.015
        # target_hand_dof[:,:3] += actions[:,:3] * 0.005 # 0.05 # 0.015
        # target_hand_dof[:,3:6]  = self.hand_dof_default_pos[3:6] # fix all rot dof
        target_hand_dof[:,3]  = self.hand_dof_default_pos[3] # fix roll
        target_hand_dof[:,4]  = self.hand_dof_default_pos[4] # fix pitch
        target_hand_dof[:,5]  = self.hand_dof_default_pos[5] # fix yaw
        # target_hand_dof[:,5] += actions[:,5] * 0.1
        # target_hand_dof[:,5]  = self.hand_dof_default_pos[5] # fix yaw
        target_hand_dof[:,6:] = scale(actions[:,6:],self.hand_dof_lower_limits[self.finger_dof_indices],self.hand_dof_upper_limits[self.finger_dof_indices])
        target_hand_dof[:,10] = self.hand_dof_upper_limits[10] # thumb yaw fix

        self.cur_targets[:, joint_indices] = 1.0 * target_hand_dof + 0. * self.prev_targets[:, joint_indices] # 0.2 / 0.8
        self.cur_targets[:,self.base_trans_dof_indices] = torch.clamp(self.cur_targets[:,self.base_trans_dof_indices], lower_limit, upper_limit) # base action clamp
        self.cur_targets[:,self.base_rot_dof_indices] = torch.clamp(self.cur_targets[:,self.base_rot_dof_indices], self.hand_dof_lower_limits[self.base_rot_dof_indices], self.hand_dof_upper_limits[self.base_rot_dof_indices])
        self.cur_targets[:,self.finger_dof_indices] = torch.clamp(self.cur_targets[:,self.finger_dof_indices], 
                                                        self.hand_dof_lower_limits[self.finger_dof_indices], 
                                                        self.hand_dof_upper_limits[self.finger_dof_indices] * 1.0) # 1.0 to 0.8, to avoid finger break issue
        self._hands.set_joint_position_targets(self.cur_targets[:, joint_indices],joint_indices = joint_indices)
        self.prev_targets = self.cur_targets
        
    
        
        # # # for rule-base grasping
        
        # self.actions = actions.clone().to(self.device)
        # joint_indices = torch.tensor(self.base_trans_dof_indices + self.base_rot_dof_indices + self.finger_dof_indices)
        # hand_dof = self.hand_dof_pos.clone()
        # # print(hand_dof[:,10])
    
        # hand_dof[:,:] = self.hand_dof_default_pos
        # target_hand_dof = hand_dof[:, joint_indices]

        # # lower_limit = torch.tensor([-1, -1, -0.05]).cuda()
        # lower_limit = torch.tensor([-1, -1, -0.02]).cuda()
        # upper_limit = torch.tensor([1, 1, 1]).cuda()


        
        # # print(self._hands.get_measured_joint_efforts())
        # np.array(self._hands._dof_names)[joint_indices]
        # # target_hand_dof[:,2] += self.progress_buf * -0.01
        
        
        
        # stage1_timestep = 30
        # stage2_timestep = 100
        # if self.progress_buf[0] <stage1_timestep:
        #     # target_hand_dof[:, 6:] = (self.progress_buf).unsqueeze(-1) * 0.02 *self.hand_dof_upper_limits[self.finger_dof_indices]
        #     target_hand_dof[:,2] = self.progress_buf * -0.005
        #     target_hand_dof[:,10] = 1.3
        #     # target_hand_dof[:,-1] = (self.progress_buf).unsqueeze(-1) * 0.1 *self.hand_dof_upper_limits[15]
        # elif self.progress_buf[0] <stage2_timestep:
        #     target_hand_dof[:,2] = stage1_timestep * -0.005
        #     rand_floats_trans = torch_rand_float(-1.0, 1.0, (len(target_hand_dof), 3), device=self.device) *0.02
        #     target_hand_dof[:,:3] += rand_floats_trans[:,:3]
        #     target_hand_dof[:, 6:] = (self.progress_buf).unsqueeze(-1) * 0.02 *self.hand_dof_upper_limits[self.finger_dof_indices]
        #     target_hand_dof[:,10] = 1.3
        # else:
        #     target_hand_dof[:,2] = stage1_timestep * -0.005 + self.progress_buf * 0.005
        #     target_hand_dof[:,6:] = self.hand_dof_upper_limits[self.finger_dof_indices]
        #     target_hand_dof[:,10] = 1.3
        # # print(target_hand_dof[:,2])
        # # else:
        # #     target_hand_dof[:, 6:] += stage1_timestep* 0.02 *self.hand_dof_upper_limits[self.finger_dof_indices]
        # #     target_hand_dof[:,2] += (self.progress_buf -stage1_timestep) * -0.01
        
        # # stage1_timestep = 300
        # # if self.progress_buf[0] <stage1_timestep:
        # #     target_hand_dof[:, 6:] = (self.progress_buf).unsqueeze(-1) * 0.02 *self.hand_dof_upper_limits[self.finger_dof_indices]
        # #     # target_hand_dof[:,-2] = (self.progress_buf).unsqueeze(-1) * 0.1 *self.hand_dof_upper_limits[10]
        # #     target_hand_dof[:,10] = 1.3
        # #     # target_hand_dof[:,-1] = (self.progress_buf).unsqueeze(-1) * 0.1 *self.hand_dof_upper_limits[15]
            
  
        # # else:
        # #     target_hand_dof[:, 6:] += stage1_timestep* 0.02 *self.hand_dof_upper_limits[self.finger_dof_indices]
        # #     target_hand_dof[:,2] += (self.progress_buf -stage1_timestep) * -0.01
        # target_hand_dof[:,:3] = torch.clamp(target_hand_dof[:,:3], lower_limit, upper_limit) # base action clamp
        # target_hand_dof[:,6:] = torch.clamp(target_hand_dof[:,6:], 
        #                                     self.hand_dof_lower_limits[self.finger_dof_indices], 
        #                                     self.hand_dof_upper_limits[self.finger_dof_indices] * 1.0) 
        # self._hands.set_joint_position_targets(target_hand_dof,joint_indices = joint_indices) # direct position control, wo smoothing

            

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

        # self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def reset_record_buf(self):
        
        self.record_buf = {}
        self.hand_dof_pos_buf = []
        self.hand_dof_velocities_buf = []
        self.action_buf = []
        self.cur_targets_buf = []
        self.right_hand_base_pose_buf = []
        self.right_hand_palm_pose_buf = []
        self.object_pose_buf = []
        self.obj_to_goal_dist_buf = []
        self.hold_flag_buf = []
        self.lowest_buf = []
        self.goal_reach_buf = []
        self.goal_pos
        self.hand_dof_default_pos
        self.hand_start_translation
        self.hand_start_orientation
        
    def gather_record_buf(self):
        self.hand_dof_pos_buf = torch.stack(self.hand_dof_pos_buf).detach().cpu()
        self.hand_dof_velocities_buf = torch.stack(self.hand_dof_velocities_buf).detach().cpu()
        self.action_buf = torch.stack(self.action_buf).detach().cpu()
        self.cur_targets_buf = torch.stack(self.cur_targets_buf).detach().cpu()
        self.right_hand_base_pose_buf = torch.stack(self.right_hand_base_pose_buf).detach().cpu()
        self.right_hand_palm_pose_buf = torch.stack(self.right_hand_palm_pose_buf).detach().cpu()
        self.hold_flag_buf = torch.stack(self.hold_flag_buf).detach().cpu()
        self.lowest_buf = torch.stack(self.lowest_buf).detach().cpu()
        self.goal_reach_buf = torch.stack(self.goal_reach_buf).detach().cpu()
        self.object_pose_buf = torch.stack(self.object_pose_buf).detach().cpu()
        self.obj_to_goal_dist_buf = torch.stack(self.obj_to_goal_dist_buf).detach().cpu()
        self.record_buf['hand_dof_pos_buf'] = self.hand_dof_pos_buf # [200, 1, 18]
        self.record_buf['hand_dof_velocities_buf'] = self.hand_dof_velocities_buf # [200, 1, 18]
        self.record_buf['action_buf'] = self.action_buf# [200, 1, 12] 
        self.record_buf['cur_targets_buf'] = self.cur_targets_buf# [200, 1, 18]
        self.record_buf['right_hand_base_pose_buf'] = self.right_hand_base_pose_buf# [200, 1, 7]
        self.record_buf['right_hand_palm_pose_buf'] = self.right_hand_palm_pose_buf# [200, 1, 7]
        self.record_buf['object_pose_buf'] = self.object_pose_buf
        self.record_buf['obj_to_goal_dist_buf'] = self.obj_to_goal_dist_buf
        self.record_buf['hold_flag_buf'] = self.hold_flag_buf# [200, 1]
        self.record_buf['lowest_buf'] = self.lowest_buf# [200, 1]
        self.record_buf['goal_reach_buf'] = self.goal_reach_buf# [200, 1]
        self.record_buf['goal_pos'] = self.goal_pos.detach().cpu()# [1, 3]
        self.record_buf['hand_dof_default_pos'] = self.hand_dof_default_pos.detach().cpu()# [18]
        self.record_buf['hand_start_translation'] = self.hand_start_translation.detach().cpu()# [3]
        self.record_buf['hand_start_orientation'] = self.hand_start_orientation.detach().cpu()# [4]
    
    def reset_idx(self, env_ids):
        if self.trajectory_recording:
            if self._env.video_recorder is not None:
                # self.video_path = os.path.join(self._cfg['recording_dir'], f"{self.record_traj_id:04d}.mp4")
                # self.trajectory_path = os.path.join(self.trajectory_dir,f"{self.record_traj_id:04d}.npy")
                self.trajectory_path = os.path.join(self.trajectory_dir,f"step-{self.record_traj_id*200}.npy")
                self.gather_record_buf()
                np.save(self.trajectory_path, self.record_buf)
                self.reset_record_buf()
                self.record_traj_id+=1
                
        indices = env_ids.to(dtype=torch.int32)
        self.action_buf = []
        self.reset_target_pose(env_ids) # set reset_goal_buf to 0 

        # reset object
        # new_object_pos = (
        #     self.object_init_pos[env_ids] + self.reset_position_noise * rand_floats[:, 0:3] + self._env_pos[env_ids]
        # )  # add world env pos

        # new_object_rot = randomize_rotation(
        #     rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        rand_floats_trans = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device) *0.05
        rand_floats_trans[:,2] = 0
        
        new_object_rot = self.object_start_orientation.unsqueeze(0).repeat([len(env_ids),1])
        if self.random_init:
            new_object_pos = self.object_start_translation.unsqueeze(0).repeat([len(env_ids),1]) + rand_floats_trans+  self._env_pos[env_ids] # random init pos
        else:
            new_object_pos = self.object_start_translation.unsqueeze(0).repeat([len(env_ids),1]) + self._env_pos[env_ids] # fixed init pos
        
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


# @torch.jit.script
def compute_supgrasp_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    finger_dof_indices,
    object_pos,
    object_rot,
    object_pos_cache,
    object_pos_static,
    goal_pos,
    goal_rot, # 
    hand_dof_pos,
    hand_dof_pregrasp_pos,
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
    test,
    hand_dof_upper_limits,
    training_params,
    reset_lower,
    reset_upper,
):
 
    # goal_hand_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
    # goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
    # goal_hand_dist = torch.norm(object_pos - right_hand_palm_pos, p=2, dim=-1) # zl *
    # goal_hand_dist = right_hand_pc_dist # palm to object
    # goal_hand_dist = right_hand_finger_pc_dist /5
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]
    # max_finger_dist = 0.3 
    max_finger_dist = training_params['max_finger_dist']
    # max_hand_dist = 0.045
    # max_hand_dist = 0.05 # cube and orange, realsense_box
    max_hand_dist = training_params['max_hand_dist'] # cube and orange, realsense_box
    # max_hand_dist = 0.08 # banana
    # max_goal_dist = 0.05
    max_goal_dist = training_params['max_goal_dist']
    # hand_up_goal_dist = 1.0
    hand_up_goal_dist_weight = training_params['hand_up_goal_dist_weight']
    obj_pos_change_weight = training_params['obj_pos_change_weight']
    wrapped_hand_dof_pos = ( hand_dof_pos[:,finger_dof_indices] + torch.pi) % (2 * torch.pi) - torch.pi
    # delta_init_qpos_value = torch.norm(wrap_to_pi(hand_dof_pos[:,finger_dof_indices]) - hand_dof_pregrasp_pos[finger_dof_indices], p=1, dim=-1) # pass
    delta_init_qpos_value = torch.norm(wrapped_hand_dof_pos - hand_dof_pregrasp_pos[finger_dof_indices], p=1, dim=-1) # pass
    right_hand_dist = right_hand_pc_dist
    right_hand_finger_dist = right_hand_finger_pc_dist
    goal_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1) # zl *
    goal_hand_dist = torch.norm(goal_pos - right_hand_palm_pos, p=2, dim=-1)
    
    compute_obj_pos_change_flag = progress_buf >10
    
    previous_obj_pose = object_pos_static
    obj_pos_change = torch.norm(previous_obj_pose[:,:2] - object_pos[:,:2],dim=1)
    obj_pos_change_rew = torch.zeros_like(goal_dist)
    obj_pos_change_rew = torch.where(lowest < 0.01, obj_pos_change_weight * obj_pos_change, obj_pos_change_rew)
    obj_pos_change_rew*= compute_obj_pos_change_flag
    # print(object_pos)
    # if abs(obj_pos_change).max() >10:
    #     print('123')
    # print(obj_pos_change_rew)
    hold_value = 2
    hold_flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
    
        # # ---------------------- Reward After Holding ---------------------- # #
    # Distanc from object pos to goal target pos
    goal_rew = torch.zeros_like(goal_dist)
    goal_rew = torch.where(hold_flag == hold_value, 1.0 * (0.9 - 2.0 * goal_dist), goal_rew)
    # Distance from hand pos to goal target pos
    hand_up = torch.zeros_like(goal_dist)
    # hand_up = torch.where(lowest >= 0.01, torch.where(hold_flag == hold_value, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.01, torch.where(hold_flag == hold_value, 0.1 + 1.0 * actions[:, 2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.20, torch.where(hold_flag == hold_value, 0.2 - goal_hand_dist * 0 + hand_up_goal_dist_weight * (0.2 - goal_dist), hand_up), hand_up)
    # Already hold the object and Already reach the goal
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(hold_flag == hold_value, torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
        
    right_hand_dist_weight = training_params['right_hand_dist_weight']
    delta_init_qpos_value_weight = training_params['delta_init_qpos_value_weight']
    right_hand_finger_dist_weight = training_params['right_hand_finger_dist_weight']
    goal_dist_weight = training_params['goal_dist_weight']
    
    init_reward = delta_init_qpos_value_weight* delta_init_qpos_value
    init_reward += right_hand_dist_weight * right_hand_dist
    init_reward += goal_dist_weight * goal_dist
    init_reward += obj_pos_change_rew
    

    right_hand_dist_weight_s2 = training_params['right_hand_dist_weight_s2']
    goal_rew_weight = training_params['goal_rew_weight']
    hand_up_weight = training_params['hand_up_weight']
    bonus_weight = training_params['bonus_weight']
    
    grasp_reward = right_hand_finger_dist_weight * right_hand_finger_dist
    grasp_reward += right_hand_dist_weight_s2 * right_hand_dist
    grasp_reward += goal_dist_weight * goal_dist
    grasp_reward += goal_rew_weight * goal_rew 
    grasp_reward += hand_up_weight * hand_up
    grasp_reward += bonus_weight * bonus
    grasp_reward += obj_pos_change_rew
    
    reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)
    
    resets = reset_buf
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    

    
    
    # lower_bound = torch.tensor([-0.42, -0.65, 0.025]).cuda()
    # upper_bound = torch.tensor([0.58, 0.35, 1.0]).cuda()
    # lower_bound = torch.tensor([-0.39, -0.68, 0.025]).cuda() # bottle
    # lower_bound = torch.tensor([-0.39, -0.68, 0.035]).cuda() Coke_can
    # lower_bound = torch.tensor([-0.5, -0.5, 0.00]).cuda() # other objects
    # lower_bound = torch.tensor([-0.5, -0.5, 0.035]).cuda() # realsense_box
    # upper_bound = torch.tensor([0.5, 0.5, 1.0]).cuda()
    lower_bound = reset_lower
    upper_bound = reset_upper
    # hand_out_of_box = (object_pos < lower_bound).sum(1) + (object_pos > upper_bound).sum(1) # object
    hand_out_of_box = (right_hand_palm_pos < lower_bound).sum(1) + (right_hand_palm_pos > upper_bound).sum(1) # hand
    obj_out_of_box = (object_pos < lower_bound).sum(1) + (object_pos > upper_bound).sum(1) # hand
    if hand_out_of_box.sum() !=0:
        print('out of box')
    if obj_out_of_box.sum() !=0:
        print('obj out of box')
    # resets = torch.where(object_pos[:,2] <0.025, torch.ones_like(resets), resets)
    # reward = torch.where(object_pos[:,2] <0.025, reward -1, reward) # object fall reward
    if not test:
        resets = torch.where(hand_out_of_box !=0 , torch.ones_like(resets), resets)
        resets = torch.where(obj_out_of_box !=0 , torch.ones_like(resets), resets)
        reward = torch.where(obj_out_of_box !=0, reward -1, reward) # object fall reward
    
    # print(reward.detach().cpu())
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


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

# @torch.jit.script
def wrap_to_pi(angles):
    # Wrap the angles to the range [-pi, pi]
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi