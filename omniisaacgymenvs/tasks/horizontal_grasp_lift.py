# Ref: omniisaacgymenvs/tasks/shadow_hand.py

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omni.isaac.gym.vec_env import VecEnvBase

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage

from omniisaacgymenvs.robots.articulations.movable_inspire_R import MovableInspireHandR
from omniisaacgymenvs.robots.articulations.views.movable_inspire_R_view import MovableInspireRView

from omniisaacgymenvs.tasks.shared.horizontal_grasp_basic import HorizontalGraspBasicTask

from omni.isaac.core.utils.torch import *
from omniisaacgymenvs.tasks.utils.torch_jit_utils import *
import torch
import numpy as np

class HorizontalGraspTask(HorizontalGraspBasicTask):
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
        self._num_actions = 12
        # self._num_actions = 9

        self.obs_type = self._task_cfg["env"]["observationType"]
        if self.obs_type == "full_no_vel":
            self._num_observations = 42 # 42 # 30
        else:
            # self._num_observations = 54 # 42
            # self._num_observations = 110 #  xyz
            # self._num_observations = 119 # xyz rpy
            # self._num_observations = 134 # bottle
            self._num_observations = 188 # xyz rpy full joint states
        # 12: inspire hand joints position (action space)
        # 12: inspire hand joints velocity
        # 3: object position
        # 4: object rotation
        # 3: goal position
        # 4: goal rotation
        # 4: goal relative rotation
        # 12: previous action

        #???
        self._num_states = 0
        self.repose_z = True # zl  = self.cfg['env']['repose_z']
        self.link_names = ['R_hand_base_link', 'R_index_proximal', 'R_index_intermediate', 'R_middle_proximal', 'R_middle_intermediate', 
                           'R_pinky_proximal', 'R_pinky_intermediate', 'R_ring_proximal', 'R_ring_intermediate', 'R_thumb_proximal_base', 
                           'R_thumb_proximal', 'R_thumb_intermediate', 'R_thumb_distal', 'R_movable_root_link', 'R_movable_basex', 
                           'R_movable_basey', 'R_movable_basez', 'R_movable_rot0', 'R_movable_rot1', 'R_movable_rot2']
        self.link_dict = {
            "R_hand_base_link":0, # palm
            "R_index_proximal":1,
            "R_index_intermediate":2,
            "R_middle_proximal":3,
            "R_middle_intermediate":4,
            "R_pinky_proximal":5,
            "R_pinky_intermediate":6,
            "R_ring_proximal":7,
            "R_ring_intermediate":8,
            "R_thumb_proximal_base":9,
            "R_thumb_proximal":10,
            "R_thumb_intermediate":11,
            "R_thumb_distal":12,
            "R_movable_root_link":13, # fixed base link
            "R_movable_basex":14,
            "R_movable_basey":15,
            "R_movable_basez":16,
            "R_movable_rot0":17,
            "R_movable_rot1":18,
            "R_movable_rot2":19
        }
        self.fingertip_indices = torch.tensor([2, 4, 6, 8, 12]).to(self.cfg['sim_device'])

        
        HorizontalGraspBasicTask.__init__(self, name=name, env=env)
   
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
        self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        # smaller cube is easier to rotate for inspire hand
        # self.object_scale = torch.tensor([0.9, 0.9, 0.9])
        # self.object_scale = torch.tensor([0.85, 0.85, 0.85])
        # smaller cube is easier to roll by gravity, bigger is easier to be pushed by hand links
        # self.object_scale = torch.tensor([0.80, 0.80, 0.80])
        # self.object_scale = torch.tensor([0.70, 0.70, 0.70])
        # self.object_scale = torch.tensor([0.60, 0.60, 0.60])
        # self.object_scale = torch.tensor([0.50, 0.50, 0.50])
        
        HorizontalGraspBasicTask.update_config(self)


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
    #         # self._num_observations, device=self.device, dtype=torch.float)l
    #     observations = {self._inspire_hand_R.name: {"obs_buf": self.obs_buf}}
    #     return observations

    # def get_starting_positions(self):
    #     self.hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
    #     self.hand_start_orientation = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=self.device)
    #     self.pose_dy, self.pose_dz = -0.39, 0.10
    def get_starting_positions(self):
   
        self.hand_start_translation = torch.tensor([0.01, 0.01, 0.2], device=self.device) # object falls on the ground
        self.hand_start_orientation = torch.tensor([1., 0., 0., 0.], device=self.device)
        self.table_start_translation = torch.tensor([0.01, 0.01, 0.], device=self.device)
    
        # self.object_start_translation = torch.tensor([0.05, -0.10, 0.04], device=self.device) # for static grasping
        # self.object_start_translation = torch.tensor([0.11, -0.18, 0.04], device=self.device) # for fixed init position
        # self.object_start_translation = torch.tensor([0.14, -0.23, 0.04], device=self.device) # for random init position
        self.object_start_translation = torch.tensor([0.14, -0.23, 0.075], device=self.device) # for random init position
        self.object_start_orientation = torch.tensor([0.7071, -0.7071, 0., 0], device=self.device) # vertical to hand init pose

           
        self.pose_dy, self.pose_dz = 0., 0.
        self.pose_dx = 0. # object scale 0.9, close to thumb for a little bit    
    #Ref: omniisaacgymenvs/tasks/franka_deformable.py
    def get_hand(self):
        # add a single robot to the stage
        inpsire_hand = MovableInspireHandR(
            prim_path=self.default_zero_env_path + "/movable_inspire_R", 
            name="movable_inspire_R", 
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
        return MovableInspireRView(prim_paths_expr="/World/envs/.*/movable_inspire_R", name="movable_inspire_R_view")
    
    def get_observations(self):
        self.get_object_goal_observations()
  
        
        # print(self.link_pos[0])
        # print(self.hand_dof_pos[0])
        
        # self._hands.fingers.prim_paths
        # link_names = []
        # for i in range(20):
        #     link_names.append(self._hands.fingers.prim_paths[i].split('/')[-1])
        

        self.object_pose = torch.concat([self.object_pos, self.object_rot], dim=1) # pos in local env
        self.object_pos # have
        self.object_rot # have
        
        new_object_rot = self.object_start_orientation.unsqueeze(0).repeat([self.num_envs,1])
        new_object_pos = self.object_start_translation.unsqueeze(0).repeat([self.num_envs,1]) + self._env_pos
        object_velocities = torch.zeros_like(self.object_init_velocities, dtype=torch.float, device=self.device)

        
        
        # self.object_points = batch_quat_apply(self.object_rot,self.object_init_pc) + self.object_pos.unsqueeze(1)
        # self.object_points = batch_quat_apply(self.object_rot,self.object_init_pc) + self.object_pos.unsqueeze(1)
        # self.object_points = batch_quat_apply_wxyz(self.object_rot,self.object_init_pc) + self.object_pos.unsqueeze(1)
        self.object_points = batch_quat_apply_wxyz(self.object_rot,self.object_init_pc) + self.object_pos.unsqueeze(1)
        self.object_points_vis = self.object_points + self._env_pos.unsqueeze(1) # use for rendering
        indices = torch.randperm(1024)[:256]
        self.object_points_vis = self.object_points_vis[:,indices,:]
        # self.object_handle_pos = self.object_pos # not used 
        # self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04) # not used
        self.object_linvel # have
        self.object_angvel # have



        self.hand_dof_pos = self._hands.get_joint_positions()
        self.hand_dof_velocities = self._hands.get_joint_velocities()
        self.hand_dof_force = self._hands.get_measured_joint_efforts()
        self.link_pose = self._hands.fingers.get_world_poses()
        self.link_vel = self._hands.fingers.get_velocities()
        self.link_vel = self.link_vel.view(self.num_envs,-1, 6)# vel : linear + angular
        self.fingertip_vel = self.link_vel[:,self.fingertip_indices,:] # linear + angular
        

        self.link_pos = self.link_pose[0].view(self.num_envs,-1, 3) - self._env_pos.unsqueeze(1)
        self.link_rot = self.link_pose[1].view(self.num_envs,-1, 4)
        
        idx = 0
        self.right_hand_pos = self.link_pos[:, idx, :]
        self.right_hand_rot = self.link_rot[:, idx, :]

        idx = 13
        self.right_hand_root_pos = self.link_pos[:, idx, :]
        self.right_hand_root_rot = self.link_rot[:, idx, :]
        # print(self.right_hand_root_pos)

        self.right_hand_palm_pos = self.right_hand_pos + quat_apply_wxyz(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.11)
        self.right_hand_palm_pos = self.right_hand_palm_pos + quat_apply_wxyz(self.right_hand_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.015)
        self.right_hand_palm_pos = self.right_hand_palm_pos + quat_apply_wxyz(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.01)
        self.right_hand_palm_rot = self.link_rot[:, 0, :]
        idx = self.link_dict['R_index_intermediate']
        self.right_hand_ff_pos = self.link_pos[:, idx, :]
        self.right_hand_ff_rot = self.link_rot[:, idx, :]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply_wxyz(self.right_hand_ff_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.035)
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply_wxyz(self.right_hand_ff_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.005)
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply_wxyz(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.005)
        idx = self.link_dict['R_middle_intermediate']
        self.right_hand_mf_pos = self.link_pos[:, idx, :]
        self.right_hand_mf_rot = self.link_rot[:, idx, :]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply_wxyz(self.right_hand_mf_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.035)
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply_wxyz(self.right_hand_mf_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.005)
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply_wxyz(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.005)
        idx = self.link_dict['R_ring_intermediate']
        self.right_hand_rf_pos = self.link_pos[:, idx, :]
        self.right_hand_rf_rot = self.link_rot[:, idx, :]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply_wxyz(self.right_hand_rf_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.035)
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply_wxyz(self.right_hand_rf_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.005)
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply_wxyz(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.005)
        idx = self.link_dict['R_pinky_intermediate']
        self.right_hand_lf_pos = self.link_pos[:, idx, :]
        self.right_hand_lf_rot = self.link_rot[:, idx, :]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply_wxyz(self.right_hand_lf_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.035)
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply_wxyz(self.right_hand_lf_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.005)
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply_wxyz(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.005)
        idx = self.link_dict['R_thumb_distal']
        self.right_hand_th_pos = self.link_pos[:, idx, :]
        self.right_hand_th_rot = self.link_rot[:, idx, :]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply_wxyz(self.right_hand_th_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.015)
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply_wxyz(self.right_hand_th_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.015)
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply_wxyz(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.005)
    


        self.right_hand_pc_dist = batch_sided_distance(self.right_hand_palm_pos.unsqueeze(1), self.object_points).squeeze(-1)     
        self.right_hand_pc_dist = torch.where(self.right_hand_pc_dist >= 0.5, 0.5 + 0 * self.right_hand_pc_dist, self.right_hand_pc_dist)

        self.right_hand_finger_pos = torch.stack([self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos], dim=1)
        self.right_hand_finger_rot = torch.stack([self.right_hand_ff_rot, self.right_hand_mf_rot, self.right_hand_rf_rot, self.right_hand_lf_rot, self.right_hand_th_rot], dim=1)
        self.right_hand_finger_pc_dist = torch.sum(batch_sided_distance(self.right_hand_finger_pos, self.object_points), dim=-1)
        self.right_hand_finger_pc_dist = torch.where(self.right_hand_finger_pc_dist >= 3.0, 3.0 + 0 * self.right_hand_finger_pc_dist, self.right_hand_finger_pc_dist)
        self.right_hand_finger_pc_dist_individual = batch_sided_distance(self.right_hand_finger_pos, self.object_points)
        # a = batch_sided_distance(self.right_hand_finger_pos, self.object_points)
        # print('finger: \n')
        # print(a[:4])
        # print('palm')
        # print(self.right_hand_pc_dist)
        # print('base link \n')
        # print(self.right_hand_pos)
        # print('\n')
        # self.hand_root_shift = self.right_hand_pos - self.right_hand_root_pos
        self.hand_root_shift = self.right_hand_palm_pos - self.right_hand_root_pos
        # print(self.hand_root_shift[:4])
        
        
        # hand_points = torch.concat([self.right_hand_pos.unsqueeze(1), self.right_hand_palm_pos.unsqueeze(1), 
        #                        self.right_hand_ff_pos.unsqueeze(1), self.right_hand_mf_pos.unsqueeze(1),
        #                        self.right_hand_rf_pos.unsqueeze(1), self.right_hand_lf_pos.unsqueeze(1), 
        #                        self.right_hand_th_pos.unsqueeze(1)],dim=1)
        # hand_points += self._env_pos.unsqueeze(1)
        # hand_points = hand_points.view(-1,3)
        # object_points = self.object_points_vis.reshape(-1,3)
        # points = torch.concat([hand_points, object_points],dim=0)
        # points = points.detach().cpu().double().numpy()
        # self.point_cloud_prim.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
        # # self.point_cloud_prim.GetWidthsAttr().Set([0.005] * len(points))  # Example size
        # self.point_cloud_prim.GetWidthsAttr().Set([0.01] * len(points))  # Example size
  
  
        # isaacsim_colors = [
        #     (0.9, 0.1, 0.1),    # Red
        #     (0.1, 0.9, 0.1),    # Green
        #     (0.1, 0.1, 0.9),    # Blue
        #     (0.9, 0.9, 0.1),    # Yellow
        #     (0.9, 0.1, 0.9),    # Magenta
        #     (0.1, 0.9, 0.9)     # Cyan
        # ]

        # self.point_cloud_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*color) for color in isaacsim_colors])
        
        goal_points = self.goal_pos + self._env_pos
        points = goal_points.detach().cpu().double().numpy()
        self.point_cloud_prim.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
        self.point_cloud_prim.GetWidthsAttr().Set([0.04] * len(points))  # Example size
        isaacsim_colors = [
            (0.9, 0.1, 0.1),    # Red
            (0.1, 0.9, 0.1),    # Green
            (0.1, 0.1, 0.9),    # Blue
            (0.9, 0.9, 0.1),    # Yellow
            (0.9, 0.1, 0.9),    # Magenta
            (0.1, 0.9, 0.9)     # Cyan
        ]

        self.point_cloud_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*color) for color in isaacsim_colors])
        
        self.compute_full_observations()
        if self.trajectory_recording:
            
            self.compute_frame_record()

        # if self.obs_type == "full_no_vel":
        #     self.compute_full_observations(True)
        # elif self.obs_type == "full":
        #     self.compute_full_observations()
        # else:
        #     print("Unkown observations type!")

        # if self.test:
            # self.record_buf.append(self.obs_buf)
        
        observations = {'object': {"obs_buf": self.obs_buf}}
        return observations
    
    def compute_frame_record(self):
        
     
        lowest = torch.min(self.object_points[:, :, -1], dim=1)[0]
        max_finger_dist = 0.3 
        max_hand_dist = 0.06
        max_goal_dist = 0.05
        hand_up_goal_dist = 1.0
        
   
        
        delta_init_qpos_value = torch.norm(self.hand_dof_pos[:,self.finger_dof_indices] - self.hand_dof_pregrasp_pos[self.finger_dof_indices], p=1, dim=-1) # pass
        right_hand_dist = self.right_hand_pc_dist
        right_hand_finger_dist = self.right_hand_finger_pc_dist
        goal_dist = torch.norm(self.goal_pos - self.object_pos, p=2, dim=-1) # zl *
        goal_hand_dist = torch.norm(self.goal_pos - self.right_hand_palm_pos, p=2, dim=-1)
        
        hold_value = 2
        hold_flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
        
            # # ---------------------- Reward After Holding ---------------------- # #
        # Distanc from object pos to goal target pos
        goal_rew = torch.zeros_like(goal_dist)
        goal_rew = torch.where(hold_flag == hold_value, 1.0 * (0.9 - 2.0 * goal_dist), goal_rew)
        # Distance from hand pos to goal target pos
        hand_up = torch.zeros_like(goal_dist)
        hand_up = torch.where(lowest >= 0.01, torch.where(hold_flag == hold_value, 0.1 + 0.1 * self.actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.20, torch.where(hold_flag == hold_value, 0.2 - goal_hand_dist * 0 + hand_up_goal_dist * (0.2 - goal_dist), hand_up), hand_up)
        # Already hold the object and Already reach the goal
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(hold_flag == hold_value, torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
            
        
        
        init_reward = -0.1* delta_init_qpos_value  + -1.0 * right_hand_dist + -0.5 * goal_dist
        # init_reward = -0.1* delta_init_qpos_value  + -1.0 * right_hand_dist + -0.5 * goal_dist + action_penalty_scale * action_penalty

        grasp_reward = -1.0 * right_hand_finger_dist + -2.0 * right_hand_dist + -0.5 * goal_dist + 1.0 * goal_rew + 2.0 * hand_up + 1.0 * bonus
        # grasp_reward = -1.0 * right_hand_finger_dist + -2.0 * right_hand_dist + -0.5 * goal_dist + 1.0 * goal_rew + 2.0 * hand_up + 1.0 * bonus + action_penalty_scale * action_penalty
        reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)
        self.hand_dof_pos_buf.append(self.hand_dof_pos.clone())
        self.hand_dof_velocities_buf.append(self.hand_dof_velocities.clone())
        self.action_buf.append(self.actions.clone())
        self.cur_targets_buf.append(self.cur_targets.clone())
        # print(self.cur_targets[0,:3])
        self.right_hand_base_pose_buf.append(torch.concatenate([self.right_hand_pos,self.right_hand_rot],dim=1).clone())
        self.right_hand_palm_pose_buf.append(torch.concatenate([self.right_hand_palm_pos,self.right_hand_palm_rot],dim=1).clone())
        self.object_pose_buf.append(torch.concatenate([self.object_pos,self.object_rot],dim=1).clone())
        self.goal_pos
        self.hand_dof_default_pos
        self.hand_start_translation
        self.hand_start_orientation
        self.obj_to_goal_dist_buf.append((self.goal_pos - self.object_pos).clone())
        self.hold_flag_buf.append(hold_flag.clone())
        self.lowest_buf.append(lowest.clone())
        self.goal_reach_buf.append((goal_dist <= max_goal_dist).clone())


    def compute_full_observations(self, no_vel=False):


        # full finger states
        time_encode = torch.cat([self.progress_buf.unsqueeze(-1), compute_time_encoding(self.progress_buf, 28)], dim=-1)
        finger_pos = self.right_hand_finger_pos.reshape(-1, 5 * 3)
        finger_rot = self.right_hand_finger_rot.reshape(-1, 5 *4)
        self.obs_buf[:,0:12] = self.actions #palm z
        self.obs_buf[:, 12:30] = self.hand_dof_pos
        
        self.obs_buf[:, 30:48] = self.hand_dof_velocities
        self.obs_buf[:, 48:51] = self.right_hand_palm_pos
        self.obs_buf[:, 51:55] = self.right_hand_palm_rot
        self.obs_buf[:,55:58] = self.object_pos
        self.obs_buf[:,58:62] = self.object_rot
        self.obs_buf[:, 62:77] = finger_pos
        self.obs_buf[:, 77:97] = finger_rot
        self.obs_buf[:, 97:102] = self.right_hand_finger_pc_dist_individual
        self.obs_buf[:, 102:131] = time_encode
        self.obs_buf[:, 131:134] = self.goal_pos - self.object_pos
        self.obs_buf[:, 134:152] = self.hand_dof_force
        self.obs_buf[:, 152:182] = self.fingertip_vel.view(self.num_envs,-1) # finger distal link linear + angular vel
        self.obs_buf[:, 182:188] = self.object_velocities # linear + angular

        
        
      
    # # ---------------------- Compute Full State: ShadowHand and Object Pose ---------------------- # #
    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta), -self.z_theta)
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state
    
    def unpose_pc(self, pc):
        if self.repose_z:
            num_pts = pc.shape[1]
            return quat_apply(self.unpose_z_theta_quat.view(-1, 1, 4).expand(-1, num_pts, 4), pc)
        return pc

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta), self.z_theta)
        return

    def pose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def compute_time_encoding(time, dimension):
    # Create a tensor for dimension indices: [0, 1, 2, ..., dimension-1]
    div_term = torch.arange(0, dimension, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / dimension)
    div_term = torch.exp(div_term).unsqueeze(0).to(time.device)  # Shape: (1, dimension/2)
    # Apply sin to even indices in the array; 2i
    encoding = torch.zeros(time.shape[0], dimension).to(time.device)
    encoding[:, 0::2] = torch.sin(time.unsqueeze(1) * div_term)
    # Apply cos to odd indices in the array; 2i+1
    encoding[:, 1::2] = torch.cos(time.unsqueeze(1) * div_term)
    return encoding