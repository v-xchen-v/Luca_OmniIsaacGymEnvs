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


from typing import Optional

import carb
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class MovableInspireHandR(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "movable_inspire_R",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        self._usd_path = usd_path
        self._name = name
        if self._usd_path is None:
            # self._usd_path = "omniverse://localhost/Projects/Luca_Data/Robots/InspireHand/L_inspire_mimic_noflange.usd"
            # self._usd_path = "omniverse://localhost/Projects/Luca_Data/Luca_Data/Robots/InspireHand/L_inspire_mimic.usd"
            # replace it to your own path
            # self._usd_path = "/home/wenbo/Documents/repos/Luca_Data/Robots/InspireHand/R_inspire_mimic_noflange_movable_tested.usd"
            # self._usd_path = "/home/wenbo/R_inspire_mimic_noflange_movable_tested.usd"
            # self._usd_path = "/home/wenbo/Documents/repos/Robots/InspireHand/R_inspire_mimic_noflange_movable_tested.usd"
            # self._usd_path = "/home/wenbo/R_inspire_mimic_noflange_movable_tested.usd"
            # self._usd_path = "/home/wenbo/R_inspire_constrained.usd"
            # self._usd_path = "/home/wenbo/R_inspire_1009_v3_maxforce10000_fixed.usd"
            self._usd_path = "/home/wenbo/R_inspire_1011_v2_filterpair_thumb_plam_proximal.usd"
            # self._usd_path = "/home/wenbo/R_inspire_full_drive_new.usd" # full drive hand
            # self._usd_path = "/home/wenbo/R_inspire_sh_property_04.usd"

        self._position = torch.tensor([0.0, 0.0, 0.5]) if translation is None else translation
        self._orientation = (
            torch.tensor([0.257551, 0.283045, 0.683330, -0.621782]) if orientation is None else orientation
        )

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        
        
        
    def set_inspire_properties(self, stage, shadow_hand_prim):
        for link_prim in shadow_hand_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)
   
    def set_motor_control_mode(self, stage, shadow_hand_path):
        base_stiffness = 100000
        base_damping = 0.01
        base_maxforce = 200
        finger_stiffness = 10
        finger_damping = 0.01
        finger_stiffness = 100
        finger_maxforce = 10
        joints_config = {

            "move_x": {"stiffness": base_stiffness, "damping": base_damping, "max_force": base_maxforce, 'drive_type':'linear', 'joint_path': 'R_movable_root_link/move_x'},
            "move_y": {"stiffness": base_stiffness, "damping": base_damping, "max_force": base_maxforce, 'drive_type':'linear', 'joint_path': 'R_movable_basex/move_y'},
            "move_z": {"stiffness": base_stiffness, "damping": base_damping, "max_force": base_maxforce, 'drive_type':'linear', 'joint_path': 'R_movable_basey/move_z'},
            "rot_r": {"stiffness": base_stiffness, "damping": base_damping, "max_force": base_maxforce, 'drive_type':'Angular', 'joint_path': 'R_movable_basez/rot_r'},
            "rot_p": {"stiffness": base_stiffness, "damping": base_damping, "max_force": base_maxforce, 'drive_type':'Angular', 'joint_path': 'R_movable_rot0/rot_p'},
            "rot_y": {"stiffness": base_stiffness, "damping": base_damping, "max_force": base_maxforce, 'drive_type':'Angular', 'joint_path': 'R_movable_rot1/rot_y'},
            "R_index_proximal_joint": {"stiffness": finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_hand_base_link/R_index_proximal_joint'},
            "R_index_intermediate_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_index_proximal/R_index_intermediate_joint'},
            "R_middle_proximal_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_hand_base_link/R_middle_proximal_joint'},
            "R_middle_intermediate_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_middle_proximal/R_middle_intermediate_joint'},
            "R_pinky_proximal_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_hand_base_link/R_pinky_proximal_joint'},
            "R_pinky_intermediate_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_pinky_proximal/R_pinky_intermediate_joint'},
            "R_ring_proximal_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_hand_base_link/R_ring_proximal_joint'},
            "R_ring_intermediate_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_ring_proximal/R_ring_intermediate_joint'},
            "R_thumb_proximal_yaw_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_hand_base_link/R_thumb_proximal_yaw_joint'},
            "R_thumb_proximal_pitch_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_thumb_proximal_base/R_thumb_proximal_pitch_joint'},
            "R_thumb_intermediate_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_thumb_proximal/R_thumb_intermediate_joint'},
            "R_thumb_distal_joint": {"stiffness":  finger_stiffness, "damping": finger_damping, "max_force": finger_maxforce, 'drive_type':'Angular', 'joint_path': 'R_thumb_intermediate/R_thumb_distal_joint'},

        }

        for joint_name, config in joints_config.items():
            set_drive(
                f"{self.prim_path}/{config['joint_path']}",
                config['drive_type'],
                "position",
                0.0,
                config["stiffness"] * np.pi / 180,
                config["damping"] * np.pi / 180,
                config["max_force"],
            )
