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

import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class MovableInspireRView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "MovableInspireRView",
    ) -> None:

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)
        self._actuated_dof_indices = list()
        self.fingers = RigidPrimView(
            # prim_paths_expr='/World/envs/.*/movable_inspire_R/R_inspire_mimic_noflange_movable_totest_2/R_inspire/R*',
            prim_paths_expr='/World/envs/.*/movable_inspire_R/R_inspire_mimic_noflange_movable_tested/R_inspire_mimic_noflange_movable_totest_2/R_inspire/R*',
            name="finger_view",
            reset_xform_properties=False,
        )

    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        self._actuated_dof_indices = [i for i in range(self.num_dof)]

        self.actuated_joint_names = [
            "move_x",
            "move_y",
            "move_z",
            "rot_r",
            "rot_p",
            "rot_y",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
        ]
        self._actuated_dof_indices = list()
        for joint_name in self.actuated_joint_names:
            self._actuated_dof_indices.append(self.get_dof_index(joint_name))
        self._actuated_dof_indices.sort()