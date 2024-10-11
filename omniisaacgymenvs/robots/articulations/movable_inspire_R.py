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
            # self._usd_path = "/home/wenbo/R_inspire_constrained_all_2.usd"
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