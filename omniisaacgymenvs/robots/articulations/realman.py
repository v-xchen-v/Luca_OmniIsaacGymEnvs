# Ref: franka.py
from typing import Optional
import torch
import math
import numpy as np

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema

class Realman(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "realman",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name
        
        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation
        
        if self._usd_path is None:
            # self._usd_path = "/home/yichao/Documents/rm_75_6f_description/rm_75_6f_description/urdf/rm_75_6f_description/rm_75_6f_description_fixed.usd"
            self._usd_path = "omniverse://localhost/Projects/Luca_Data/Luca_Data/Robots/Realman/rm_75_6f_description.usd"
        
        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None
        )
        
        # dof_paths = [
        #     "rm_75_6f_description/base_link/joint1",
        #     "rm_75_6f_description/Link1/joint2",
        #     "rm_75_6f_description/Link2/joint3",
        #     "rm_75_6f_description/Link3/joint4",
        #     "rm_75_6f_description/Link4/joint5",
        #     "rm_75_6f_description/Link5/joint6",
        #     "rm_75_6f_description/Link6/joint7"
        # ]
        
        dof_paths = [
            "base_link/joint1",
            "Link1/joint2",
            "Link2/joint3",
            "Link3/joint4",
            "Link4/joint5",
            "Link5/joint6",
            "Link6/joint7"
        ]
        
        drive_type = ["angular"] * 7
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]]
        stiffness = [400 * np.pi / 180] * 7
        damping = [80 * np.pi / 180] * 7
        max_force = [87, 87, 87, 87, 12, 12, 12]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]]
        
        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )
            
            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )

    def set_realman_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)