from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

class RealmanView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "RealmanView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)
        
        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/realman/rm_75_6f_description/Link7", name="hands_view", reset_xform_properties=False
        )
        
        # self.
        
    
    def initialize(self, physics_sim_view) -> None:
        return super().initialize(physics_sim_view)
        