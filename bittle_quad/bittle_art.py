# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import numpy as np
from isaacsim import SimulationApp
import sys
# URDF import along with configuration and simulation sample
kit = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

# importing some more packages for the simulation and control
import omni.kit.commands
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
from omni.isaac.dynamic_control import _dynamic_control

#Setting up the import configurations
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.distance_scale = 1

#Getting the path to the extension data:
extension_path = get_extension_path_from_name("omni.importer.urdf")

#Import the URDF 
status, stage_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path = extension_path + "/data/urdf/robots/bittle/urdf/bittle.urdf",
    import_config = import_config,
    get_articulation_root = True
)

#Get the Stage Handle
stage = omni.usd.get_context().get_stage()

#Enable Physics
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
#set gravity
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
#set the solver settings
PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")

# Add ground plane
omni.kit.commands.execute(
    "AddGroundPlaneCommand",
    stage=stage,
    planePath="/groundPlane",
    axis="Z",
    size=1500.0,
    position=Gf.Vec3f(0, 0, -0.58),
    color=Gf.Vec3f(0.5),
)

# Add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(500)

#Getting to some Maths and coding ":)"
left_posvec = np.load("left_posvec.npy") 
right_posvec = np.load("right_posvec.npy")


LEFT_JOINTS = [ "/bittle/base_frame_link/left_back_shoulder_joint",
                "/bittle/base_frame_link/left_front_shoulder_joint",
                "/bittle/left_back_shoulder_link/left_back_knee_joint",
                "/bittle/left_front_shoulder_link/left_front_knee_joint",
                 ]

RIGHT_JOINTS = [ "/bittle/base_frame_link/right_back_shoulder_joint",
                 "/bittle/base_frame_link/right_front_shoulder_joint",
                 "/bittle/right_back_shoulder_link/right_back_knee_joint",
                 "/bittle/right_front_shoulder_link/right_front_knee_joint",
                  ]
stage = omni.usd.get_context().get_stage()

# Get the Joints

for joint_path, x in zip(LEFT_JOINTS, left_posvec):
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise Exception(f"Joint {joint_path} not found !!!")
    else:
        print(" {joint} Set to go !!")    

    # Get the handle to control the robto
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    print(float(x))    
    pos = float(x)
    drive.GetDampingAttr().Set(1500)
    drive.GetStiffnessAttr().Set(2000)
    drive.GetTargetPositionAttr().Set(pos)

for joint_path, y in zip(RIGHT_JOINTS, right_posvec):
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise Exception(f"Joint {joint_path} not found !!!")
    else:
        print(" {joint} Set to go !!")    

    # Get the handle to control the robto
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    print(float(y))    
    pos = float(y)
    drive.GetDampingAttr().Set(1500)
    drive.GetStiffnessAttr().Set(2000)
    drive.GetTargetPositionAttr().Set(pos)

# for joint_path in RIGHT_JOINTS:
#     joint_prim = stage.GetPrimAtPath(joint_path)
#     if not joint_prim.IsValid():
#         raise Exception(f"Joint {joint_path} not found !!!")
#     else:
#         print(" {joint} Set to go !!")    

#     # Get the handle to control the robto
#     drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        
#     drive.GetDampingAttr().Set(700)
#     drive.GetStiffnessAttr().Set(1000)
#     drive.GetTargetPositionAttr().Set(-25)

# Start simulation
omni.timeline.get_timeline_interface().play()
# perform one simulation step so physics is loaded and dynamic control works.
kit.update()

art = Articulation(prim_path=stage_path)
art.initialize()

if not art.handles_initialized:
    print(f"{stage_path} is not an articulation")
else:
    print(f"Got articulation {stage_path}")

# art.set_fixed_base(False)

# perform simulation
for frame in range(100000):
    kit.update()


# Shutdown and exit
omni.timeline.get_timeline_interface().stop()
kit.close()
