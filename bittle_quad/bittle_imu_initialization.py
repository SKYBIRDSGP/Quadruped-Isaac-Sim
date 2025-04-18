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
from omni.isaac.sensor import IMUSensor

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

stage = omni.usd.get_context().get_stage()
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


# Declaring and initializing the IMU sensor
imu_sensor = IMUSensor(prim_path="/bittle/mainboard_link/imu_joint")

imu_sensor.initialize()

if imu_sensor:
    print("IMU initialization is Successful !!!")

else:
    print("IMU initialization not done!!!")

# perform simulation
for frame in range(100000):
    kit.update()
    # Retrieving the IMU data 
    print(imu_sensor.get_current_frame())

# Shutdown and exit
omni.timeline.get_timeline_interface().stop()
kit.close()
