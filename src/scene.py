#!/usr/bin/env python
import numpy as np
import yaml
import geometry
from rigid_body import RigidBody
from sensor_platform import SensorPlatform
from environment import Environment
from trajectory import Trajectory, TrajectoryEvaluation, ReferenceTrajectory, MatchedTrajectory
from alignment import Alignment, AlignmentSettings
from time_manager import TimeManager
import logging
from error_messages import ErrorMessage as err
from jecto_scene_object import JectoSceneObject
import geometry 

class Scene(JectoSceneObject):
    def __init__(self, platform: SensorPlatform, environment: Environment, trajectories: list):
        
        self.platform = platform
        self.env = environment
        self._trajectories = trajectories
           
    

    @classmethod
    def _setup(cls, config: dict) -> tuple:
        config = cls.validate_config(config, 'scene', context=None)
        cls.get_logger().info('configuring sensor platform...')
        if 'platform' in config:
            platform = SensorPlatform.config(config['platform'], config_path)
        else:
            cls.get_logger().info('no platform defined in scene configuration, initializing default platform')
            platform = SensorPlatform.default()
        cls.get_logger().info('succesfully configured sensor platform: \'{}\', configuring environment...'.format(platform.id))

        if 'environment' in config:
            env = Environment.config(config['environment'], config_path)
        else:
            cls.get_logger().info('no environment defined in scene configuration, initializing default environment')
            env = Environment.default()
        cls.get_logger().info('succesfully configured environment: \'{}\''.format(env.id))

        # if 'inverse_projection' in env.feature_types #TODO

        if 'trajectories' in config:
            trajectories = []
            n = len(config['trajectories'])
            cls.get_logger().info('configuring {}'.format(['1 trajectory...',str(n)+' trajectories...'][bool(n>1)]))
            ref_traj_config = config['trajectories'][0]
    
            cls.get_logger().debug('configuring reference trajectory \'{}\'...'.format(ref_traj_config['id']))
            if 'align' in ref_traj_config:
                cls.config_fail(err.traj_self_align(ref_traj_config['id']))
           
            ref_traj = ReferenceTrajectory.config(ref_traj_config, config_path, platform=platform, environment=env)
            trajectories.append(ref_traj)
            for i in range(1,n):
                traj_config = config['trajectories'][i]
                traj = Trajectory.config(traj_config, config_path, platform=platform, environment=env)
                if 'align' in traj_config:
                    alignment = Alignment.config(traj_config['align'], config_path, ref_traj=ref_traj, traj=traj)
                else:
                    alignment = Alignment.default()

                traj = MatchedTrajectory.config(traj_config, config_path, reference_trajectory=ref_traj, alignment=alignment)
            
                # if traj.frame_id not in env.frame_ids:
                #     cls.config_fail(err.traj_frame_not_found(traj.id, traj.frame_id, env.id, env.frame_ids))
                # if traj.body_frame_id not in platform.frame_ids:
                #     cls.config_fail(err.traj_body_frame_not_found(traj.id, traj.body_frame_id, platform.id, platform.frame_ids)) 
                alignment = cls.align_trajectory(ref_traj, traj, platform, env, settings)
                env.add_frame(frame_from=ref_traj.frame_id, frame_to=alignment.frame_id, transform=alignment.transform) #add the aligned frame to the environment
                traj_eval = TrajectoryEvaluation(ref_traj, traj, platform, env, alignment)
                trajectories.append(traj_eval)
        
            cls.get_logger().info('successfully configured {}'.format(['reference trajectory','all trajectories'][bool(n>1)]))
        else:
            cls.get_logger().info('no trajectories defined in Scene configuration, initializing empty trajectory evaluations list')
            trajectories = []

        return (platform, env, trajectories), {}
    
    @classmethod 
    def construct(cls, config_path: str, init: bool = True):
        try:
            cls.get_logger().info('loading scene configuration from file: {}'.format(config_path))
            config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        except Exception as e:
            cls.config_fail('error loading scene configuration file \'{}\': {}'.format(config_path, str(e)))
        return cls.config(config, config_path, init=init)
        
    
    
    @classmethod
    def align_trajectory(cls, reference_trajectory: Trajectory, trajectory: Trajectory, platform: RigidBody, environment: RigidBody, settings: AlignmentSettings) -> Alignment:

        ref_pos, ref_rot = reference_trajectory.positions, reference_trajectory.rotations
        pos,rot = trajectory.positions, trajectory.rotations

        if trajectory.body_frame_id != reference_trajectory.body_frame_id:
            transform = platform.relative(trajectory.body_frame_id, reference_trajectory.body_frame_id)
            pos, rot = geometry.transfer(positions = pos,
                                         rotations = rot,
                                         pos_transform = transform.pos,
                                         rot_transform = transform.rot)

        if trajectory.frame_id != reference_trajectory.frame_id:
            transform = environment.relative(trajectory.frame_id, reference_trajectory.frame_id)
            pos, rot = geometry.reframe(positions = pos,
                                        rotations = rot,
                                        pos_transform = transform.pos,
                                        rot_transform = transform.rot)
                                                
        if settings.temporal.mode == 'manual':
            time_offset = settings.temporal.constant
        else:
            raise NotImplementedError('Temporal alignment mode \'{}\' not implemented'.format(settings.temporal.mode))

        #initialize time manager to handle time alignment and selection
        time = TimeManager(reference_trajectory.times, trajectory.times, tol=1e-3, offset=time_offset)


        #select the poses to be used for alignment
        if settings.selection_mode == 'time': 
            ref_start = time.nearest_index(settings.start, reference=True)
            ref_end = time.nearest_index(settings.end, reference=True)
            start = time.nearest_index(settings.start, reference=False)
            end = time.nearest_index(settings.end, reference=False)
        else: #index mode
            ref_start, ref_end = settings.start, settings.end
            start, end = settings.start, settings.end
        
        #mask out the unselected values, in addition to the time matching mask
        time.ref_mask[0:ref_start] = False; time.ref_mask[ref_end:] = False 
        time.mask[0:start] = False; tm.mask[end:] = False 
        #TODO: mask out more values to reflect subsampling at a certain rate if specified in settings

        mask = tm.mask[start:end]
        ref_pos_selection = ref_pos[ref_mask]
        ref_rot_selection = ref_rot[ref_mask]
        pos_selection = pos[mask]
        rot_selection = rot[mask]


        allow_rescale = settings.spatial.allow_rescale
        centering_mode = settings.spatial.centering_mode
        yaw_only = settings.spatial.yaw_only


        rot_align, pos_align = geometry.kabsh_umeyama(ref_pos_selection, pos_selection, allow_rescale, centering_mode, yaw_only)
        alignment_transform = RigidTransform(rot_align, pos_align)
        aligned_frame_id = trajectory.id + '_aligned'

        alignment = Alignment(reference_trajectory.id, aligned_frame_id, alignment_transform, time_offset, settings)
        return alignment
    


    def _get_trajectory(self, traj_id) -> Trajectory:
        if isinstance(traj_id, int):
            traj_index = traj_id
            try:
                self._trajectories[traj_index]
            except IndexError:
                n = len(self._trajectories)
                self.get_logger().error('Trajectory index {} out of range for Scene with {}'.format(
                    traj_index, [str(n)+ ' trajectories', '1 trajectory' ][bool(len(self._trajectories) == 1)]))
            
        elif isinstance(traj_id, str):
            try:
                traj = next(i for i in self._trajectories if i.id == traj_id) 
                traj_index = self._trajectories.index(traj)   
                self.get_logger().debug('Trajectory id \'{}\' corresponds to trajectory index {}'.format(traj_id, traj_index))
            except StopIteration:
                self.get_logger().error('Trajectory id \'{}\' not found in Scene. Defined trajectories are: {}'.format(
                    traj_id, ', '.join([v.id for v in self._trajectories])))
        else:
            self.get_logger().error('Trajectory identifier must be of type int or str, got type {}'.format(type(traj_id)))
        return self._trajectories[traj_index]
    

    def traj(self, traj_id) -> Trajectory: 
        return self._get_trajectory(traj_id)
    
    

if __name__ == "__main__":
    logging.basicConfig(format='%(classname)s: %(message)s',level=logging.DEBUG)
    # logging.basicConfig(format='%(message)s',level=logging.INFO)
    config_path = '/home/jesse/ros2_ws/src/vinlab/config/test_scene_02.yaml'
    scene = Scene.construct(config_path)