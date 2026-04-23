#!/usr/bin/env python 
import numpy as np
from geometry import as_scipy_rotation
from reference_frame import ReferenceFrame
from rigid_body import RigidBody
from feature_set import FeatureSet, EmptyFeatureSet
from param_parser import param_info
from error_messages import ErrorMessage as err

class Environment(RigidBody): 
    def __init__(self, environment_id: str, gravity_frame_id: str, gravity: np.ndarray, features: FeatureSet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = environment_id
        self._gravity = gravity
        self.gravity_frame_id = gravity_frame_id
        self.features = features
        self.has_features = features is not None
        self.feature_types = {}

    @classmethod
    def _setup(cls, config: dict) -> tuple:
        cls.get_logger().debug('configuring environment...')
        config = cls.validate_config(config, 'environment', context='scene')
        env_id = config['id']
        if 'frames' in config:
            frames_config = config['frames']
        else:
            cls.get_logger().warning('no frames defined for environment \'{}\', using single reference frame \'global\''.format(env_id))
            frames_config = [{'id': 'global'}]
        baseframe_id = frames_config[0]['id']

        rigid_body_args, rigid_body_kwargs = RigidBody.config(frames_config, cls.config_path, init=False)
       
        if not 'gravity' in config:
            config['gravity'] = {'enable': False, 'frame': baseframe_id}
        gravity_config = cls.validate_config(config['gravity'], 'gravity', context='environment')

        if 'frame' in gravity_config:
            gravity_frame_id = gravity_config['frame']
            if not any(f['id'] == gravity_frame_id for f in frames_config):
                cls.config_fail(err.gravity_frame_not_found(gravity_frame_id, env_id, [f['id'] for f in frames_config]))
        else:
            cls.get_logger().debug('no gravity frame defined for environment \'{}\', using environment baseframe'.format(env_id))
            gravity_frame_id = baseframe_id
        
        gravity = float(gravity_config['multiplier']) * np.array(gravity_config['vector'])
        cls.get_logger().debug('gravity frame: \'{}\', gravity vector: [{:.3f},{:.3f},{:.3f}]'.format(gravity_frame_id, *gravity))

        gravity_enable = gravity_config['enable']
        cls.get_logger().debug('gravity is {} for environment \'{}\'.'.format(['disabled','enabled'][gravity_enable], env_id))

                                              
        features = FeatureSet.config(config['features']) if 'features' in config else EmptyFeatureSet()
        return (env_id, gravity_frame_id, gravity, features, *rigid_body_args), {}
    
    @classmethod
    def default(cls, init=True):
        args, kwargs = cls._setup(config={})
        if init:
            return cls(*args, **kwargs)
        return args, kwargs
    
    