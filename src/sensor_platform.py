#!/usr/bin/env python 
from reference_frame import ReferenceFrame
from rigid_body import RigidBody
from error_messages import ErrorMessage as err


class SensorPlatform(RigidBody): 
    def __init__(self, platform_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = platform_id
        # self.frames = frames
    
    @classmethod
    def _setup(cls, config: dict) -> tuple:
        cls.get_logger().debug('configuring sensor platform...')
        config = cls.validate_config(config, 'platform', context='scene')
        platform_id = config['id']
        if 'frames' in config:
            frames_config = config['frames']
        else:
            cls.get_logger().warning('no frames defined for sensor platform \'{}\', using single reference frame \'body\''.format(platform_id))
            frames_config = [{'id': 'body'}]
        
        rigid_body_args, rigid_body_kwargs = RigidBody.config(frames_config, cls.config_path, init=False)
        return (platform_id, *rigid_body_args), rigid_body_kwargs

    @classmethod
    def default(cls, init=True):
        args, kwargs = cls._setup(config={})
        if init:
            return cls(*args, **kwargs)
        return args, kwargs