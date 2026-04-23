#!/usr/bin/env python
import sys
import logging
from logger_setup import logger
from param_parser import param_info, validate

class JectoSceneObject():
    config_path = ''
    @classmethod
    def config(cls, config: dict, config_path: str, init=True, **kwargs):
        cls.config_path = config_path
        init_args, init_kwargs = cls._setup(config, **kwargs)
        if init:
            return cls(*init_args, **init_kwargs)
        else:
            return init_args, init_kwargs
    
    @classmethod
    def validate_config(cls, *args, **kwargs) -> dict:
        config, success, msg = validate(*args,**kwargs)
        if not success:
            cls.config_fail(msg)
        if success and msg:
            cls.get_logger().warning(msg)
        return config


    # @classmethod
    # def default_value(cls, name: str, context: str):
    #     return param_info(name, context).get('default')
    
    @classmethod
    def config_fail(cls, msg: str):
        cls.get_logger().error('configuration failed in file: {} \n\n{}'.format(cls.config_path, msg))
        sys.exit(1)

    @classmethod
    def get_logger(cls):
        return logging.LoggerAdapter(logger, extra={"classname": cls.__name__}) 
    
