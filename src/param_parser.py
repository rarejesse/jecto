#!/usr/bin/env python
import numpy as np
import yaml
from geometry import as_scipy_rotation
from os.path import dirname, abspath, split
from subprocess import check_output
import datetime
import os
import sys


def validate(config: dict, name: str, context: str) -> tuple:
    """
    Check if the params/sub-params provided in the yaml file are valid in their context.

    params:
        name: string, the name of the class or object that the params are associated with
        context: string, the context in which the params are being used
        config: dictionary whose params (the sub-params) will be checked according the entry in 'param_spec.yaml' corresponding to 'name'.
    
    The following sets of params are defined in 'param_spec.yaml' for each name (class):
        
        required:            all of these params must be included
        required_exclusive:  exactly one of these params must be included (unless it contains any groups of codependent params)
        optional:            any of these params may optionally be included
        codependent:         if any of these params are included, all of them must be included

    Returns:
        config (dict): the validated configuration
        success (bool): configuration is valid
        msg (str): either empty, or contains error or warning message about configuration
    
    """

    # need to figure out a good way to support nested yaml files, possibly only one level of nesting would be allowed...
    # nested_file = None
    # if 'yaml' in config:
    #     nested_file = config['yaml']
    #     config = yaml.load(config['yaml'], Loader=yaml.FullLoader)
    #     # enforce that the nested config contains only the expected high level key (the 'name' that was passed to this function)
    #     if not list(config.keys()) == [name]:
    #         return False, 'invalid nested configuration file: {}'.format(nested_file), nested_file
    #     nested_success, nested_msg, nested_nested_file = validate(config[name], name, context)
    # if not nested_success:
    #     return False, nested_msg, nested_file

    if config is None:
        msg = 'empty configuration for \'{}\' in context \'{}\', expected subparameters'.format(name, context)
        return config, False, msg
    if not isinstance(config, dict):
        msg = 'invalid configuration for \'{}\' in context \'{}\': expected subparameters, got {} value: {}'.format(name, context, type(config).__name__, config)
        return config, False, msg
    
    provided = set(config.keys())
    all_specs = yaml.load(open('/home/jesse/ros2_ws/src/vinlab/config/param_spec.yaml'),Loader=yaml.FullLoader)
    
    try:
        spec_contexts = all_specs[name]
    except (KeyError):
        msg = '\'{}\' is not a valid key in any context)'.format(name)
        return config, False, msg
    try: 
        spec = next(i for i in spec_contexts if i['context'] == context) 
        # print('')
        # print('using spec for \'{}\' in context \'{}\': {}'.format(name, context, spec))
    except (StopIteration):
        msg = '\'{}\' not listed as valid context for \'{}\'. The listed contexts are: {}'.format(context, name, ', '.join("'{}'".format(i['context']) for i in spec_contexts))
        return config, False, msg
    

    #compare the set of subparams in the spec for this context to the keys of 'config' provided
    subparams = spec['subparams'] #TODO if this is empty: then check if the values meet the type and conditions given for it in the config file    
    req = set(subparams['required'])
    req_excl = set(subparams['required_exclusive'])
    opt = set(subparams['optional'])
    # opt_excl = set(subparams['optional_exclusive']) 
    codeps = [set(i) for i in subparams['codependent']]
    allowed = req | req_excl | opt
    allowed_valued = set([p for p in allowed if not p.endswith('*')]) # any param name with a '*' at the end indicates it has subparams

    # print('validating config for \'{}\' in context \'{}\'...'.format(name, context))
    # print('req: {}, req_excl: {}, opt: {}, codeps: {}'.format(req, req_excl, opt, codeps))
    # print('allowed: {}'.format(allowed))
    # print('allowed_valued: {}'.format(allowed_valued))
    # print('provided: {}'.format(provided))
    
    for s in (req, req_excl, opt, allowed):
        for v in s.copy():
            s.remove(v)
            s.add(v.rstrip('*'))
    for s in codeps.copy():
        for v in s.copy():
            s.remove(v)
            s.add(v.rstrip('*'))
    # print ('after stripping *')
    # print('req: {}, req_excl: {}, opt: {}, codeps: {}'.format(req, req_excl, opt, codeps))
                
    
    codeps_req = [] #handle situation when codependent params are in 'required_exclusive', meaning they are required as a group, if any are included 
    for codep in codeps:
        if len(codep&provided) > 0 and not codep <= provided: #then you provided some but not all of the codependent params
            msg = 'param(s): {} also requires: {} to be specified in configuration for \'{}\''.format(
                  ', '.join("'{0}'".format(i) for i in list(codep&provided)),', '.join("'{0}'".format(i) for i in list(codep-provided)), name)
            return config, False, msg
        codeps_req = codeps_req + [codep&req_excl] if codep <= req_excl else codeps #add the codep set to the list if it's a subset of 'required_exclusive'
      
    if not provided <= allowed: #then you provided a param that is not allowed
        msg = 'invalid param(s): {} in configuration for \'{}\', the allowed params here are: {}'.format(
              ', '.join("'{0}'".format(i) for i in list(provided-allowed)),name, ', '.join("'{0}'".format(i) for i in list(allowed)))
        return config, False, msg
    
    if not req <= provided: #then you did not provide all of the required params
        msg = 'required param(s): {} not found in configuration for \'{}\''.format(
              ', '.join("'{}'".format(i) for i in list(req-(provided&req))),name)
        return config, False, msg
    
    if req_excl and len(provided&req_excl) > 1: #then you provided more than one param from 'required_exclusive'
        if provided&req_excl not in codeps_req: #then you did not provide exactly one codependent group from 'required_exclusive'
            msg = 'invalid combination of params: {} in configuration for \'{}\''.format(
                  ', '.join("'{}'".format(i) for i in list(provided&req_excl)),name)
            return config, False, msg
    
    if req_excl and len(provided&req_excl) < 1: #then you provided none of 'required_exclusive'
            msg = 'one of the following params must be included in configuration for \'{}\': {}'.format(
                  name, ', '.join("'{}'".format(i) for i in list(req_excl)))
            return config, False, msg
   
    for param in provided & allowed_valued:
        param_type = type(config[param]).__name__
        if config[param] is None:  # then you provided a value for this param, but it's empty
            msg = 'empty value for \'{}\' in configuration for \'{}\''.format(param, name)
            return config, False, msg
        try:
            value_specs = next(i['values'] for i in all_specs[param] if i['context'] == name)
        except (StopIteration, KeyError):
            print('no value conditions specified for param \'{}\' in context \'{}\' in param_spec.yaml'.format(param, context))
            sys.exit()

        value_types = [i['type'] for i in value_specs] 
        if param_type not in value_types:  # then the type of the value you provided is not one of the allowed types for this param
            expected = ' or '.join('{}'.format(i) for i in value_types)
            msg = 'invalid type for param \'{}\' in context \'{}\': expected {}, got {} value: {}'.format(param, name, expected, param_type, config[param])
            return config, False, msg
        
        value_conditions = next(i.get('conditions') for i in value_specs if i['type'] == param_type)
        if value_conditions:
            for condition, err_msg in value_conditions:
                value = config[param]
                success = eval(condition) 
                if not success:
                    msg = 'param value: {} for param \'{}\' in context \'{}\' is invalid: {}'.format(value, param, name, err_msg)
                    return config, False, msg

    for param in allowed_valued - provided: #optional params that were not supplied: assign default value if it exists
        try:
            default = next(i.get('default') for i in all_specs[param] if i['context'] == name)
        except (StopIteration, KeyError):
            print('no default value specified for param \'{}\' in context \'{}\' in param_spec.yaml'.format(param, context))
            continue
        if default is not None:
            config[param] = default

    success, msg = True, ''
    return config, success, msg


   # TODO: handle optional-exclusive keys
    # e.g. only 1 sensor can be on each frame (imu, camera, position sensor are optional-exclusive)
    # probably don't want this though... it can be useful to simulate two sensors similtaneously attached to the same frame (two imus with diffrent noise parameters, etc)
    # similar for feature point set types (only 1 can be associated with each id)
    # if opt_excl and len(keys&opt_excl) != 1: #then you provided more than one key from 'required_exclusive'
    #     if keys&opt_excl not in codeps_req: #then you did not provide a exactly one codependent group from 'required_exclusive'
    #         raise ConfigurationError('invalid combination of keys: {} in configuration for \'{}\''
    #                                  .format(', '.join("'{0}'".format(i) for i in list(keys&opt_excl)),name))
        
    # config_mode = keys&req_excl
    # config_mode = keys&req_excl$opt_excl

    #iterate through the provided keys and check if any of them have value specifications
    #if so, then check if the value is of the correct type and meets the conditions specified in the config file
    # for k in keys:
    #     try:
    #         subkey_spec = next(i for i in all_specs[k] if i['context'] == name and 'value' in i) 
    #         print('this should be a simple subkey spec: {}-->{}'.format(k,subkey_spec['description']))
    #         print('the value is: {}'.format(config[k]))
    #         print('the type is {}, should be {}'.format(type(config[k]).__name__, subkey_spec['value']['type']))
    #         print('')
    #     except (StopIteration):
    #         pass
    #     pass



def param_info(name: str, context: str):
    all_specs = yaml.load(open('/home/jesse/ros2_ws/src/vinlab/config/param_spec.yaml'),Loader=yaml.FullLoader) #list of lists of dictionaries
    param_spec = all_specs[name]
    param = next(i for i in param_spec if i['context'] == context) 
    return param
    



# def set_output(con    fig,config_file=None):
#     """
#     make output directories and generate a string representation according to the output 
#     configuration in the scene config file.

#     should just generate the keys from the yaml dict itself, with indentations becoming underscores

#     """
#     check_keys(config, 'output', 'scene')

#     vinlab_base = dirname(dirname(abspath(__file__))) #works as long as this file remains two directories deep from vinlab base
#     vinlab_commit = check_output(['git', 'rev-parse', '--short', 'HEAD'],cwd=vinlab_base).decode('ascii').strip()
#     config_name = split(config_file)[1].replace('.yaml','') #file name without the extension
#     now = datetime.datetime.now().strftime("%Y.%m.%d.%H%M%S")
#     output_path = vinlab_base+'/output/'+'{}.{}'.format(now,config_name)

#     flat_config = flatten(config)
#     flat_config['output_path'] = output_path
#     flat_config['vinlab_commit'] = vinlab_commit
#     output_str =''
#     for k,v in flat_config.items():
#         if isinstance(v,bool):
#             v = int(v)
#         output_str += '{} {} '.format(k, v)
#     print(output_str)
#     return output_str


# class ConfigurationError(Exception):
#     """
#     Raise error with descriptive messsage when loading a wrongly configured Scene yaml file. 
#     Any number of arguments can be passed to msg_lookup as needed to format the error message. 
#     Each error message is identified by an all-caps string error code.
#     """
#     @classmethod
#     def get_msg(cls, msg_code: str, *args):
#         args = (*args, [], [], [], [])  #temporary hack
#         msg = msg_lookup.get(msg_code, 'invalid yaml configuration')
#         # return cls(msg)
#         return msg

#     def __init__(self, msg: str):
#         color_code = 31 # ANSI color red=31, yellow=33, green=32, blue=34, magenta=35, cyan=36
#         super().__init__('\033[{}m{}\033[0m'.format(color_code, msg))

        
