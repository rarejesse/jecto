#!/usr/bin/env python

from utils import nicelist

class ErrorMessage():
    @staticmethod
    def baseframe_invalid(baseframe_config) -> str:
        return 'invalid baseframe config for rigid body: the first entry in \'frames\' must contain only an \'id\' with a string value, got config: {}'.format(baseframe_config)

    @staticmethod
    def duplicate_frame_id(duplicate_ids: list) -> str:
        return 'duplicate definitions found for the following frame(s): {}'.format(nicelist(duplicate_ids))

    @staticmethod
    def expected_list(key: str, value) -> str:
        return 'expected a list in context \'{}\', got {} value: {}'.format(key, type(value).__name__, value)

    @staticmethod
    def floating_frame_group(group: list, baseframe_id: str) -> str:
        return 'invalid transform tree: frame group {} forms a cycle, no connection to the base frame \'{}\'.'.format(group, baseframe_id)

    @staticmethod
    def floating_frames(baseframe_id: str) -> str:
        return 'invalid transform tree: at least one frame must have a transform with respect to baseframe \'{}\''.format(baseframe_id)

    @staticmethod
    def gravity_frame_not_found(frame_id: str, env_id: str, defined_frames: list) -> str:
        return 'gravity frame id \'{}\' not found in environment \'{}\', the defined frames are: {}'.format(frame_id, env_id, nicelist(defined_frames))
    
    @staticmethod
    def rigid_body_frame_exists(frame_id: str) -> str:
        return 'frame \'{}\' already exists in rigid body'.format(frame_id)

    @staticmethod
    def rigid_body_index_error() -> str:
        return 'rigid body must have at least one frame defined'

    @staticmethod
    def rigid_body_no_frame(frame_id: str) -> str:
        return 'frame {} not found in rigid body'.format(frame_id)

    @staticmethod
    def transform_frame_not_found(frame_id: str, defined_frames: list) -> str:
        return 'transform frame id \'{}\' not found, the defined frames are: {}'.format(frame_id, nicelist(defined_frames))
    
    @staticmethod
    def transform_from_self(frame_id: str) -> str:
        return 'frame \'{}\' transform defined with respect to itself'.format(frame_id)

    @staticmethod
    def traj_bad_data(traj_id: str, filename: str, delimiter: str, exception_msg: str) -> str:
        return 'loading the data for trajectory \'{}\' from file: {} using delimeter: \'{}\' failed with Exception: {}'.format(traj_id, filename, delimiter, exception_msg)

    @staticmethod
    def traj_body_frame_not_found(trajectory_id: str, body_frame_id: str, platform_id: str, body_frames: list) -> str:
        return 'body frame id \'{}\' for trajectory \'{}\' not found on platform \'{}\', the defined body frames are: {}'.format(body_frame_id, trajectory_id, platform_id, nicelist(body_frames))
    
    @staticmethod
    def traj_frame_not_found(frame_id: str, traj_id: str, env_id: str, defined_frames: list) -> str:
        return 'frame id \'{}\' for trajectory \'{}\' not found in environment \'{}\', the defined environment frames are: {}'.format(frame_id, traj_id, env_id, nicelist(defined_frames))
    
    @staticmethod
    def traj_invalid_component(traj_id: str, invalid_chars) -> str:
        return 'trajectory \'{}\' format string has invalid component(s): \'{}\''.format(traj_id, ', '.join(invalid_chars))

    @staticmethod
    def traj_invalid_quat(traj_id: str) -> str:
        return 'trajectory \'{}\' invalid or incomplete quaternions. Format string must include \'qx\', \'qy\', \'qz\', and \'qw\' exactly once.'.format(traj_id)

    @staticmethod
    def traj_invalid_unit(unit: str, traj_id: str, valid_options, valid_units) -> str:
        return 'invalid time unit \'{}\' in context trajectory \'{}\'. Valid options are: {}'.format(unit, traj_id, ', '.join(valid_units))

    @staticmethod
    def traj_no_data(traj_id: str) -> str:
        return 'no data source specified for trajectory \'{}\''.format(traj_id)

    @staticmethod
    def traj_no_file(filename: str, traj_id: str) -> str:
        return 'no such file \'{}\' found for trajectory \'{}\''.format(filename, traj_id)

    @staticmethod
    def traj_no_jpl(traj_id: str, filename: str) -> str:
        return 'must specify \'jpl\' as True or False for trajectory \'{}\' to indicate the quaternion convention used in file \'{}\''.format(traj_id, filename)

    @staticmethod
    def traj_no_time(traj_id: str) -> str:
        return 'trajectory \'{}\' format string must contain time indicator: \'t\''.format(traj_id)

    @staticmethod
    def traj_self_align(traj_id: str) -> str:
        return '\'align\' can not be applied to the reference trajectory \'{}\''.format(traj_id)

    @staticmethod
    def traj_unknown_body_frame(traj_id: str, body_frame_id: str, platform_id: str, body_frames) -> str:
        return 'trajectory \'{}\' has body frame set to \'{}\'. No such frame found on platform \'{}\' which has body frame(s): \'{}\''.format(traj_id, body_frame_id, platform_id, ', '.join(body_frames))
