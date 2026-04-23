#!/usr/bin/env python 
from ctypes import alignment
from os import times

import numpy as np
from scipy.spatial.transform import Rotation
import environment
from row_vector_array import RowVectorSequence
from sensor_platform import SensorPlatform
from scipy.spatial.transform import Rotation
from rigid_body import RigidBody
from time_manager import TimeManager
from array_utils import q_conjugate
from environment import Environment
import motion_utils
from alignment import AlignmentSettings, Alignment
from logger_setup import logger
from error_messages import ErrorMessage as err
import datetime
import geometry
from rigid_transform import RigidTransform
from time_manager import TimeCorrespondanceManager


from jecto_scene_object import JectoSceneObject



class Trajectory(JectoSceneObject):
# class SE3Bspline(Trajectory): #control poses
# class SO3Bspline(Trajectory): #control rotations, independent positions
# class PositionBspline(Trajectory): #control positions, independent rotations (rule based)
    def __init__(self, trajectory_id: str, frame_id: str, body_frame_id: str, times: np.ndarray,  positions: np.ndarray, rotations: Rotation, *args, **kwargs):
        assert isinstance(times, np.ndarray) and times.shape==(len(times),), 'Trajectory: times must be a 1D numpy array'
        assert isinstance(rotations, Rotation), 'Trajectory: rotations must be a scipy.spatial.transform.Rotation object'
        assert isinstance(positions, np.ndarray) and positions.shape==(len(positions),3), 'Trajectory: positions must be a 2D numpy array of shape ({},3)'.format(len(times))
        assert len(times) == rotations.__len__() == len(positions), 'Trajectory: length of times, rotations and positions arrays must match'
        self.type = 'data'
        self.id = trajectory_id
        self.frame_id = frame_id
        self.body_frame_id = body_frame_id
        self.n = len(times)
        self.initial_time = times[0]
        self.times = times - self.initial_time 
        self.rotations = rotations
        self.positions = positions

    def apply_time_offset(self, time_offset: float):
        self.times += time_offset



    #these methods return a new Trajectory object, do not modify the original Trajectory in-place
    def reframe(self, frame_id: str, transform: RigidTransform):
        if frame_id == self.frame_id:
            self.get_logger().debug('trajectory reframe called with the already existing frame id \'{}\', not applying transform'.format(frame_id))
            return self
        self.get_logger().debug('reframing trajectory from frame \'{}\' to frame \'{}\' with transform: {}'.format(self.frame_id, frame_id, transform))
        positions, rotations = geometry.reframe(self.positions, self.rotations, transform.pos, transform.rot)
        return Trajectory(self.id, frame_id, self.body_frame_id, self.times, positions, Rotation.from_matrix(rotations))

    def body_reframe(self, body_frame_id: str, body_transform: RigidTransform):
        if body_frame_id == self.body_frame_id:
            self.get_logger().debug('trajectory body_reframe called with the already existing body frame id \'{}\', not applying transform'.format(body_frame_id))
            return self
        positions, rotations = geometry.body_reframe(self.positions, self.rotations, body_transform.pos, body_transform.rot)
        return Trajectory(self.id, self.frame_id, body_frame_id, self.times, positions, Rotation.from_matrix(rotations))
    
    def full_reframe(self, frame_id: str, transform: RigidTransform, body_frame_id: str, body_transform: RigidTransform):
        reframed_traj = self.reframe(frame_id, transform)
        return reframed_traj.body_reframe(body_frame_id, body_transform)



    @classmethod
    def _setup(cls, config: dict, **kwargs) -> tuple:
        """
        Get trajectory arguments from config without initializing the Trajectory object
        """
        try:
            platform, env = kwargs['platform'], kwargs['environment']
        except KeyError:
            raise KeyError('Trajectory _setup requires \'platform\' and \'environment\' keyword arguments')
        
        config = cls.validate_config(config, 'trajectory', context='trajectories')
        trajectory_id = config['id']
        frame_id = config['frame'] 
        if not frame_id in env.frame_ids:
            cls.config_fail(err.traj_frame_not_found(trajectory_id, frame_id, env.id, env.frame_ids))
        
        body_frame_id = config['body_frame'] if 'body_frame' in config else None
        if not body_frame_id in platform.frame_ids:
            cls.config_fail(err.traj_body_frame_not_found(trajectory_id, body_frame_id, platform.id, platform.frame_ids))

        cls.get_logger().debug('trajectory \'{}\': frame_id: \'{}\' body_frame_id: \'{}\''.format(trajectory_id, frame_id, body_frame_id))

        if 'data' in config:
            times, rotations, positions = cls.from_data(config)
            cls.get_logger().info('trajectory \'{}\': successfully loaded trajectory data from file'.format(trajectory_id))
    

        # if 'generate' in traj_config:
        #     generate_config = traj_config['generate']
        #     if 'bspline' in generate_config:
        #         raise NotImplementedError('bspline trajectory generation not yet implemented')
        #     # if 'smoothstep' in generate_config:

        #     # if 'parametric' in generate_config:
        #     #     raise NotImplementedError('parametric trajectory generation not yet implemented')
        #     #     raise NotImplementedError('smoothstep trajectory generation not yet implemented')
        #     time_config = check_keys(generate_config, 'time', context='trajectory generate')
        #     start_time = float(time_config['start'])
        #     end_time = float(time_config['end'])
        #     dt = float(time_config['dt'])
        #     times = np.arange(start_time, end_time+dt, dt)
        

        #     motion_config = check_keys(generate_config, 'motion', context='trajectory generate')
        #     rotations, positions = motion_utils.generate_motion(motion_config, times)
        # # (self, trajectory_id: str, frame_id: str, body_frame_id: str, times: np.ndarray, rotations: Rotation, positions: np.ndarray)
        return (trajectory_id, frame_id, body_frame_id, times, positions, rotations), {'platform': platform, 'environment': env}


    @classmethod
    def from_data(cls, config: dict) -> tuple:
        data_config = cls.validate_config(config['data'], 'data', context='trajectory')
        if 'text_file' in data_config:
            return cls.from_text_file(config)
        if 'rosbag' in config:
            cls.get_logger().error('rosbag trajectory loading not implemented')
            # return cls.from_rosbag(config['rosbag'])
        if 'ulog' in config:
            cls.get_logger().error('ulog trajectory loading not implemented')
        cls.get_logger().error('no valid data source specified for trajectory \'{}\' in configuration file'.format(args[0]))
    
    
    @classmethod
    def generate(cls, config: dict):
        raise NotImplementedError('Trajectory: trajectory generation not yet implemented')
    
    @classmethod
    def from_text_file(cls, config: dict) -> tuple:
        trajectory_id, frame_id, body_frame_id = config['id'], config['frame'], config['body_frame']
        text_file_config = cls.validate_config(config['data']['text_file'], 'text_file', context='data')
        file_path = text_file_config['path']
        cls.get_logger().debug('trajectory \'{}\': loading data from text file: {}...'.format(trajectory_id, file_path))

        if 'delimiter' in text_file_config:
            delimiter = text_file_config['delimiter']
        else:
            delimiter = ',' if file_path.endswith('.csv') else ' '
            cls.get_logger().debug('trajectory \'{}\': no delimiter specified for trajectory text data, using \'{}\' based on file extension \'.{}\''.format(trajectory_id, delimiter, file_path.split('.')[-1]))
        
        try:
            data = np.loadtxt(file_path, delimiter=delimiter, skiprows=1) #TODO: make skiprows a parameter
        except Exception as e:
            cls.config_fail(err.traj_bad_data(trajectory_id, file_path, delimiter, str(e)))
   
        time_unit = text_file_config['time_unit'] 
        cls.get_logger().debug('trajectory \'{}\': interpreting time unit in trajectory data file as {}...'.format(trajectory_id, {'s': 'seconds', 'ms': 'milliseconds', 'us': 'microseconds', 'ns': 'nanoseconds'}[time_unit]))
        format_str = text_file_config['format']
        format_ref = format_str.split()
        cls.get_logger().debug('trajectory \'{}\': text file format string: \'{}\', parsed format components: {}'.format(trajectory_id, format_str, format_ref))
        ignore_cols = [i for i in range(len(format_ref)) if format_ref[i] == '_']
        cls.get_logger().debug('trajectory \'{}\': {} column(s) of data ignored'.format(trajectory_id, len(ignore_cols)))
        has_position_data = any(i in format_ref for i in 'xyz')
        has_rotation_data = 'q' in format_str

        cls.get_logger().debug('trajectory \'{}\': has position date: {}, has rotation data: {}'.format(trajectory_id, has_position_data, has_rotation_data))
        time_scales = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
        
        idx_t = format_ref.index('t')


        
        initial_time = data[0,idx_t]
        n = len(data)
        t = (data[:,idx_t] - initial_time)*time_scales[time_unit] 
        dur = t[-1] - t[0]
        p = np.zeros((n,3))
        q = np.zeros((n,4))

        if has_position_data:
            idx_x = format_ref.index('x') if 'x' in format_ref else None
            idx_y = format_ref.index('y') if 'y' in format_ref else None
            idx_z = format_ref.index('z') if 'z' in format_ref else None
            p[:,0] = (data[:,idx_x] if idx_x is not None else np.zeros(n))
            p[:,1] = (data[:,idx_y] if idx_y is not None else np.zeros(n))
            p[:,2] = (data[:,idx_z] if idx_z is not None else np.zeros(n))

        if has_rotation_data: #use scalar-last order because the default for a scipy Rotation 
            jpl = text_file_config['jpl']
            cls.get_logger().debug('trajectory \'{}\': assuming {} quaternions in text file based on \'jpl\' parameter value: {}'.format(trajectory_id, ['Hamilton', 'JPL'][jpl], jpl))
            
            sign = -1.0 if jpl else 1.0
            q[:,0] = sign * data[:,format_ref.index('qx')]
            q[:,1] = sign * data[:,format_ref.index('qy')]
            q[:,2] = sign * data[:,format_ref.index('qz')]
            q[:,3] = data[:,format_ref.index('qw')]
        
        times = t
        positions = p
        rotations = Rotation.from_quat(q)
        cls.get_logger().debug('trajectory \'{}\': loaded {} rows of data from text file, initial time value: {}s, duration: {}s'.format(
            trajectory_id, n, initial_time, str(datetime.timedelta(seconds=dur))))
        return (times, rotations, positions)
    
    @classmethod
    def from_rosbag(cls, config: dict):
        """
        Load trajectory from rosbag file
        """
        raise NotImplementedError('rosbag trajectory loading not yet implemented')
    

class ConstantTrajectory(Trajectory):
    """
    Trajectory with a constant rotation and position at all times - equivalent to a static frame.
    """
    def __init__(self, trajectory_id: str, frame_id: str, body_frame_id: str, times: np.ndarray, rotation: Rotation, position: np.ndarray):
        _ids = (trajectory_id, frame_id, body_frame_id)
        self.type = 'constant'
        self.position = position
        self.rotation = rotation
        n = len(times)
        positions = np.tile(position, (n, 1)) 
        rotations = Rotation.from_quat(np.tile(rotation.as_quat(), (n, 1)))
        super().__init__(*_ids, times, rotations, positions)

    
class ZeroTrajectory(ConstantTrajectory):
    def __init__(self, trajectory_id: str, frame_id: str, body_frame_id: str, times: np.ndarray):
        _ids = (trajectory_id, frame_id, body_frame_id)
        self.type = 'zero'
        rotation = Rotation.from_quat([0.,0.,0.,1.])
        position = np.array([0.,0.,0.])
        super().__init__(*_ids, times, rotation, position)

class TranslationSequence(): 
    def __init__(self, time_manager: TimeManager, positions: np.ndarray,  velocities: np.ndarray = None, accelerations: np.ndarray = None):
        n = time_manager.n
        times = time_manager.times
        dur = time_manager.dur
        dt = time_manager.dt
        deltas = time_manager.deltas


        #compute numerical vel 
        numerical_vel = np.gradient(positions, axis=0)/(deltas.reshape(n,1)) 
        numerical_acc = np.gradient(numerical_vel, axis=0)/(deltas.reshape(n,1))
        
        pos = positions
        _vel = numerical_vel if (velocities is None) else velocities
        _acc = numerical_acc if (accelerations is None) else accelerations

        self.pos = RowVectorSequence(time_manager, pos)
        self.vel = RowVectorSequence(time_manager, _vel)
        self.acc = RowVectorSequence(time_manager, _acc)
        # self.numerical_vel = RowVectorSequence(numerical_vel)
        # self.numerical_acc = RowVectorSequence(numerical_acc)
        


class RotationSequence():
    def __init__(self, time_manager: TimeManager, rotations: Rotation, angvel: np.ndarray = None, angacc: np.ndarray = None):
        n = time_manager.n
        assert rotations.__len__() == n, 'RotationSequence: length of rotation and times array must match: got {} and {}'.format(rotations.__len__(), n)
        times = time_manager.times
        dur = time_manager.dur
        dt = time_manager.dt
        deltas = time_manager.deltas

        self.rot = rotations
        angvel, body_angvel = motion_utils.angvel_from_rotation_matrices(rotations.as_matrix(),dt)
        angacc = np.gradient(angvel, axis=0)/(deltas.reshape(n,1)) #n x 3 array
        body_angacc = (rotations.as_matrix().swapaxes(1,2)@angacc.reshape(n,3,1)).reshape(n,3)

        self.angvel = RowVectorSequence(time_manager, angvel)
        self.body_angvel = RowVectorSequence(time_manager, body_angvel)
        self.angacc = RowVectorSequence(time_manager, angacc)
        self.body_angacc = RowVectorSequence(time_manager, body_angacc)
        
    

class TrajectoryEvaluation():
    def __init__(self, translation: TranslationSequence, rotation: RotationSequence, body_vel: RowVectorSequence, body_acc: RowVectorSequence):
        self.translation = translation
        self.rotation = rotation
        self.body_vel = body_vel
        self.body_acc = body_acc
    

class ReferenceTrajectory(Trajectory):
    #ReferenceTrajectory.config(kw:platform,env)->Trajecory.config(kw:platform,env)->JectoSceneObject.config(kw:platform,env)->Trajectory._setup(kw:platform,env)->Traj args ->
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = TimeManager(self.times)
        self._platform = kwargs['platform']
        self._environment = kwargs['environment']
    
    



    def _evaluate(self, frame_id: str, body_frame_id: str) -> TrajectoryEvaluation:
        transform = self._environment.relative(self.frame_id, frame_id)
        body_transform = self._platform.relative(self.body_frame_id, body_frame_id)
        self.get_logger().debug('evaluating trajectory of body frame \'{}\' expressed in frame \'{}\''.format(self.body_frame_id, self.frame_id))
        traj = self.full_reframe(frame_id, transform, body_frame_id, body_transform)
        self.get_logger().debug('trajectory reframed to frame \'{}\' and body frame \'{}\''.format(frame_id, body_frame_id))

        translation_sequence = TranslationSequence(self.time, self.positions)
        rotation_sequence = RotationSequence(self.time, self.rotations)

        vel = translation_sequence.vel.values()
        acc = translation_sequence.acc.values()
        R = self.rotations.as_matrix() # n x 3 x 3 array
        n = self.n

        body_vel = RowVectorSequence(self.time,(R.swapaxes(1,2)@vel.reshape(n,3,1)).reshape(n,3)) 
        body_acc = RowVectorSequence(self.time,(R.swapaxes(1,2)@acc.reshape(n,3,1)).reshape(n,3))
        
        return TrajectoryEvaluation(translation_sequence, rotation_sequence, body_vel, body_acc)

        
    def frames(self, frame_id: str, body_frame_id: str) -> TrajectoryEvaluation:
        return self._evaluate(frame_id, body_frame_id)



class MatchedTranslationSequence(TranslationSequence):
    pass

class MatchedRotationSequence(RotationSequence):
    pass


class MatchedTrajectoryEvaluation(TrajectoryEvaluation):
    def __init__(self, time: TimeCorrespondanceManager, translation: TranslationSequence, rotation: RotationSequence, body_vel: RowVectorSequence, body_acc: RowVectorSequence):
        self.time = time
        self.translation = translation
        self.rotation = rotation
        self.body_vel = body_vel
        self.body_acc = body_acc


class MatchedTrajectory(Trajectory):
    def __init__(alignment: Alignment, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #get time correspondance manager based on time offset and two time sequences
        self.times = TimeCorrespondanceManager(ref_times = ref_traj.time.values(), traj_times = traj.times, time_offset = alignment.time_offset)




    def _evaluate(self, frame_id: str, body_frame_id: str) -> TrajectoryEvaluation:
        transform = self._environment.relative(self.frame_id, frame_id)
        body_transform = self._platform.relative(self.body_frame_id, body_frame_id)
        self.get_logger().debug('evaluating trajectory of body frame \'{}\' expressed in frame \'{}\''.format(self.body_frame_id, self.frame_id))
        traj = self.full_reframe(frame_id, transform, body_frame_id, body_transform)
        self.get_logger().debug('trajectory reframed to frame \'{}\' and body frame \'{}\''.format(frame_id, body_frame_id))

        translation_sequence = TranslationSequence(self.time, self.positions)
        rotation_sequence = RotationSequence(self.time, self.rotations)

        vel = translation_sequence.vel.values()
        acc = translation_sequence.acc.values()
        R = self.rotations.as_matrix() # n x 3 x 3 array
        n = self.n

        body_vel = RowVectorSequence(self.time,(R.swapaxes(1,2)@vel.reshape(n,3,1)).reshape(n,3)) 
        body_acc = RowVectorSequence(self.time,(R.swapaxes(1,2)@acc.reshape(n,3,1)).reshape(n,3))
        
        return MatchedTrajectoryEvaluation(translation_sequence, rotation_sequence, body_vel, body_acc)
    
    def frames(self, frame_id: str, body_frame_id: str) -> MatchedTrajectoryEvaluation:
        return self._evaluate(frame_id, body_frame_id)



    @classmethod
    def _setup(cls, config: dict, **kwargs) -> tuple:
        #need to get the alignment info, init matchedtraj
        ref_traj = kwargs['reference_trajectory']
        platform, env = ref_traj._platform, ref_traj._environment
        traj_args = super()._setup(config, platform=platform, environment=env)
        alignment_config = cls.validate_config(config['alignment'], 'alignment', context='trajectory')
        return ref_traj, alignment, traj_args, {'platform': platform, 'environment': env}

   

        
    

# class MatchedTrajectoryEvaluation():
#     def __init__(self, trajectory_id: str, time: TimeManager, rotations: Rotation, positions: np.ndarray, error_rotations: Rotation, error_positions: np.ndarray):
#         self.id = trajectory_id
#         self.time = time
#         self.n = len(positions)
#         assert self.n == sum(time.mask), 'TrajectoryEvaluation: length of positions array does not match number of matched times'

#         all_times = time.values(match=False)
#         matched_times = time.values(match=True)

       
    
#         self.translation = TranslationSequence(all_times, positions)
#         self.rotation = RotationSequence(all_times, rotations)
#         self.error = ErrorPoseSequence(matched_times, error_positions, error_rotations) 

#         # #Compute body-frame velocities and accelerations (for the platform frame)
#         R = trajectory.rotations.as_matrix() # n x 3 x 3 array
#         n  = trajectory.n
#         vel = self.translation.vel.values()
#         acc = self.translation.acc.values()
#         self.body_vel = RowVectorSequence(times,(R.swapaxes(1,2)@vel.reshape(n,3,1)).reshape(n,3),mask) 
#         self.body_acc = RowVectorSequence(times,(R.swapaxes(1,2)@acc.reshape(n,3,1)).reshape(n,3),mask)
        
#         def position_rmse():
#             pass
#         def rotation_angle_rmse():
#             pass #return an angle value?
#         # self.rotation = trajectory.rotation
#         # self.body_vel = trajectory.body_vel
#         # self.body_acc = trajectory.body_acc

#     @classmethod
#     def as_reference(cls, trajectory: Trajectory):
#         # trivial time manager and zero error since this is the reference trajectory
#         pass






# the complicated general way to do the transform
 # def _transfer_trajectory(self, frame, body_frame):  
        # """
        # Get the trajectory correponding to the given static frame 'frame' and platform body frame 'body_frame'.
        # Transform both the ref traj and non-ref traj as needed
        # in both cases: if static_frame=traj.frame_id and platform_frame=traj.body_frame_id, transforms are identity, nothing will be changed

        # Use the following identifiers for the transform variables
        #     frame 'a' = static frame in which the trajectory is defined
        #     frame 'b' = static frame given

        #     frame '0' = platform body frame corresponding to the trajectory
        #     frame '1' = platform body frame given


        # After applying the transform to both trajectories, use the results it to initialize and return an TrajectoryEvaluation
        # """
        # frame_index, frame_id = self._env._check_frame_exists(frame)
        # body_frame_index, body_frame_id = self._platform._check_frame_exists(body_frame)

        # #static environment frames
        # _a_ref = self._ref_traj.frame_id
        # _a = self._traj.frame_id
        # _b = frame_id
        # print('TrajectoryEvaluator: static frames: _a_ref: {}, _a: {}, _b: {}'.format(_a_ref, _a, _b))

        # #platform body frames
        # _0_ref = self._ref_traj.body_frame_id
        # _0 = self._ref_traj.body_frame_id
        # _1 = body_frame_id
        # print('TrajectoryEvaluator: platform frames: _0_ref: {}, _0: {}, _1: {}'.format(_0_ref, _0, _1))

        # #static transform values
        # _aRb_ref = self._env.relative(_a_ref, _b).rot.as_matrix()
        # _aPb_ref = self._env.relative(_a_ref, _b).pos
        # _aRb = self._env.relative(_a,_b).rot.as_matrix()
        # _aPb = self._env.relative(_a, _b).pos
        # print('TrajectoryEvaluator: static transform values: _aRb_ref: {}, _aPb_ref: {}, _aRb: {}, _aPb: {}'.format(
        #     Rotation.from_matrix(_aRb_ref).as_euler('xyz',degrees=True), _aPb_ref, Rotation.from_matrix(_aRb).as_euler('xyz',degrees=True), _aPb))
        
        # #platform transform values
        # _0R1_ref = self._platform.relative(_0_ref, _1).rot.as_matrix()
        # _0P1_ref = self._platform.relative(_0_ref, _1).pos
        # _0R1 = self._platform.relative(_0, _1).rot.as_matrix()
        # _0P1 = self._platform.relative(_0, _1).pos
        # print('TrajectoryEvaluator: platform transform values: _0R1_ref: {}, _0P1_ref: {}, _0R1: {}, _0P1: {}'.format(
        #     Rotation.from_matrix(_0R1_ref).as_euler('xyz',degrees=True), _0P1_ref, Rotation.from_matrix(_0R1).as_euler('xyz',degrees=True), _0P1))

        #existing trajectory values
        # n_ref = self._ref_traj.n
        # n = self._traj.n
        # _aR0_ref = self._ref_traj.rotations.as_matrix() # n_ref x 3 x 3 
        # _aR0 = self._traj.rotations.as_matrix() # n x 3 x 3
        # _aP0_ref = self._ref_traj.positions # n_ref x 3 
        # _aP0 = self._traj.positions # n x 3 
        # print('existing: _aR0_ref shape: {}, _aR0 shape: {}, _aP0_ref shape: {}, _aP0 shape: {}'.format(
        #     _aR0_ref.shape, _aR0.shape, _aP0_ref.shape, _aP0.shape))

        # #transformed rotations:
        # _bR1_ref = (_aRb_ref.T).reshape(1,3,3) @ _aR0_ref @ _0R1_ref.reshape(1,3,3) # n_ref x 3 x 3
        # _bR1 = (_aRb.T).reshape(1,3,3) @ _aR0 @ _0R1.reshape(1,3,3)  # n x 3 x 3

        # #transformed positions:
        # #METHOD 1
        # term1_ref = (-_aRb_ref.T @ _aPb_ref).reshape(3,1)
        # term2_ref = _aRb_ref.T@_aP0_ref.T
        # term3_ref = np.squeeze(_aRb_ref.T@(_aR0_ref@_0P1_ref.reshape(3,1))).T
        # _bP1_ref = (term1_ref + term2_ref + term3_ref).T

        # term_1 = (-_aRb.T @ _aPb).reshape(3,1) 
        # term_2 = _aRb.T@_aP0.T
        # term_3 = np.squeeze(_aRb.T@(_aR0@_0P1.reshape(3,1))).T
        # _bP1 = (term_1 + term_2 + term_3).T

        # print('METHOD 1')
        # print('term1_ref shape:', term1_ref.shape)
        # print('term2_ref shape:', term2_ref.shape)
        # print('term3_intermediate_ref shape:', (_aRb_ref.T@(_aR0_ref@_0P1_ref.reshape(3,1))).shape)
        # print('term3_ref shape:', term3_ref.shape) 
        # print('_bP1_ref shape:', _bP1_ref.shape)
        # print('')
        # print('term_1 shape:', term_1.shape)
        # print('term_2 shape:', term_2.shape)
        # print('term_3_intermediate shape:', (_aRb.T@(_aR0@_0P1.reshape(3,1))).shape)
        # print('term_3 shape:', term_3.shape)
        # print('_bP1 shape:', _bP1.shape)
        # print('')

        #==================================================== todo: see which method is faster
        #METHOD 2 factor out the _aRb_ref.T
        # term1_ref = -_aPb_ref.reshape(3,1)
        # term2_ref = _aP0_ref.T
        # term3_ref = np.squeeze(_aR0_ref @ _0P1_ref.reshape(3,1)).T
        # _bP1_ref_method2 = (_aRb_ref.T @ (term1_ref + term2_ref + term3_ref)).T
        
        # term_1 = -_aPb.reshape(3,1)
        # term_2 = _aP0.T
        # term_3 = np.squeeze(_aR0 @ _0P1.reshape(3,1)).T
        # _bP1_method2 = (_aRb.T @ (term_1 + term_2 + term_3)).T
        
        # print('METHOD 2')
        # print('term1_ref shape:', term1_ref.shape)
        # print('term2_ref shape:', term2_ref.shape)
        # print('term3_intermediate_ref shape:', (_aR0_ref @ _0P1_ref.reshape(1,3,1)).shape)
        # print('term3_ref shape:', term3_ref.shape)
        # print('_bP1_ref shape:', _bP1_ref.shape)    
        # print('')
        # print('term_1 shape:', term_1.shape)
        # print('term_2 shape:', term_2.shape)
        # print('term_3_intermediate shape:', (_aR0 @ _0P1.reshape(1,3,1)).shape)
        # print('term_3 shape:', term_3.shape)
        # print('_bP1 shape:', _bP1.shape)
        # assert np.allclose(_bP1_ref, _bP1_ref_method2), 'TrajectoryEvaluator: _bP1_ref does not match _bP1_ref_method2'
        # assert np.allclose(_bP1, _bP1_method2), 'TrajectoryEvaluator: _bP1 does not match _bP1_method2'
