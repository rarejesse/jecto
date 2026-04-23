#!/usr/bin/env python
import numpy as np
from array_utils import unit
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

"""
This file (geometry.py) is for functions that involve 3D positions and rotations, but not time.

Functions that involve time (e.g. for differentiating trajectory data) are in motion.py

All functions in this file should input and output generic types (builtins and numpy arrays)
to maximize reusability. The one exception is the scipy Rotation object for rotations, to maintain 
consistency with the rest of the codebase.
"""

def skew(vec):
    """
    compute the skew symmetric matrix corresponding to a 3D vector
    params:
        vec: array of length 3
    returns:
        skew symmetric matrix of shape (3,3)
    """

    return np.array([[0,-vec[2],vec[1]],
                     [vec[2],0,-vec[0]],
                     [-vec[1],vec[0],0]])

def rotation_align_axis(axis,vec,grounded_axis=None,flip=False):
    """
    compute the rotation(s) that aligns given axis with the given vector(s), 
    and keeps one other axis parallel to the original xy-plane (ground plane)

    params:
        axis: string, 'x','y', or 'z' axis to align with vec
        vec: 1d array of length 3 (vector to align axis with) or 2d array of these vectors with shape (n,3)
        grounded: string, 'x','y', or 'z' axis to remain parallel to the original xy-plane.
                  cannot be the same as 'axis' param.
                  if not provided, it will be the next sequential axis after 'axis' (x->y, y->z, z->x).
        flip: bool, if True, the grounded axis will be negated, giving the other available solution
    returns:
        R: rotation matrix where columns are the unit vectors of aligned frame 
           expressed in the same frame as provided vectors. If multiple vectors are provided,
           R will be a 3d array with shape (3,3,n)

    use: a1 = aligned axis, a2 = grounded axis, a3 = third axis
    """
    remaining = [0,1,2] #axis numbers corresponding to 'xyz'
    try:
        aligned_axis_id = 'xyz'.index(axis)
        remaining.remove(aligned_axis_id)
        if not grounded_axis:
            constrained_axis_id = (aligned_axis_id+1)%3 #next sequential axis number
        elif grounded_axis==axis:
            raise ValueError('axes cannot be the same')
        else:
            constrained_axis_id='xyz'.index(grounded_axis)
        remaining.remove(constrained_axis_id)
        third_axis_id = remaining[0] #whichever the last remaining axis number is
    except ValueError:
        print('axis names must be \'x\',\'y\', or \'z\'')
        raise 

    positive_seq = bool(axis+grounded_axis in ['xy','yz','zx'])
    tol = 1e-6 #tolerance for nearly vertical alignment vectors
    sign = -1.0 if flip else 1.0

    if vec.ndim == 1: #handle the case of a single vector
        a1 = unit(vec)
        x,y,z = a1
        if 1.0-abs(z) < tol:
            a2 = sign*np.array([1.,0.,0.])
        else:
            a2 = unit(sign*np.array([y,-x,0]))
        a3 = unit((1.0 if positive_seq else -1.0)*np.cross(a1,a2))
        # print('a1: {}, a2: {}, a3: {}'.format(np.linalg.norm(a1),np.linalg.norm(a2),np.linalg.norm(a3)))
        R = np.zeros((3,3))
        R[:,aligned_axis_id] = a1
        R[:,constrained_axis_id] = a2
        R[:,third_axis_id] = a3
        return R

    n = len(vec)
    a1 = np.zeros((n,3)).astype(np.float32)
    a2 = np.zeros((n,3)).astype(np.float32)
    a3 = np.zeros((n,3)).astype(np.float32)
    R = np.zeros((n,3,3)) #output array of rotation matrices
    
    a1[:,:] = unit(vec) #first axis
    abs_z = np.abs(a1[:,2]).flatten()
    mask = 1.0-abs_z>tol #False where z is nearly 1 or -1

    #for now, enforce that the first and last entry in 'vec' are not nearly vertical
    #TODO: don't enforce this, find nearest non-vertical vector from both ends and the corresponding grounded axis, use that as the grounded axis
    # better way:
    # choose one of the following, given vec=[a,b,c]
    # [ b,-a,0] (c is not near 1)
    # [-b, a,0]
    # [-c, 0 a] (b is not is near 1)
    # [c,0,-a]
    # [0,c,-b] (a is not is near 1)
    # [0,-c,b]
    if not mask[0] or not mask[-1]:
        raise ValueError('rotation_align_axis: first and last alignment vector must not be nearly vertical')

    # for each non-vertical alignment axis [x,y,z]: grounded axis = [y,-x,0] (or [-y,x,0] if flip=True)
    a2[mask,0] =  a1[mask,1]
    a2[mask,1] =  -a1[mask,0]
    a2 = unit(a2) #renormalize
    
    mask = mask.astype(int)
    start = np.where(np.diff(mask)==-1)[0] #indices right before each group of vertical vectors starts
    end = np.where(np.diff(mask)==1)[0]+1 #indices right after each group of vertical vectors ends
    assert len(start)==len(end)

    num_groups = len(start) #number of groups of nearly vertical vectors within 'vec' array
    for i in range(num_groups):
        s = start[i]
        e = end[i]
        ng = e-s+1 #number of vectors in the group, including the two bounding vectors
        a2[s:e+1,:] = unit(linear_interp(a2[s],a2[e],ng))

    a2[:,:] = sign*a2 #negates the 'grounded' axis if flip=True, since there are two possible solutions
    a3[:,:] = (1.0 if positive_seq else -1.0)*np.cross(a1,a2) #third axis

    R[:,:,aligned_axis_id] = a1
    R[:,:,constrained_axis_id] = a2
    R[:,:,third_axis_id] = a3

    # for i in np.arange(n): #debug
    #     print('i: {} a1: {} a2: {} a3: {}'.format(i,a1[i],a2[i],a3[i]))
    return R

def as_scipy_rotation(rot):
    """
    wrapper to initialize a scipy rotation
    return scipy Rotation object based on dimensions of given the input
    return rotation object

     if given: 4x1 vector, assume quaternion [x,y,z,w]
            3x3 matrix, assume rotation matrix
            3x1 vector, assume roll-pitch-yaw = (x,y,z) 
                                with convention x --> y --> z in fixed frame
                                (equivalent to z --> y --> x in body frame)
    """
    #if its already a scipy roation object, return it
    if isinstance(rot,Rotation):
        return rot
    if isinstance(rot,list):
        rot = np.array(rot)
    n = len(rot)
    dim = rot.shape if isinstance(rot,np.ndarray) else (n,)
    if dim == (4,) or dim == (n,4):
        out = Rotation.from_quat(rot) #assumes hamilton
    elif dim == (3,3) or dim == (n,3,3):
        out = Rotation.from_matrix(rot)
    elif dim == (3,) or dim == (n,3):
        out = Rotation.from_euler('xyz', rot, degrees=True)
    else:
        raise ValueError('Invalid rotation input with shape {}'.format(dim))
    return out



def random_point_set(center, radius, num):
    """
    generate 'n' random points uniformly distributed within a sphere with 'radius' and 'center'
    """
    c = center
    n = num
    r_max = radius

    v = np.random.uniform(-1,1,(n,3))
    u = (v.T/np.linalg.norm(v,axis=1)) #random unit vectors
    r = np.cbrt(np.random.uniform(0,r_max**3,n)) #random scales, prevent clustering at center
    points = np.multiply(u,r).T.astype(np.float32) + np.tile(c,(n,1))
    return points


def planar_point_set(center, normal, radius, num=None, grid_spacing=None):
    c = center
    n = num
    r_max = radius
    # print('planar_point_set: center: {}, normal: {}, radius: {}'.format(c,normal,r_max))
    if grid_spacing is not None:
        _min = -r_max/2
        _max = r_max/2+grid_spacing #add one grid_spacing to the maximum to include the last point
        X,Y = np.mgrid[_min:_max:grid_spacing, _min:_max:grid_spacing]
        xy = np.vstack((X.ravel(),Y.ravel())).T
        points_xy = np.hstack((xy,np.zeros((xy.shape[0],1)))).astype(np.float32)
        n = len(points_xy)
    elif n is not None:
        v = np.hstack((np.random.uniform(-1,1,(n,2)),np.zeros((n,1)))) #z component is zero
        u = (v.T/np.linalg.norm(v,axis=1)) #random unit vectors on xy plane
        r = np.sqrt(np.random.uniform(0,r_max**2,n)) #random scales, prevent clustering at center
        points_xy = np.multiply(u,r).T.astype(np.float32)
    else:
        raise ValueError('planar_point_set: provide either number of points or grid_spacing')
    # print('center: {}\nradius: {}\nnormal: {}\npoints_xy: {}'.format(center.shape,radius.shape,normal.shape,points_xy.shape))
    R = rotation_align_axis(axis='z', vec=normal, grounded_axis='x')
    assert np.allclose(np.linalg.det(R),1.0)
    assert np.allclose(R@R.T,np.eye(3),atol=1e-4), 'rotation matrix is not orthogonal within tolerance'

    #print the shape of all of these variables:
    points = (R@points_xy.T).T + np.tile(center,(n,1))
    return points

def linear_interp(a,b,n): 
    """
    compute linear interpolation with 'n' steps from vector 'a' to vector 'b', including 'a' and 'b'
    returns: array of shape (n,3)
    """
   
    t = np.linspace(0,1,n).reshape(-1,1)
    return (1-t)*a + t*b

def smooth_interp(a,b,n):
    """
    compute smoothstep interpolation with 'n' steps from vector 'a' to vector 'b', including 'a' and 'b'
    returns: array of shape (n,3)
    """
    t = np.linspace(0,1,n).reshape(-1,1)
    return a + (b-a)*t**2*(3-2*t)


def linear_interp_so3(a,b,n):
    """
    compute linear interpolation with 'n' steps from rotation matrix 'a' to rotation matrix 'b', including 'a' and 'b'
    returns: array of shape (n,3,3)
    """
    # R[i+1] = R[i]@expm(utils.skew(wm[i])*dt)
    pass


def get_point_positions(pos, rot, points): #should exist somewhere else probably
    n = len(pos) #number of poses
    m = len(points) #number of features

    gPgf = points # m x 3
    gPgc = pos # n x 3
    gRc  = rot.as_matrix() # n x 3 x 3

    assert gPgf.shape == (m,3)
    assert gPgc.shape == (n,3)

    #reshape for broadcasting
    gPgf = gPgf.reshape(1,m,3).swapaxes(1,2)
    gPgc = gPgc.reshape(n,3,1) 
    assert gPgf.shape == (1,3,m)
    assert gPgc.shape == (n,3,1)

    cRg = gRc.swapaxes(1,2) # n x 3 x 3 (transpose each rotation matrix)
    assert cRg.shape == (n,3,3)

    cPcg = (-cRg@gPgc).swapaxes(1,2) # n x 3 x 3 @ n x 3 x 1 = n x 3 x 1 -> swapaxes -> n x 1 x 3
    assert cPcg.shape == (n,1,3)

    cPgf = (cRg@gPgf).swapaxes(1,2) # n x 3 x 3 @ 1 x 3 x m = n x 3 x m -> swapaxes -> n x m x 3
    assert cPgf.shape == (n,m,3)

    cPcf = cPcg + cPgf # n x m x 3 + n x 1 x 3 = n x m x 3
    assert cPcf.shape == (n,m,3)

    return cPcf
    
def radtan_distort(self, points, coefficients):
    """
    apply radial and tangential distortion to a set of points
    
    params:
        points: n x 2 array of uv points
        coefficients: 4 element array of distortion coefficients
    returns: n x 2 array of distorted points
    """
    k1,k2,p1,p2 = coefficients
    #TODO apply the distortion
    return points
        

def transfer(positions: np.array, rotations: Rotation, pos_transform: np.array, rot_transform: Rotation) -> tuple:
    """
    should move this to motion_utils 
    should call this something else, like 'transfer', and have another method that 
    transforms the trajectory to another fixed frame

    transform the trajectory to another rigidly attached frame - apply relative transform at every time step - express result in global frame
    return the new trajectory expressed in the global frame
    params:
        rotation: rotation matrix (axes of the new frame expressed in the current body frame) or euler angles (fixed-axis xyz)
        translation: position vector (origin of the new frame expressed in the current body frame)
    returns:
        traj: new Trajectory object corresponding to the new frame
    """
    #current position and rotation arrays
    R = self.rotation.rot.as_matrix()
    pos = self.translation.pos.values
    t = self.times #timestamps

    #static transform given
    _rot = as_scipy_rotation(rotation).as_matrix()
    R_static = _rot.reshape(1,3,3) #reshape for broadcasting
    pos_static = translation

    #transform each posec
    R_new = R@R_static # (n x 3 x 3) @ (1 x 3 x 3) = n x 3 x 3
    pos_new = pos + np.squeeze(R@pos_static.reshape(1,3,1)) 
    
    traj = Trajectory.from_arrays(pos_new,R_new,t=t,frame=self.frame,body_frame=self.body_frame,_id=self.id)

    return new_positions, new_rotations
    # return transferred_positions, transferred_rotations

def reframe(positions: np.array, rotations: Rotation, pos_transform: np.array, rot_transform: Rotation) -> tuple:
    """
    compute the trajectory of the same body frame expressed in a different fixed frame
    """
    R = rot_transform.as_matrix().T 
    
    new_rotations = R.reshape(1,3,3)@rotations.as_matrix() # (1 x 3 x 3) @ (n x 3 x 3) = (n x 3 x 3)
    new_positions = (R@(positions - pos_transform).T).T # (n x 3)
    print('reframe: new_positions shape: {}, new_rotations shape: {}'.format(new_positions.shape, new_rotations.shape))
    return new_positions, new_rotations

def body_reframe(positions: np.array, rotations: Rotation, pos_transform: np.array, rot_transform: Rotation) -> tuple:
    """
    compute the trajectory of a different rigidly-attached body frame expressed in the same fixed frame
    """
    R = rot_transform.as_matrix()
    
    new_rotations = rotations.as_matrix()@R.reshape(1,3,3) # (n x 3 x 3) @ (1 x 3 x 3) = n x 3 x 3
    new_positions = positions + np.squeeze(rotations.as_matrix()@pos_transform.reshape(1,3,1)) # (n x 3) + squeeze((n x 3 x 3) @ (1 x 3 x 1)) = (n x 3)
    print('body_reframe: new_positions shape: {}, new_rotations shape: {}'.format(new_positions.shape, new_rotations.shape))
    return new_positions, new_rotations

def transform(positions: np.array, rotations: Rotation, pos_transform: np.array, rot_transform: Rotation) -> tuple:
    """
    transform the trajectory to another fixed frame - apply relative transform at every time step - express result in global frame
    """
    #current position and rotation arrays
    R = rotations
    pos = positions

    #static transform given
    _rot = as_scipy_rotation(rot_transform).as_matrix()
    R_static = _rot.reshape(1,3,3) #reshape for broadcasting
    pos_static = pos_transform

    #transform each pose
    R_new = R_static@R # (1 x 3 x 3) @ (n x 3 x 3) = n x 3 x 3
    pos_new = np.squeeze(R_static.reshape(1,3,3)@pos.reshape(-1,3,1)) + pos_static.reshape(1,3) 
    
    return new_positions, new_rotations
    # return transformed_positions, transformed_rotations

def rescale(positions: np.ndarray, scale: float) -> np.ndarray:
    """
    scale the trajectory positions by a scale factor
    """
    #current position and rotation arrays
    R = rotations
    pos = positions

    pos_new = pos * scale_factor
    
    return pos_new, rotations
    # return scaled_positions, rotations


def kabsh_umeyama(ref_positions: np.ndarray, positions: np.ndarray, centering_mode: str = 'centroid', weights: np.ndarray = None, yaw_only: bool = False, error_metric: str = 'rmse') -> tuple:
    """
    compute the optimal rotation and translation that aligns two sets of 3D points A and B using the Kabsh-Umeyama algorithm
    A and B are arrays of shape (n,3)
    returns:
        R: rotation matrix (3x3)
        t: translation vector (3,)
        error: rmse or mae alignment error after applying the transform
    """
    pass


def inverse_camera_projection(positions: np.ndarray, rotations: Rotation, mask: np.ndarray = None, density: int = 100, optical_axis: str = 'z', fov: float = 60.0, depth_range: tuple = (1.0,10.0), min_distance: float = 0.0) -> np.ndarray:
    """
    features = geometry.random_inverse_projections(
        positions=ref_traj.positions,
        rotations=ref_traj.rotations,
        mask=mask, #mask corresponding to 'rate'
        density=200, #per pose, N=16546 total
        optical_axis='z',
        fov=60.0,
        depth_range=(1.0, 10.0),
        min_distance=0.1, #sweep the trajectory after generating the points
        colors=['red','purple'],
        colors_per_projection=1)
    """
    mask = np.ones(len(positions),dtype=bool) if mask is None else mask
    cam_pos = positions[mask]
    cam_rot = rotations[mask]
    points = np.zeros((len(mask),density,3)).astype(np.float32)
    flattened_poses = np.hstack((cam_pos, cam_rot.as_matrix().reshape(-1,9))) #n x 12 array
    #nx12 -> n*density x 3
    inverse_projection_args = (density, optical_axis, fov, depth_range[0])
    points = np.apply_along_axis(inverse_projection, 1, flattened_poses, *inverse_projection_args)
    distance_mask = np.apply_along_axis()
    return points[distance_mask]
    
    

def inverse_projection(flattened_pose: np.ndarray, density: int, optical_axis: str, fov: float, depth: float) -> np.ndarray:
    R = flattened_pose[3:].reshape(3,3)
    t = flattened_pose[0:3].reshape(3,1)
    z = np.random.uniform(low=depth_min, high=depth_max, size=density)
    z_tan_theta = z*np.tan(np.radians(fov)/2)
    y = z_tan_theta * np.random.uniform(-1,1, size=density)
    x = z_tan_theta * np.random.uniform(-1,1, size=density)
    points = (R@np.vstack((x,y,z)) + t.reshape(3,1)).T
    

def distance_mask(position: np.ndarray, points: np.ndarray, min_distance: float) -> np.ndarray:
    diff = points - position.reshape(1,3)
    mask = np.sum(diff*diff, axis=1) >= min_distance**2
    return mask

# def color_assign_group(colors, group_size, num_points):
#     pass

# def color_assign_binary(color_true, color_false, mask):
#     # assign two colors by binary mask
#     pass

# def color_map(colors=['green','yellow','red'], range=(0.0,1.0), values=None,rep='rgb'):
#     # assign colors to points from a list of colors
#     pass

# def trajectory_transform(traj: Trajectory, rot: Rotation, pos):
#     """
#     apply rotation and translation to trajectory
#     traj: trajectory object
#     rot: scipy Rotation object
#     pos: 3d position vector
#     returns: new trajectory object with transformed positions and rotations
#     """
#     pos_orig = traj.translation.pos.values
#     rot_orig = traj.rotation.rot.values

#     n = len(pos_orig)
#     pos_new = np.zeros((n,3)).astype(np.float32)
#     rot_new = np.zeros((n,4)).astype(np.float32)

#     for i in range(n):
#         pos_new[i] = rot.as_matrix()@pos_orig[i] + pos
#         rot_new[i] = (rot*Rotation.from_quat(rot_orig[i])).as_quat()

#     traj_new = traj.copy()
#     traj_new.translation.pos.values = pos_new
#     traj_new.rotation.rot.values = rot_new

#     return traj_new



# def bearing_vectors(self,target):
#     """
#     Compute bearing vectors from each position point in the trajectory to either a fixed global point,
#     or to each point in another (equally sized) trajectory
#     params
#         target: 1D array of length 3, or 2D array of shape (n,3), where n = self.n
#     returns:
#         bearing_vectors: 2D array of shape (n,3)
#     """
#     n = self.n
#     if target.shape==(3,) or target.shape==(n,3):
#         bearing_vectors = target - self.pos.values
#         return bearing_vectors
#     else:
#         raise ValueError('bearing vector target must have shape (3,) or ({},3), got shape {}'.format(n,target.shape))
