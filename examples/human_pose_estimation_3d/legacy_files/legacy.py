def read_3d_data(actions, data_dir, camera_frame, camera_params_dict,
                 predict_14=False):
    """Loads 3D poses, zero-centres and normalizes them

    # Arguments
        actions: list of strings. Actions to load
        data_dir: string. Directory where the data can be loaded from
        camera_frame: boolean. Whether to convert the data to camera coordinates
        camera_params_dict: dictionary with camera parameters
        predict_14: boolean. Whether to predict only 14 joints
        
    # Returns
        train_set: dictionary with loaded 3d poses for training
        test_set: dictionary with loaded 3d poses for testing
        data_mean: vector with the mean of the 3d training data
        data_std: vector with the standard deviation of the 3d training data
        dim_to_ignore: list with the dimensions to not predict
        dim_to_use: list with the dimensions to predict
        train_root_positions: dictionary with the 3d positions of the root in train
        test_root_positions: dictionary with the 3d positions of the root in test
    """
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)
    if camera_frame:
        train_set = transform_world_to_camera(train_set, camera_params_dict)
        test_set = transform_world_to_camera(test_set, camera_params_dict)
    train_set, train_root_positions = postprocess_3d(train_set)
    test_set, test_root_positions = postprocess_3d(test_set)
    train_data = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(
        train_data, dim=3, predict_14=predict_14)
    train_set = normalize_data(train_set, data_mean[dim_to_use],
                               data_std[dim_to_use])
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)
    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions
    
    
    
def create_2d_data(actions, data_dir, rcams):
    """Creates 2D poses by projecting 3D poses with the corresponding camera
       parameters. Also normalizes the 2D poses

    # Arguments
        actions: list of strings. Actions to load
        data_dir: string. Directory where the data can be loaded from
        rcams: dictionary with camera parameters

    # Returns
        train_set: dictionary with projected 2d poses for training
        test_set: dictionary with projected 2d poses for testing
        data_mean: vector with the mean of the 2d training data
        data_std: vector with the standard deviation of the 2d training data
        dim_to_ignore: list with the dimensions to not predict
        dim_to_use: list with the dimensions to predict
    """
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)
    train_set = project_to_cameras(train_set, rcams)
    test_set = project_to_cameras(test_set, rcams)
    train_data = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(
        train_data, dim=2)
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)
    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use
    
    
def postprocess_3d(poses_set):
    """Center 3D points around root
    
    # Arguments
        poses_set: dictionary with 3d data

    # Returns
        poses_set: dictionary with 3d data centred around root (center hip) joint
        root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    for key in poses_set.keys():
        root_positions[key] = copy.deepcopy(poses_set[key][:, :3])
        poses = poses_set[key]
        poses = poses - np.tile(poses[:, :3], [1, len(H36M_NAMES)])
        poses_set[key] = poses
    return poses_set, root_positions
    
    
    
    
def project_to_cameras(poses_3D_dict, cameras, num_cameras=4):
    """Project 3D poses using camera parameters

    # Arguments
        poses3d_set: dictionary with 3D poses
        cams: dictionary with camera parameters
        ncams: number of cameras per subject
        
    # Returns
        poses2d_set: dictionary with 2D poses
    """
    poses_3D_dict = {}
    poses2D_set = {}
    for pose in sorted(poses_3D_dict.keys()):
        subject, action, sequence_name = pose
        points_3D = poses_3D_dict[pose]
        for camera in range(num_cameras):
            poses2D_set = camera_to_2D(camera, points_3D, poses2D_set, subject,
                                       action, sequence_name)
    return poses2D_set


def camera_to_2D(camera, points_3D, poses2D_set, subject, action,
                 sequence_name):
    """Project 3D poses using camera parameters

    # Arguments
        camera: camera subject
        points_3D: Nx3 points in world coordinates
        poses2d_set: dictionary with 2D poses
        subj:
        action:
        seqname:
        
    # Returns
        poses2d_set: dictionary with 2D poses
    """
    rotation, translation, focal_length, camera_center, camera_radial_distortion, camera_tangential_distortion, camera_id = \
        camera[(subject, camera + 1)]
    points_2D, _, _, _, _ = cameras.project_point_radial(
        np.reshape(points_3D, [-1, 3]), rotation, translation, \
        focal_length, camera_center, camera_radial_distortion,
        camera_tangential_distortion)
    points_2D = np.reshape(points_2D, [-1, len(H36M_NAMES) * 2])
    sequence_name = sequence_name[:-3] + camera_id + ".h5"
    poses2D_set[(subject, action, sequence_name)] = points_2D
    return poses2D_set
    
def transform_world_to_camera(poses3D_dict, cameras, num_cameras=4):
    """Project 3D poses from world coordinate to camera coordinate system

    # Arguments
        poses3D_dict: dictionary with 3d poses
        cameras: dictionary with cameras
        ncams: number of cameras per subject
        
    # Returns
        camera_poses_3D: dictionary with 3d poses in camera coordinate
    """
    camera_poses_3D = {}
    for pose in sorted(poses3D_dict.keys()):
        subject, action, sequence_name = pose
        points_3D_world = poses3D_dict[pose]
        for camera in range(num_cameras):
            camera_poses_3D = world_to_camera(camera, camera_poses_3D,
                                              points_3D_world, subject,
                                              action, sequence_name)

    return camera_poses_3D


def world_to_camera(camera, camera_poses_3D, points_3D_world, subject, action,
                    sequence_name):
    """Project 3D poses from world coordinate to camera coordinate system

    # Arguments
        points: dictionary with 3d poses
        camera_poses_3D: dictionary with cameras
        subject: number of cameras per subject
        action:
        sequence_name:
        
    # Returns
        camera_poses_3D: dictionary with 3d poses in camera coordinate
    """
    rotation, translation, _, _, _, _, name = camera[(subject, camera + 1)]
    camera_cordinates = cameras.world_to_camera_frame(
        np.reshape(points_3D_world, [-1, 3]), rotation, translation)
    camera_cordinates = np.reshape(camera_cordinates,
                                   [-1, len(H36M_NAMES) * 3])
    sname = sequence_name[:-3] + name + ".h5"  # e.g.: Waiting 1.58860488.h5
    camera_poses_3D[(subject, action, sequence_name)] = camera_cordinates
    return camera_poses_3D
    
    
def load_data(bpath, subjects, actions, dim=3):
    """Loads 2D ground truth from disk, and puts it in an easy-to-acess dictionary

    # Arguments
        bpath: String. Path where to load the data from
        subjects: List of integers. Subjects whose data will be loaded
        actions: List of strings. The actions to load
        dim: Integer={2,3}. Load 2 or 3-dimensional data
        
    # Returns
        data: Dictionary with keys k=(subject, action, seqname)
        values v=(nx(32*2) matrix of 2d ground truth)
        There will be 2 entries per subject/action if loading 3d data
        There will be 8 entries per subject/action if loading 2d data
    """
    data = {}
    for subject in subjects:
        for action in actions:
            dpath = os.path.join(bpath, 'S{0}'.format(subject),
                                 'MyPoseFeatures/D{0}_Positions'.format(dim),
                                 '{0}*.cdf'.format(action))
            file_names = glob.glob(dpath)
            loaded_seqs = 0
            data = parse_data(file_names, action, loaded_seqs, subject, data)

    return data


def parse_data(file_names, action, loaded_seqs, subject, data):
    for file_name in file_names:
        sequence_name = os.path.basename(file_name)
        if action == "Sitting" and sequence_name.startswith("SittingDown"):
            continue
        if sequence_name.startswith(action):
            loaded_seqs = loaded_seqs + 1
            cdf_file = cdflib.CDF(file_name)
            poses = cdf_file.varget("Pose").squeeze()
            cdf_file.close()
            data[(subject, action, sequence_name)] = poses
            return poses
            
            
def define_actions(action):
    """Given an action string, returns a list of corresponding actions.

    # Arguments
        action: String. either "all" or one of the h36m actions
        
    # Returns
        actions: List of strings. Actions to use.
    
    """
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all":
        return actions
    if not action in actions:
        raise ValueError("Unrecognized action: %s" % action)
    return [action]
    
def normalization_stats(complete_data, dim, predict_14=False):
    """Computes normalization statistics: mean and stdev, dimensions used and ignored

    # Arguments
        complete_data: nxd np array with poses
        dim. integer={2,3} dimensionality of the data
        predict_14. boolean. Whether to use only 14 joints
        
    # Returns
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dimensions_to_ignore: list of dimensions not used in the model
        dimensions_to_use: list of dimensions used in the model
    """

    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)
    dimensions_to_ignore = []
    dimensions_to_use, dimensions_to_ignore = get_dimensions_to_use_ignore(dim)
    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def get_dimensions_to_use_ignore(dim):
    dimensions_to_use = np.array(
        [joint != '' and joint != 'Neck/Nose' for joint in H36M_NAMES])
    if dim == 2:
        dimensions_to_use, dimensions_to_ignore = normalization_stats_2D(
            dimensions_to_use)
    else:
        dimensions_to_use, dimensions_to_ignore = normalization_stats_3D(
            dimensions_to_use, predict_14=False)
    return dimensions_to_use, dimensions_to_ignore


def normalization_stats_2D(dimensions_to_use):
    """Computes 2D normalization statistics: mean and stdev, dimensions used and ignored

   # Arguments
       dim. integer={2,3} dimensionality of the data

   # Returns
       dimensions_to_ignore: list of dimensions not used in the model
       dimensions_to_use: list of dimensions used in the model
   """
    print('########################################################3')
    print(dimensions_to_use)
    dimensions_to_use = np.where(dimensions_to_use)[0]
    dimensions_to_use = np.sort(
        np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))
    dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 2),
                                     dimensions_to_use)
    return dimensions_to_use, dimensions_to_ignore


def normalization_stats_3D(dimensions_to_use, predict_14=False):
    """Computes 3D normalization statistics: mean and stdev, dimensions used and ignored

    # Arguments        
        dim. integer={2,3} dimensionality of the data
        
    # Returns
        dimensions_to_ignore: list of dimensions not used in the model
        dimensions_to_use: list of dimensions used in the model
    """
    print('########################################################3DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')
    print(dimensions_to_use)
    dimensions_to_use = np.array([joint != '' for joint in H36M_NAMES])
    dimensions_to_use = np.where(dimensions_to_use)[0]
    dimensions_to_use = np.delete(dimensions_to_use,
                                  [0, 7, 9] if predict_14 else 0)
    dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 3,
                                           dimensions_to_use * 3 + 1,
                                           dimensions_to_use * 3 + 2)))
    dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 3),
                                     dimensions_to_use)
    return dimensions_to_use, dimensions_to_ignore
