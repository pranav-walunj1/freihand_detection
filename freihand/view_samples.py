from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt
import argparse
import pyrender
import trimesh
from utils.fh_utils import *


def show_training_samples(base_path, version, num2show=None, render_mano=False):
    if render_mano:
        from utils.model import HandModel, recover_root, get_focal_pp, split_theta

    if num2show == -1:
        num2show = db_size('training') # show all

    # load annotations
    db_data_anno = load_db_annotation(base_path, 'training')

    # Convert the zip object to a list
    db_data_anno = list(db_data_anno)
    
    
    # iterate over all samples
    for idx in range(db_size('training')):
        if idx >= num2show:
            break

        # load image and mask
        img = read_img(idx, base_path, 'training', version)
        msk = read_msk(idx, base_path)

        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        # render an image of the shape
        msk_rendered = None
        if render_mano:
            # split mano parameters
            poses, shapes, uv_root, scale = split_theta(mano)
            focal, pp = get_focal_pp(K)
            xyz_root = recover_root(uv_root, scale, focal, pp)

            # set up the hand model and feed hand parameters
            renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
            renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
            #msk_rendered = renderer.render(K, img_shape=img.shape[:2])   ###NEED to update this way of working
            ###########
            V, F = renderer._get_verts_faces() # this gives you vertices and faces
            msk_rendered = render_with_pyrender(V, F, K, img_shape=img.shape[:2], render_mask=render_mano)
            ###########
        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img)
        ax2.imshow(msk if msk_rendered is None else msk_rendered)
        plot_hand(ax1, uv, order='uv')
        plot_hand(ax2, uv, order='uv')
        ax1.axis('off')
        ax2.axis('off')
        plt.show()

###########################
def render_with_pyrender(vertices, faces, cam_intrinsics, dist=None, M=None, img_shape=None, render_mask=False):
    if dist is None:
        dist = np.zeros(5)
    dist = dist.flatten()
    if M is None:
        M = np.eye(4)

    # get R, t from M (has to be world2cam)
    R = M[:3, :3]
    t = M[:3, 3]

    w, h = (320, 320)
    if img_shape is not None:
        w, h = img_shape[1], img_shape[0]

    pp = np.array([cam_intrinsics[0, 2], cam_intrinsics[1, 2]])  # Principal point
    f = np.array([cam_intrinsics[0, 0], cam_intrinsics[1, 1]])  # Focal length

    # Create pyrender scene
    scene = pyrender.Scene()

    # Create mesh from vertices and faces
    mesh = trimesh.Trimesh(vertices, faces)
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh)

    # Add camera with intrinsics
    camera = pyrender.IntrinsicsCamera(fx=f[0], fy=f[1], cx=pp[0], cy=pp[1])
    scene.add(camera, pose=M)

    # Add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=np.eye(4))  # Light is in default position

    # Render the scene
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene)

    if render_mask:
        # Return only the mask
        return np.ones_like(color[:, :, 0]) * 255  # Example mask (all white)

    return color  # Return rendered image
###########################




def show_eval_samples(base_path, num2show=None):
    if num2show == -1:
        num2show = db_size('evaluation') # show all

    for idx in  range(db_size('evaluation')):
        if idx >= num2show:
            break

        # load image only, because for the evaluation set there is no mask
        img = read_img(idx, base_path, 'evaluation')

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        ax1.axis('off')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--show_eval', action='store_true',
                        help='Shows samples from the evaluation split if flag is set, shows training split otherwise.')
    parser.add_argument('--mano', action='store_true',
                        help='Enables rendering of the hand if mano is available. See README for details.')
    parser.add_argument('--num2show', type=int, default=-1,
                        help='Number of samples to show. ''-1'' defaults to show all.')
    parser.add_argument('--sample_version', type=str, default=sample_version.gs,
                        help='Which sample version to use when showing the training set.'
                             ' Valid choices are %s' % sample_version.valid_options())
    args = parser.parse_args()

    # check inputs
    msg = 'Invalid choice: ''%s''. Must be in %s' % (args.sample_version, sample_version.valid_options())
    assert args.sample_version in sample_version.valid_options(), msg

    if args.show_eval:
        """ Show some evaluation samples. """
        show_eval_samples(args.base_path,
                          num2show=args.num2show)

    else:
        """ Show some training samples. """
        show_training_samples(
            args.base_path,
            args.sample_version,
            num2show=args.num2show,
            render_mano=args.mano
        )

