import os
import random
import shutil
import sys
from collections import namedtuple

import numpy as np
import odio_urdf
from odio_urdf import *
import pybullet as p
from pybullet_object_models import ycb_objects
import pb_robot
from pb_robot.geometry import multiply, Point, Pose, tform_from_pose
import trimesh


def apply_transform(tform, point):
    vec = np.array([[point[0], point[1], point[2], 1]]).T
    return (tform @ vec)[0:3, 0]


# Note: CoM, mass, and friction properties of the urdf will be overwritten
# before loading into PyBullet.
GraspableBody = namedtuple('GraspableBody', ['object_name',
                                             'com',
                                             'mass',
                                             'friction'])
Grasp = namedtuple('Grasp', ['graspable_body',
                             'pb_point1',
                             'pb_point2',
                             'pitch',
                             'roll',
                             'ee_relpose',
                             'force'])


def offset_pose(pose, offset):
    pos, orn = pose
    new_pos = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
    return (new_pos, orn)


class GraspSimulationClient:

    def __init__(self, graspable_body, show_pybullet, recompute_inertia=False):
        """ Support both PyBullet and Trimesh for simulations and visualizations. """
        self.shapenet_root = os.environ['SHAPENET_ROOT']

        # NOTE: this assumes that primitive data is coming from _one_ set of data,
        # and we may be pulling from more than one (or create_object_lists.py enables that)
        # either (1) enable this file for multiple primitive data source folders OR
        # disable functionality multi-primitive sources in create_object_lists.py
        # it also may be nice when creating object lists to output a small .sh
        # so we can source the environment variables easily to reduce headache
        self.primitive_root = os.environ['PRIMITIVE_ROOT']
        self.recompute_inertia = recompute_inertia

        self.pb_client_id = pb_robot.utils.connect(use_gui=True if show_pybullet else False)
        if self.pb_client_id > 5:
            print('[ERROR] Too many pybullet clients open.')
            sys.exit()

        pb_robot.utils.set_pbrobot_clientid(self.pb_client_id)
        p.setGravity(0, 0, 0, physicsClientId=self.pb_client_id)
        p.setTimeStep(1/240., physicsClientId=self.pb_client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=300,
            physicsClientId=self.pb_client_id
        )

        self.graspable_body = graspable_body
        self.urdf_directory = os.path.join(self.shapenet_root, 'tmp_urdfs')
        if not os.path.exists(self.urdf_directory):
            os.mkdir(self.urdf_directory)

        urdf_path = self._get_object_urdf(graspable_body)
        if recompute_inertia:
            self.body_id = p.loadURDF(
                urdf_path,
                physicsClientId=self.pb_client_id
            )
        else:
            self.body_id = p.loadURDF(
                urdf_path,
                physicsClientId=self.pb_client_id,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )
        self.mesh, self.mesh_tform, self.mesh_fname = self._load_mesh()

        self.rightVisualShapeId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[1, 0, 0, 1],
            radius=0.005,
            physicsClientId=self.pb_client_id
        )
        self.leftVisualShapeId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[0, 1, 0, 1],
            radius=0.005,
            physicsClientId=self.pb_client_id
        )

        self.RED = [255, 0, 0, 255]
        self.show_pybullet = show_pybullet

        self.hand, self.hand_control = self._load_hand()

    def _get_object_urdf(self, graspable_body):
        """ First copy the YCB Object to a new folder then modify its URDF to include
        the specified intrinsic properties.
        """
        object_dataset, object_name = graspable_body.object_name.split('::')
        if object_dataset.lower() == 'shapenet':
            src_path = os.path.join(self.shapenet_root, 'urdfs', f'{object_name}.urdf')
            dst_object_name = '%s_%.3fm_%.3ff_%.3fcx_%.3fcy_%.3fcz.urdf' % (
                graspable_body.object_name,
                graspable_body.mass,
                graspable_body.friction,
                graspable_body.com[0],
                graspable_body.com[1],
                graspable_body.com[2]
            )
            urdf_path = os.path.join(self.urdf_directory, dst_object_name)

        elif object_dataset.lower() == 'primitive':
            src_path = os.path.join(self.primitive_root, 'urdfs', f'{object_name}.urdf')
            dst_object_name = '%s_%.3fm_%.3ff_%.3fcx_%.3fcy_%.3fcz.urdf' % (
                graspable_body.object_name,
                graspable_body.mass,
                graspable_body.friction,
                graspable_body.com[0],
                graspable_body.com[1],
                graspable_body.com[2]
            )
            urdf_path = os.path.join(self.urdf_directory, dst_object_name)

        # Only create a tmp urdf the first time seeing an object.        
        if os.path.exists(urdf_path):
            return urdf_path
        
        shutil.copy(src_path, urdf_path)

        robot = self._parse_urdf_to_odio_tree(urdf_path)

        contact = odio_urdf.Contact(
            odio_urdf.Friction_anchor(),
            odio_urdf.Lateral_friction(value=graspable_body.friction),
            odio_urdf.Rolling_friction(0.),
            odio_urdf.Spinning_friction(0.005),  # 0.005
        )

        remove_ixs = []
        for ex, element in enumerate(robot[0]):
            if isinstance(element, odio_urdf.Inertial):
                for subelement in element:
                    if isinstance(subelement, odio_urdf.Origin):
                        origin = subelement
                remove_ixs.append(ex)
            elif isinstance(element, odio_urdf.Contact):
                remove_ixs.append(ex)

        origin.xyz = '%f %f %f' % graspable_body.com
        # if not self.recompute_inertia:
        #     origin.rpy = '0.707 0.707 0'
        I = 0.001
        inertial = odio_urdf.Inertial(
            odio_urdf.Mass(value=graspable_body.mass),
            origin,
            # odio_urdf.Inertia(ixx=1e-3, iyy=1e-3, izz=1e-3, ixy=0, ixz=0, iyz=0)
            odio_urdf.Inertia(ixx=I, iyy=I, izz=I, ixy=0, ixz=0, iyz=0)
        )

        for ex in sorted(remove_ixs, reverse=True):
            del robot[0][ex]

        robot[0].append(contact)
        robot[0].append(inertial)
        with open(urdf_path, 'w') as handle:
            handle.write(robot.urdf())

        return urdf_path

    def _parse_urdf_to_odio_tree(self, urdf_path):
        with open(urdf_path, 'r') as handle:
            urdf = odio_urdf.urdf_to_odio(handle.read())
        locals_dict = {}
        exec('robot = ' + urdf[1:], globals(), locals_dict)
        robot = locals_dict['robot']
        assert len(robot) == 1  # Right now we only support single link objects.
        return robot

    def _load_hand(self):
        pb_robot.utils.set_pbrobot_clientid(self.pb_client_id)
        hand = pb_robot.panda.PandaHand()
        init_pos, init_orn = [0.1, -0.115, 0.5], [0, 0, 0, 1]
        hand_control = pb_robot.panda_controls.FloatingHandControl(
            hand, init_pos, init_orn, client_id=self.pb_client_id
        )
        hand_control.open()

        p.changeDynamics(hand.id, 0,
                         lateralFriction=1,
                         contactStiffness=30000,
                         contactDamping=1000,
                         spinningFriction=0.01,
                         frictionAnchor=1,
                         physicsClientId=self.pb_client_id)
        p.changeDynamics(hand.id, 1,
                         lateralFriction=1,
                         contactStiffness=30000,
                         contactDamping=1000,
                         spinningFriction=0.01,
                         frictionAnchor=1,
                         physicsClientId=self.pb_client_id)  # spinning_friction 0.01

        return hand, hand_control

    def _load_mesh(self):
        visual_data = p.getVisualShapeData(self.body_id, physicsClientId=self.pb_client_id)[0]

        scale = visual_data[3][0]
        mesh_fname = visual_data[4]
        dataset, object_id = self.graspable_body.object_name.split('::')
        if 'YCB' == dataset:
            pb_mesh = pb_robot.meshes.read_obj(mesh_fname, decompose=False)
            t_mesh = trimesh.Trimesh(pb_mesh.vertices,
                                     pb_mesh.faces,
                                     face_colors=[[150, 150, 150, 150]] * len(pb_mesh.faces))
            t_mesh = t_mesh.apply_scale(scale)
        elif 'ShapeNet' == dataset:
            object_id = object_id.split('_')[-1]
            mesh_fname = os.path.join(self.shapenet_root,
                                      'visual_models',
                                      f'{object_id}_centered.obj')
            pb_mesh = pb_robot.meshes.read_obj(mesh_fname, decompose=False)
            t_mesh = trimesh.Trimesh(pb_mesh.vertices,
                                     pb_mesh.faces,
                                     face_colors=[[150, 150, 150, 150]] * len(pb_mesh.faces))
            t_mesh = t_mesh.apply_scale(scale)
        elif 'Primitive' == dataset:
            # the only way to do this is to parse the urdf to get the params
            _, object_id = self.graspable_body.object_name.split('::')
            urdf_tree = self._parse_urdf_to_odio_tree(
                os.path.join(self.primitive_root, 'urdfs', object_id + '.urdf'))

            for elt in urdf_tree[0]:
                # urdfs may be organized differently, so only explore if it is a visual elt
                if type(elt) == odio_urdf.Visual:
                    # then with primitives, geometry tags have very similar structure, so we
                    # can dive right in
                    primitive_geom = elt[0][0]

                    assert type(primitive_geom) == odio_urdf.Box or \
                           type(primitive_geom) == odio_urdf.Cylinder or \
                           type(primitive_geom) == odio_urdf.Sphere

                    # use the corresponding trimesh primitive corresponding to the one
                    # specified by the urdf
                    if type(primitive_geom) == odio_urdf.Box:
                        size = [float(s) for s in primitive_geom.size.split()]
                        t_mesh = trimesh.primitives.Box(extents=size)

                    elif type(primitive_geom) == odio_urdf.Cylinder:
                        radius = float(primitive_geom.radius)
                        length = float(primitive_geom.length)
                        t_mesh = trimesh.primitives.Cylinder(radius=radius, height=length)

                    elif type(primitive_geom) == odio_urdf.Sphere:
                        radius = float(primitive_geom.radius)
                        t_mesh = trimesh.primitives.Sphere(radius=radius)

        t_mesh.fix_normals()
        # import IPython
        # IPython.embed()

        mesh_pos = visual_data[5]
        mesh_orn = visual_data[6]
        mesh_tform = pb_robot.geometry.tform_from_pose((mesh_pos, mesh_orn))

        return t_mesh, mesh_tform, mesh_fname

    def pb_draw_contacts(self, pb_point1, pb_point2):
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=self.rightVisualShapeId,
                          basePosition=pb_point1,
                          useMaximalCoordinates=True,
                          physicsClientId=self.pb_client_id)
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=self.leftVisualShapeId,
                          basePosition=pb_point2,
                          useMaximalCoordinates=True,
                          physicsClientId=self.pb_client_id)

    def pb_check_grasp_collision(self, grasp_pose, distance):
        self.hand.set_base_link_pose(grasp_pose)
        # self.hand.set_base_link_pose(pb_robot.geometry.invert(grasp_pose))
        result = p.getClosestPoints(bodyA=self.hand.id,
                                    bodyB=self.body_id,
                                    distance=0,
                                    physicsClientId=self.pb_client_id)
        collision = len(result) != 0

        if not collision:
            half_distance = (distance + 0.08)/2.
            self.hand.MoveTo(half_distance)
            result = p.getClosestPoints(bodyA=self.hand.id,
                                        bodyB=self.body_id,
                                        distance=0,
                                        physicsClientId=self.pb_client_id)
            collision = len(result) != 0
            self.hand.Open()
        # print(result)
        # if len(result) != 0:
        #     pb_robot.viz.draw_point(result[0][5])
        #     pb_robot.viz.draw_point(result[0][6])

        # pb_robot.viz.remove_all_debug()
        return collision

    def pb_get_pose(self):
        return p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.pb_client_id
        )

    def pb_set_gravity(self, g):
        p.setGravity(g[0], g[1], g[2], physicsClientId=self.pb_client_id)

    def tm_show_grasp(self, grasp):
        grasp_arrows = self._get_trimesh_grasp_viz(grasp, [0, 255, 0, 255])
        tm_point1, tm_point2 = self._get_tm_grasp_points(grasp)

        axes = self._get_tm_com_axis()
        point_cloud = trimesh.points.PointCloud([tm_point1, tm_point2], [self.RED, self.RED])
        scene = trimesh.scene.Scene([self.mesh, point_cloud, axes] + grasp_arrows)
        scene.show()

    def _get_tm_com_axis(self):
        # TODO: Align this with visual axis.
        com_pose = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.pb_client_id)
        com_mat = pb_robot.geometry.tform_from_pose(com_pose)
        return trimesh.creation.axis(transform=com_mat, origin_size=0.005, axis_length=0.05)

    def _get_tm_grasp_points(self, grasp):
        object2mesh = np.linalg.inv(self.mesh_tform)
        tm_point1 = apply_transform(object2mesh, grasp.pb_point1)
        tm_point2 = apply_transform(object2mesh, grasp.pb_point2)
        return tm_point1, tm_point2

    def _get_trimesh_grasp_viz(self, grasp, color):
        tm_point1, tm_point2 = self._get_tm_grasp_points(grasp)
        dist = np.linalg.norm(tm_point1 - tm_point2)

        object2mesh = np.linalg.inv(self.mesh_tform)
        grasp_left = multiply(grasp.ee_relpose, Pose(Point(y=-dist / 2)))[0]
        grasp_right = multiply(grasp.ee_relpose, Pose(Point(y=dist / 2)))[0]
        grasp_left = apply_transform(object2mesh, grasp_left)
        grasp_right = apply_transform(object2mesh, grasp_right)

        grasp_arrow = trimesh.load_path([grasp_left, grasp_right], colors=[color])
        if np.linalg.norm(grasp_left - tm_point1) < np.linalg.norm(grasp_left - tm_point2):
            left_arrow = trimesh.load_path([grasp_left, tm_point1], colors=[color])
            right_arrow = trimesh.load_path([grasp_right, tm_point2], colors=[color])
        else:
            left_arrow = trimesh.load_path([grasp_left, tm_point2], colors=[color])
            right_arrow = trimesh.load_path([grasp_right, tm_point1], colors=[color])
        return [grasp_arrow, left_arrow, right_arrow]

    def _get_trimesh_grasp_array_viz(self, grasp, color):
        pb_point1, pb_point2, ee_pose = grasp[0], grasp[1], grasp[2]
        object2mesh = np.linalg.inv(self.mesh_tform)
        tm_point1 = apply_transform(object2mesh, pb_point1)
        tm_point2 = apply_transform(object2mesh, pb_point2)
        tm_eepoint = apply_transform(object2mesh, ee_pose)

        left_arrow = trimesh.load_path([tm_eepoint, tm_point1], colors=[color])
        right_arrow = trimesh.load_path([tm_eepoint, tm_point2], colors=[color])
        return [left_arrow, right_arrow]

    def disconnect(self):
        p.disconnect(self.pb_client_id)

    def tm_show_grasps(self, grasps, labels=None, fname='', acquired=[]):
        grasp_arrows = []
        for gx, g in enumerate(grasps):
            if labels is not None:
                color = [0, 255, 0, 255] if labels[gx] == 1 else [255, 0, 0, 255]
            else:
                color = [0, 0, 255, 255]

            if isinstance(g, Grasp):
                grasp_arrows += self._get_trimesh_grasp_viz(g, color)
            elif g.shape[0] == 3:
                grasp_arrows += [trimesh.points.PointCloud([g], colors=[color])]
            else:
                grasp_arrows += self._get_trimesh_grasp_array_viz(g, color)
        axis = self._get_tm_com_axis()
        self.mesh.visual.face_colors[:,-1] = 100

        if len(acquired) > 0:
            acquired_points = [trimesh.points.PointCloud(acquired, colors=[[0, 0, 255, 255]]*len(acquired))]
        else:
            acquired_points = []
        scene = trimesh.scene.Scene([self.mesh, axis] + grasp_arrows + acquired_points)
        if len(fname) > 0:
            for angles, name in zip([(0, 0, 0), (np.pi / 2, 0, 0), (np.pi / 2, 0, np.pi / 2)],
                                    ['z', 'y', 'x']):
                scene.set_camera(angles=angles, distance=0.6, center=self.mesh.centroid)
                with open(fname.replace('.png', '_%s.png' % name), 'wb') as handle:
                    handle.write(scene.save_image())
        else:
            scene.set_camera(angles=(np.pi / 2, 0, np.pi / 4), distance=0.5, center=self.mesh.centroid)
            scene.show()

    def tm_get_aabb(self, pose):
        tform = pb_robot.geometry.tform_from_pose(pose)
        return self.mesh.apply_transform(tform).bounds


class GraspStabilityChecker:
    """
    Check stability of a grasp by apply various forces to an object and comparing
    relative poses of the object.

    Stability checking will happen by applying perturbations in different directions
    and ensuring the pose does not change beyond a specified threshold. Two modes are
    currently supported:
    :stability_direction='gravity': Only apply gravity is the -z direction. Note this
        is unrealistic unless the object's initial pose is accurate.
    :stability_direction='all': Apply a perturbation in multiple directions.
    :TODO: Eventually, we might want to perform a shaking motion as in the Acronym
        dataset because this will be more realistic for collecting data on a real
        robot (we can't change gravity in the real world).

    We also support two types of stability labels:
    :label_type='contact': True if the object remains in the object's gripper.
    'label_type='relpose': True if the object's relative pose with the gripper does
        not change during the motion.
    """

    def __init__(self, graspable_body, stability_direction='all', label_type='relpose', grasp_noise=0.0,
                 show_pybullet=False, recompute_inertia=False):
        assert stability_direction in ['all', 'gravity']
        assert label_type in ['relpose', 'contact']
        self.stability_direction = stability_direction
        self.label_type = label_type
        self.grasp_noise = grasp_noise

        self.sim_client = GraspSimulationClient(
            graspable_body,
            show_pybullet=show_pybullet,
            recompute_inertia=recompute_inertia
        )
        self.reset_pose = p.getBasePositionAndOrientation(
            self.sim_client.body_id,
            physicsClientId=self.sim_client.pb_client_id
        )

        self.show_pybullet = show_pybullet

    def get_noisy_grasp(self, grasp):
        pos, orn = grasp.ee_relpose
        new_pos = np.array(pos) + np.random.randn(3) * self.grasp_noise
        new_grasp = Grasp(graspable_body=grasp.graspable_body,
                          pb_point1=grasp.pb_point1,
                          pb_point2=grasp.pb_point2,
                          pitch=grasp.pitch,
                          roll=grasp.roll,
                          ee_relpose=(new_pos, orn),
                          force=grasp.force)

        return new_grasp

    def _reset(self):
        p.resetBasePositionAndOrientation(
            self.sim_client.body_id,
            self.reset_pose[0],
            self.reset_pose[1],
            physicsClientId=self.sim_client.pb_client_id
        )
        p.setGravity(0, 0, 0, physicsClientId=self.sim_client.pb_client_id)
        pb_robot.utils.set_pbrobot_clientid(self.sim_client.pb_client_id)
        self.sim_client.hand.Open()

    def show_contact_points(self):
        results = p.getContactPoints(
            bodyA=self.sim_client.hand.id,
            bodyB=self.sim_client.body_id,
            physicsClientId=self.sim_client.pb_client_id
        )
        pb_robot.viz.remove_all_debug()
        for rx, result in enumerate(results):
            point1 = result[5]
            point2 = result[6]
            normalDir = result[7]
            end = np.array(point2) + np.array(normalDir) * 0.02

            p.addUserDebugLine(point2, end, lineColorRGB=[1, 0, 0], lineWidth=0.02,
                               lifeTime=0,
                               physicsClientId=self.sim_client.pb_client_id)

    def get_label(self, grasp):
        self._reset()
        # input('Reset...')

        grasp = self.get_noisy_grasp(grasp)

        stable = True
        # gravity_vectors = np.concatenate([
        #     self._get_gravity_vectors_inplane(grasp, 20),
        #     self._get_gravity_vectors(10)
        # ])
        gravity_vectors = self._get_gravity_vectors_inplane(grasp, 20)
        if self.show_pybullet:
            self.pb_draw_gravity(gravity_vectors)
        for gx in range(gravity_vectors.shape[0]):
            self._reset()
            self.sim_client.hand_control.set_pose(grasp.ee_relpose[0], grasp.ee_relpose[1])
            self.sim_client.hand_control.close(force=grasp.force, wait=self.show_pybullet)

            init_pose = self.sim_client.pb_get_pose()
            self.sim_client.pb_set_gravity(gravity_vectors[gx, :])
            for tx in range(100):
                if self.show_pybullet and tx % 10 == 0:
                    self.show_contact_points()
                    # input('Next?')
                self.sim_client.hand_control.move_to(
                    grasp.ee_relpose[0],
                    grasp.ee_relpose[1],
                    grasp.force,
                    wait=self.show_pybullet
                )
            end_pose = self.sim_client.pb_get_pose()

            pos_diff = np.linalg.norm(np.array(end_pose[0]) - np.array(init_pose[0]))
            angle_diff = pb_robot.geometry.quat_angle_between(end_pose[1], init_pose[1])
            if pos_diff > 0.02 or angle_diff > 10:  # 0.01/5
                stable = False
                break

        # import IPython
        # IPython.embed()

        print(f'Stable: {stable}\tPos: {pos_diff}\tAngle: {angle_diff}')
        return stable

    def pb_draw_gravity(self, gravity_vectors):
        for v in gravity_vectors:
            pb_robot.viz.draw_point(v / 20.)

    def get_label_orginal(self, grasp):
        self._reset()
        # input('Reset...')

        grasp = self.get_noisy_grasp(grasp)

        self.sim_client.hand_control.set_pose(grasp.ee_relpose[0], grasp.ee_relpose[1])
        self.sim_client.hand_control.close(force=grasp.force, wait=self.show_pybullet)

        init_pose = self.sim_client.pb_get_pose()
        if self.stability_direction == 'gravity':
            self.sim_client.pb_set_gravity((0, 0, -10))

            x, y, z = grasp.ee_relpose[0]
            for ix in range(0, 500):
                self.sim_client.hand_control.move_to(
                    [x, y, z + 0.001 * ix],
                    grasp.ee_relpose[1],
                    grasp.force,
                    wait=self.show_pybullet
                )

            for _ in range(100):
                self.sim_client.hand_control.move_to(
                    [x, y, z + 0.001 * 500],
                    grasp.ee_relpose[1],
                    grasp.force,
                    wait=self.show_pybullet
                )
        else:
            gravity_vectors = self._get_gravity_vectors(10)
            for gx in range(gravity_vectors.shape[0]):
                self.sim_client.pb_set_gravity(gravity_vectors[gx, :])
                for _ in range(100):
                    self.sim_client.hand_control.move_to(
                        grasp.ee_relpose[0],
                        grasp.ee_relpose[1],
                        grasp.force,
                        wait=self.show_pybullet
                    )
        end_pose = self.sim_client.pb_get_pose()

        pos_diff = np.linalg.norm(np.array(end_pose[0]) - np.array(init_pose[0]))
        angle_diff = pb_robot.geometry.quat_angle_between(end_pose[1], init_pose[1])
        if pos_diff > 0.02 or angle_diff > 10:  # 0.01/5
            stable = False
        else:
            stable = True

        print(f'Stable: {stable}\tPos: {pos_diff}\tAngle: {angle_diff}')
        return stable

    def disconnect(self):
        self.sim_client.disconnect()

    def _get_gravity_vectors_inplane(self, grasp, n_samples):
        n_pool = n_samples * 100
        angles = np.random.uniform(0, 2 * np.pi, n_pool)
        points_x = np.cos(angles) * 10
        points_z = np.sin(angles) * 10
        points_gripper_frame = np.hstack([
            points_x[:, None],
            np.zeros((n_pool, 1)),
            points_z[:, None],
            np.ones((n_pool, 1))
        ])
        points_gripper_frame = self._k_farthest_points(points_gripper_frame, n_samples)
        # import IPython
        # IPython.embed()
        points_global_frame = tform_from_pose(grasp.ee_relpose) @ (points_gripper_frame.T)
        points_global_frame = points_global_frame.T[:, 0:3]
        return points_global_frame

    def _get_gravity_vectors(self, n_samples):
        points = []
        for _ in range(1000):
            gravity = np.random.randn(3)
            gravity = 10 * gravity / np.linalg.norm(gravity)

            points.append(gravity)

        points = np.array(points)
        points = self._k_farthest_points(points, n_samples)
        return points

    def _k_farthest_points(self, points, k):
        ixs = [0]
        min_distances = np.linalg.norm(points - points[0:1, :], axis=1)
        for _ in range(k - 1):
            # Iteratively choose the point that is farthest.
            new_ix = np.argmax(min_distances)
            ixs.append(new_ix)

            dist_to_new = np.linalg.norm(points - points[new_ix:new_ix + 1, :], axis=1)
            min_distances = np.stack([min_distances, dist_to_new], -1)
            min_distances = np.min(min_distances, axis=1)

        return points[ixs, ...]


class GraspSampler:
    """ Given a specific object, sample antipodal grasp candidates where the Panda gripper does not
    intersect with the object. A separate sampler should be created for each object.
    """

    def __init__(self, graspable_body, antipodal_tolerance=30, show_pybullet=False, urdf_directory='urdf_models'):
        """
        :param object_urdf: Used to initialize a new PyBullet instance to perform
        grasp stability checking.
        :param show_pybullet: Use the PyBullet GUI.
        """
        self.sim_client = GraspSimulationClient(graspable_body=graspable_body,
                                                show_pybullet=show_pybullet)

        self.gripper_width = 0.08
        self.antipodal_tolerance = np.deg2rad(antipodal_tolerance)
        self.show_pybullet = show_pybullet
        self.graspable_body = graspable_body

    def _sample_antipodal_points_rays(self, max_attempts=1000):
        """ Return two antipodal points in the visual frame (which may not be the object frame).
        """
        points = []
        while True:
        # for _ in range(max_attempts):
            [point1], [index1] = self.sim_client.mesh.sample(1, return_index=True)
            normal1 = np.array(self.sim_client.mesh.face_normals[index1, :])

            # TODO: Add perturbation to ray according to antipodal tolerance.

            intersections = self.sim_client.mesh.ray.intersects_location([point1], [-normal1])
            hit_points, _, hit_faces = intersections

            if hit_points.shape[0] <= 1:
                # Only found the origin as intersection point.
                continue

            point_intersect, index_intersect = hit_points[1], hit_faces[1]
            face_weight = np.zeros((len(self.sim_client.mesh.faces)))
            face_weight[index_intersect] = 1.

            for _ in range(10):
                [point2], [index2] = self.sim_client.mesh.sample(
                    1, return_index=True, face_weight=face_weight
                )

                # print(point1, point2, index1, index2)
                normal2 = np.array(self.sim_client.mesh.face_normals[index2, :])
                # print('Normals:', normal1, normal2)
                distance = pb_robot.geometry.get_distance(point1, point2)
                # if distance > 0.01: continue
                if distance > self.gripper_width or distance < 1e-3:
                    continue

                direction = point2 - point1
                # print('Direction:', direction)
                # num = np.dot(normal1, -direction)
                # den = pb_robot.geometry.get_length(normal1)*pb_robot.geometry.get_length(-direction)
                # arg = num/den
                # print('ACOS:', num, den, arg)
                # res = np.math.acos(arg)

                error1 = pb_robot.geometry.angle_between(normal1, -direction)
                error2 = pb_robot.geometry.angle_between(normal2, direction)

                # For anitpodal grasps, the angle between the normal and
                # direction vector should be small.
                if (error1 > self.antipodal_tolerance) or (error2 > self.antipodal_tolerance):
                    continue

                return point1, point2, (error1, error2)
        return None, None

    def _sample_antipodal_points(self):
        """ Return two antipodal points in the visual frame (which may not be the object frame).
        """
        points = []
        count = 0
        while True:

            [point1], [index1] = self.sim_client.mesh.sample(1, return_index=True)
            for ax in range(200):
                [point2], [index2] = self.sim_client.mesh.sample(1, return_index=True)

                distance = pb_robot.geometry.get_distance(point1, point2)
                if distance > self.gripper_width or distance < 1e-3:
                    continue

                direction = point2 - point1
                normal1 = np.array(self.sim_client.mesh.face_normals[index1, :])
                normal2 = np.array(self.sim_client.mesh.face_normals[index2, :])
                # Make sure normals are pointing away from each other.
                # if normal1.dot(-direction) < 0:
                #     normal1 *= -1
                # if normal2.dot(direction) < 0:
                #     normal2 *= -1
                # if ax == 0:
                #     import IPython
                #     IPython.embed()
                error1 = pb_robot.geometry.angle_between(normal1, -direction)
                error2 = pb_robot.geometry.angle_between(normal2, direction)

                # For anitpodal grasps, the angle between the normal and
                # direction vector should be small.
                if (error1 > self.antipodal_tolerance) or (error2 > self.antipodal_tolerance):
                    continue

                return point1, point2

            # points.extend([point1, point2])
            # count += 1
            # print(count, self.sim_client.mesh.is_watertight)
            # if count >= 1000:
            #     break

        pc = trimesh.points.PointCloud(points)
        scene = trimesh.scene.Scene([pc, self.sim_client.mesh])
        scene.show()

    def sample_grasp(self, force, show_trimesh=False, max_attempts=100):

        while True:
            tm_point1, tm_point2, antipodal_angle = self._sample_antipodal_points_rays()
            if tm_point1 is None:
                return None
            # The visual frame of reference might be different from the object's link frame.
            pb_point1 = apply_transform(self.sim_client.mesh_tform, tm_point1)
            pb_point2 = apply_transform(self.sim_client.mesh_tform, tm_point2)
            # self.sim_client.pb_draw_contacts(pb_point1, pb_point2)
            for _ in range(10):
                # Pitch is the angle of the grasp while roll controls the
                # orientation (flipped gripper or not).
                pitch = random.uniform(-np.pi, np.pi)
                # pitch = random.choice([-np.pi, np.pi])
                # roll = random.choice([0, np.pi])  # Not used. Should be covered by point-ordering.
                roll = 0
                grasp_point = (pb_point1 + pb_point2) / 2

                # The contact points define a plane (contact plane).
                normal = (pb_point2 - pb_point1) / pb_robot.geometry.get_length(pb_point2 - pb_point1)
                origin = np.zeros(3)

                # Calculate the transform that brings the XY-plane to the contact plane.
                tform = np.linalg.inv(trimesh.points.plane_transform(origin, -normal))
                quat1 = pb_robot.geometry.quat_from_matrix(tform[0:3, 0:3])
                # z-axis in direction of point2 to point1
                pose1 = pb_robot.geometry.Pose(
                    origin,
                    euler=pb_robot.geometry.euler_from_quat(quat1)
                )

                # Project (0, 0, 1) to the contact plane.
                point = np.array((0, 0, 1))
                distance = np.dot(normal, point - np.array(origin))
                projection_world = point - distance * normal

                # This gives us the (x, y) coordinates of the projected
                # point in contact-plane coordinates.
                projection = pb_robot.geometry.tform_point(
                    pb_robot.geometry.invert(pose1),
                    projection_world
                )  # Do we need to invert pose1?
                yaw = np.math.atan2(projection[1], projection[0])
                quat2 = pb_robot.geometry.multiply_quats(
                    quat1,
                    pb_robot.geometry.quat_from_euler(
                        pb_robot.geometry.Euler(yaw=yaw)
                    )
                )

                grasp_quat = pb_robot.geometry.multiply_quats(
                    quat2,
                    pb_robot.geometry.quat_from_euler(
                        pb_robot.geometry.Euler(roll=np.pi / 2)
                    ),
                    pb_robot.geometry.quat_from_euler(
                        pb_robot.geometry.Euler(pitch=pitch)
                    ),  # TODO: local pitch or world pitch?
                    # pb_robot.geometry.quat_from_euler(
                    #     pb_robot.geometry.Euler(roll=roll)
                    # ),  # Switches fingers
                )

                # pose2 = pb_robot.geometry.Pose(
                #     grasp_point,
                #     euler=pb_robot.geometry.euler_from_quat(grasp_quat)
                # )
                # pb_robot.viz.draw_pose(pose2)
                finger_length = 0.1034
                grasp_pose = pb_robot.geometry.Pose(
                    grasp_point,
                    pb_robot.geometry.euler_from_quat(grasp_quat)
                )
                grasp_pose = pb_robot.geometry.multiply(
                    grasp_pose,
                    pb_robot.geometry.Pose(pb_robot.geometry.Point(z=-finger_length))
                )  # FINGER_LENGTH

                collision = self.sim_client.pb_check_grasp_collision(
                    grasp_pose=grasp_pose, 
                    distance=pb_robot.geometry.get_length(pb_point2 - pb_point1)
                )
                # import time
                # if collision:
                #     input('Collision, continue?')
                #     pb_robot.viz.remove_all_debug()

                if not collision:
                    grasp = Grasp(graspable_body=self.graspable_body,
                                  pb_point1=pb_point1,
                                  pb_point2=pb_point2,
                                  pitch=pitch,
                                  roll=roll,
                                  force=force,
                                  ee_relpose=grasp_pose)

                    if self.show_pybullet:
                        self.sim_client.pb_draw_contacts(pb_point1, pb_point2)

                    if show_trimesh:
                        self.sim_client.tm_show_grasp(grasp)
                    # print('Angle:', np.rad2deg(np.max(antipodal_angle)))
                    return grasp

    def disconnect(self):
        self.sim_client.disconnect()


class GraspableBodySampler:
    """
    Given an object name, sample random intrinsic properties for that object.

    COM is sampled in % of containing bounding box and rejection sampling is
    used to make sure it lies within the convex-hull of the mesh.
    """
    MASS_RANGE = (0.1, 1.0)
    FRICTION_RANGE = (0.1, 1.0)

    @staticmethod
    def sample_random_object_properties(object_name, mass=None, friction=None, com=None):
        """
        Returns GraspableBody. Only sample parameteres if they are not specified already.
        """
        if mass is None:
            mass = np.random.uniform(*GraspableBodySampler.MASS_RANGE)
        if friction is None:
            friction = np.random.uniform(*GraspableBodySampler.FRICTION_RANGE)
        if com is None:
            com = GraspableBodySampler._sample_com(object_name)
        return GraspableBody(object_name, tuple(com), mass, friction)

    @staticmethod
    def _sample_com(object_name):
        """
        Load a simulation client.
        """
        # TODO: Need to convert CoM to object
        tmp_body = GraspableBody(object_name, (0, 0, 0), 0, 1.0)
        sim_client = GraspSimulationClient(
            tmp_body,
            show_pybullet=False,
        )
        print(f'Object watertight:', sim_client.mesh.is_watertight)
        # if not sim_client.mesh.is_watertight:
        #     import IPython
        #     IPython.embed()
        aabb_min, aabb_max = sim_client.mesh.bounding_box.bounds

        while True:
            x = np.random.uniform(aabb_min[0], aabb_max[0])
            y = np.random.uniform(aabb_min[1], aabb_max[1])
            z = np.random.uniform(aabb_min[2], aabb_max[2])
            com = (x, y, z)
            if sim_client.mesh.convex_hull.contains(np.array([com])):
                break

        com = apply_transform(sim_client.mesh_tform, com).tolist()

        pb_robot.viz.draw_aabb((aabb_min, aabb_max))
        pb_robot.viz.draw_point(com)

        sim_client.disconnect()

        return com


def main_serial():
    # objects_names = [name for name in os.listdir(ycb_objects.getDataPath()) if 'Ycb' in name]
    # objects_names = ['YCB::YcbCrackerBox']
    # objects_names = ['ShapeNet::Desk_fe2a9f23035580ce239883c5795189ed']
    # objects_names = ['ShapeNet::ComputerMouse_379e93edfd0cb9e4cc034c03c3eb69d']
    # objects_names = ['ShapeNet::Chair_198a3e82b102529c4904d89e9169817b']
    # objects_names = ['ShapeNet::Barstool_55e7dc1021e15181a495c196d4f0cebb']
    # objects_names = ['ShapeNet::Dresser_e9e3f04bce3933a2c62986712894256b']
    # objects_names = ['ShapeNet::MilkCarton_64018b545e9088303dd0d6160c4dfd18']
    objects_names = ['ShapeNet::WineGlass_2d89d2b3b6749a9d99fbba385cc0d41d']
    # objects_names = ['ShapeNet::WallLamp_8be32f90153eb7281b30b67ce787d4d3']
    # objects_names = ['ShapeNet::USBStick_d1d5e86583e0e5f950648c342f01b361']
    # objects_names = ['ShapeNet::TV_1a595fd7e7043a06b0d7b0d4230df8ca']
    # objects_names = ['ShapeNet::Sideboard_12f1e4964078850cc7113d9e058b9db7']
    # objects_names = ['ShapeNet::Couch_1ed2e9056a9e94a693e60794f9200b7']
    # objects_names = ['ShapeNet::WallUnit_a642e3f84392ebacc9dd845c88786daa']
    # objects_names = ['Primitive::Cylinder_1637429346034696960']
    # objects_names = ['ShapeNet::Armoire_1857891cdabf6824b0fd397b9007d287']
    # objects_names = ['ShapeNet::TableLamp_366b8a4c8e38961597910e2efc6028df']
    objects_names = ['ShapeNet::Helicopter_3b4cbd4fd5f6819bea4732296ea50647']
    # objects_names =['ShapeNet::Lamp_de623f89d4003d189a08db804545b684']
    # objects_names = ['ShapeNet::StandingClock_cca5dddf86affe9a23522985f649a9ae']
    # objects_names = ['Primitive::Box_970355850778123776']
    objects_names = ['ShapeNet::DrinkBottle_c0b6be0fb83b237f504721639e19f609']
    objects_names = ['ShapeNet::FloorLamp_9f47557e429f4734c2d56e1f87f4dc83']

    objects_names = ['Primitive::Box_553711050557875456']

    object_name = random.choice(objects_names)
    # graspable_body = GraspableBody(
    #     object_name=object_name,
    #     com=(0.00400301, 0.01275706, 0.02090709),
    #     mass=0.52519492, friction=0.57660351
    # )

    # graspable_body = GraspableBody(
    #     object_name=object_name,
    #     com=(-0.03690317,  0.00957142,  0.00425203),
    #     mass=0.418, friction=0.141
    # )  # m=0.93
    graspable_body = GraspableBody(
        object_name=object_name,
        # com=(-0.1386, 0.0019, -0.0419),
        com=(0., 0., -0.05),
        mass=1.0, friction=1.0
    )  # mass=0.908
    # graspable_body = GraspableBody(
    #     object_name=object_name,
    #     com=(-0.105, 0.0225, -0.0251),
    #     mass=0.282, friction=0.614
    # )  # mass=0.908
    # graspable_body = GraspableBody(
    #     object_name=object_name,
    #     com=(0.0327, 0.0097, -0.0036),
    #     mass=0.693, friction=0.227
    # )  # mass=0.908
    # graspable_body = graspablebody_from_vector(
    #     object_name,
    #     [-0.02043926, -0.05384533, -0.00729232,  0.74735462,  0.84808624]
    # )
    # graspable_body = GraspableBody(
    #     object_name=object_name,
    #     com=(-0.0566, 0.0100, 0.020),
    #     mass=0.154, friction=0.534
    # )  # mass=0.908
    # graspable_body = GraspableBodySampler.sample_random_object_properties(object_name)
    # graspable_body = graspablebody_from_vector(
    #     'ShapeNet::Headphones_5fbad7dea0243acd464e3094da7d844a',
    #     [-0.01376322,  0.00172974, -0.00481943,  0.48139637,  0.45036947]
    # )
    grasp_sampler = GraspSampler(
        graspable_body=graspable_body,
        antipodal_tolerance=30,
        show_pybullet=False
    )
    grasp_sampler.sim_client.mesh.show()
    n_samples = 50
    grasps = []
    for lx in range(0, n_samples):
        print('Sampling %d/%d...' % (lx, n_samples))
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasps.append(grasp)
    grasp_sampler.disconnect()
    sim_client = GraspSimulationClient(
        graspable_body,
        show_pybullet=False
    )

    # sim_client.tm_show_grasps(grasps)#, fname='test.png')
    sim_client.disconnect()

    labeler1 = GraspStabilityChecker(
        graspable_body,
        stability_direction='all',
        label_type='relpose',
        grasp_noise=0.0,
        show_pybullet=False,
        recompute_inertia=True
    )
    labeler2 = GraspStabilityChecker(
        graspable_body,
        stability_direction='all',
        label_type='relpose',
        grasp_noise=0.0,
        show_pybullet=False,
        recompute_inertia=False
    )
    labels1 = []
    labels2 = []
    for lx, grasp in enumerate(grasps):
        print('Labeling %d/%d...' % (lx, n_samples))
        labels1.append(labeler1.get_label(grasp))
        # labels2.append(labeler2.get_label(grasp))

        # l1 = labeler.get_label(grasp, show_pybullet=False)
        # l2 = labeler.get_label(grasp, show_pybullet=False)
        # print('Consistant:', l1 == l2)
    labeler1.disconnect()
    labeler2.disconnect()

    # print('Equals: ', (np.array(labels1) == np.array(labels2)).sum())
    sim_client = GraspSimulationClient(graspable_body,
                                       show_pybullet=False)
    sim_client.tm_show_grasps(grasps, labels1)
    # sim_client.tm_show_grasps(grasps, np.array(labels1) == np.array(labels2))  # , fname='test.png')
    sim_client.disconnect()


def vary_object_properties():
    # object_name = 'ShapeNet::USBStick_ab82d56cf9cc2476d154e1b098031d39'
    # graspable_body = GraspableBody(
    #     object_name=object_name,
    #     com=(0.05, -0.02, -0.008),
    #     mass=0.93,
    #     friction=0.24
    # )  # m=0.93

    object_name = 'ShapeNet::TV_1a595fd7e7043a06b0d7b0d4230df8ca'
    graspable_body = GraspableBody(
        object_name=object_name,
        com=(-0.1386, 0.0019, -0.0419),
        mass=0.908,
        friction=0.84
    )  # mass=0.908

    n_samples = 100
    grasps = []
    grasp_sampler = GraspSampler(
        graspable_body=graspable_body,
        antipodal_tolerance=30,
        show_pybullet=False
    )

    for lx in range(0, n_samples):
        print('Sampling %d/%d...' % (lx, n_samples))
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasps.append(grasp)
    grasp_sampler.disconnect()

    for m in np.linspace(0.1, 1.0, 10):
        for f in np.linspace(0.1, 1.0, 10):
            graspable_body = GraspableBody(
                object_name=object_name,
                com=(-0.1386, 0.0019, -0.0419),
                mass=m,
                friction=f
            )

            # sim_client = GraspSimulationClient(
            #     graspable_body,
            #     show_pybullet=False
            # )
            # #sim_client.tm_show_grasps(grasps)#, fname='test.png')
            # sim_client.disconnect()

            labeler = GraspStabilityChecker(
                graspable_body,
                stability_direction='all',
                label_type='relpose',
                grasp_noise=0.0,
                show_pybullet=False
            )
            labels = []
            for lx, grasp in enumerate(grasps):
                print('Labeling %d/%d...' % (lx, n_samples))
                labels.append(labeler.get_label(grasp))
            labeler.disconnect()

            # print('Equals: ', (np.array(labels1) == np.array(labels2)).sum())
            sim_client = GraspSimulationClient(
                graspable_body,
                show_pybullet=False
            )
            sim_client.tm_show_grasps(
                grasps,
                labels,
                fname=(
                        '/home/mnosew/Pictures/grasp_data_inspection/'
                        '%s_%.2fm_%.2ff_new.png' % (object_name, m, f)
                )
            )
            # sim_client.tm_show_grasps(
            #     grasps,
            #     np.array(labels1) == np.array(labels2)
            # )  #, fname='test.png')
            sim_client.disconnect()


def display_object():
    masses = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    frictions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    from mpl_toolkits.axes_grid1 import ImageGrid
    from PIL import Image

    import matplotlib.pyplot as plt
    plt.clf()
    fig = plt.figure(figsize=(40., 40.))
    grid = ImageGrid(
        fig, 111,  # similar to subplot(111)
        nrows_ncols=(len(masses), len(frictions)),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    # object_name = 'ShapeNet::USBStick_ab82d56cf9cc2476d154e1b098031d39'
    object_name = 'ShapeNet::TV_1a595fd7e7043a06b0d7b0d4230df8ca'
    for mx, mass in enumerate(masses):
        for fx, friction in enumerate(frictions):
            fname = (
                f'/home/mnosew/Pictures/grasp_data_inspection/'
                f'{object_name}_{mass:.2f}m_{friction:.2f}f_new_y.png'
            )
            with open(fname, 'rb') as handle:
                im = np.array(Image.open(handle))

            # Iterating over the grid returns the Axes.
            ax = grid[mx * len(frictions) + fx]
            ax.imshow(im)
            ax.text(50, 375, f'm={mass: .2f}\nf={friction: .2f}',
                    bbox=dict(fill=False, edgecolor='black', linewidth=1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig('/home/mnosew/Pictures/all_monitor_new.png')


def graspablebody_from_vector(object_name, vector):
    graspable_body = GraspableBody(object_name=object_name,
                                   com=tuple(vector[0:3]),
                                   mass=vector[3],
                                   friction=vector[4])
    return graspable_body


if __name__ == '__main__':
    main_serial()
    # vary_object_properties()
    # display_object()
