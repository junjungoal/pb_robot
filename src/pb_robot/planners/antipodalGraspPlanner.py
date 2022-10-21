from cProfile import label
from collections import namedtuple
import pb_robot
import trimesh
import pybullet as p
import numpy as np
import os
import random
from odio_urdf import *
import shutil
import copy
import sys

from pybullet_object_models import ycb_objects
from pb_robot import body
from pb_robot.geometry import multiply, Point, Pose, tform_from_pose


def apply_transform(tform, point):
    vec = np.array([[point[0], point[1], point[2], 1]]).T
    return (tform@vec)[0:3, 0]

# TODO: Support ShapeNet objects.
# Note: CoM, mass, and friction properties of the urdf will be overwritten before loading into PyBullet.
GraspableBody = namedtuple('GraspableBody', ['object_name', 'com', 'mass', 'friction'])
Grasp = namedtuple('Grasp', ['graspable_body', 'pb_point1', 'pb_point2', 'pitch', 'roll', 'ee_relpose', 'force'])

def offset_pose(pose, offset):
    pos, orn = pose
    new_pos = (pos[0]+offset[0], pos[1]+offset[1], pos[2]+offset[2])
    return (new_pos, orn)

class ParallelGraspSimulationClient:

    def __init__(self, graspable_bodies, show_pybullet, urdf_directory):
        """ Support both PyBullet and Trimesh for simulations and visualizations.
        """
        self.shapenet_root = '/home/seiji/Research/object_models/shapenet-sem/'

        self.pb_client_id = pb_robot.utils.connect(use_gui=True if show_pybullet else False)
        if self.pb_client_id > 5:
            print('[ERROR] Too many pybullet clients open.')
            sys.exit()

        p.setGravity(0, 0, 0, physicsClientId=self.pb_client_id)
        self.graspable_bodies = graspable_bodies
        self.urdf_directory = urdf_directory
        if not os.path.exists(urdf_directory):
            os.mkdir(urdf_directory)

        self.offsets = []

        self.body_ids = []
        self.hands = []
        for gx, graspable_body in enumerate(graspable_bodies):
            urdf_path = self._get_object_urdf(graspable_body)
            self.body_ids.append(p.loadURDF(urdf_path, physicsClientId=self.pb_client_id))

            row_ix, col_ix = gx // 10, gx % 10
            self.offsets.append((row_ix, col_ix, 0.))

            pose = p.getBasePositionAndOrientation(self.body_ids[-1], physicsClientId=self.pb_client_id)
            new_pose = offset_pose(pose, self.offsets[-1])
            p.resetBasePositionAndOrientation(self.body_ids[-1], new_pose[0], new_pose[1], physicsClientId=self.pb_client_id)

            hand = self._load_hand()
            self.hands.append(hand)

        init_pos, init_orn = [0.1, -0.115, 0.5], [0, 0, 0, 1]
        self.hand_control = pb_robot.panda_controls.ParallelFloatingHandControl(self.hands, self.offsets, init_pos, init_orn, client_id=self.pb_client_id)
        self.hand_control.open()

        self.mesh, self.mesh_tform, self.mesh_fname = self._load_mesh()

        self.rightVisualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=0.005, physicsClientId=self.pb_client_id)
        self.leftVisualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0, 1, 0, 1], radius=0.005, physicsClientId=self.pb_client_id)

        self.RED = [255, 0, 0, 255]
        self.show_pybullet = show_pybullet

    def _get_object_urdf(self, graspable_body):
        """ First copy the YCB Object to a new folder then modify its URDF to include 
        the specified intrinsic properties.

        TODO: Right now we only support loading ycb_objects.
        """
        object_dataset, object_name = graspable_body.object_name.split('::')
        if object_dataset.lower() == 'ycb':
            # Copy all files for the object to a temporary location.
            src_path = os.path.join(ycb_objects.getDataPath(), object_name)
            dst_object_name = '%s_%.2fm_%.2ff_%.2fcx_%.2fcy_%.2fcz' % (graspable_body.object_name, graspable_body.mass, graspable_body.friction, graspable_body.com[0], graspable_body.com[1], graspable_body.com[2])

            dst_path = os.path.join(self.urdf_directory, dst_object_name)
            if not os.path.exists(dst_path):
                shutil.copytree(src=src_path, dst=dst_path)

            # Update the URDF parameters.
            urdf_path = os.path.join(dst_path, 'model.urdf')
        elif object_dataset.lower() == 'shapenet':
            src_path = os.path.join(self.shapenet_root, 'urdfs', f'{object_name}.urdf')
            dst_object_name = '%s_%.2fm_%.2ff_%.2fcx_%.2fcy_%.2fcz.urdf' % (graspable_body.object_name, graspable_body.mass, graspable_body.friction, graspable_body.com[0], graspable_body.com[1], graspable_body.com[2])
            dst_path = os.path.join(self.urdf_directory, dst_object_name)
            shutil.copy(src_path, dst_path)
            urdf_path = dst_path

        with open(urdf_path, 'r') as handle:
            urdf = urdf_to_odio(handle.read())

        locals_dict = {}
        exec('robot = ' + urdf[1:], globals(), locals_dict)
        robot = locals_dict['robot']

        assert(len(robot) == 1)  # Right now we only support single link objects.

        contact = Contact(
            Friction_anchor(),
            Lateral_friction(value=graspable_body.friction),
            Rolling_friction(0.),
            Spinning_friction(0.005), # 0.005
        )

        remove_ixs = []
        for ex, element in enumerate(robot[0]):
            if isinstance(element, Inertial):
                for subelement in element:
                    if isinstance(subelement, Origin):
                        origin = subelement
                remove_ixs.append(ex)
            elif isinstance(element, Contact):
                remove_ixs.append(ex)

        origin.xyz = '%f %f %f' % graspable_body.com
        inertial = Inertial(
            Mass(value=graspable_body.mass),
            origin,
            Inertia(ixx=1e-3, iyy=1e-3, izz=1e-3, ixy=0, ixz=0, iyz=0)
        )

        for ex in sorted(remove_ixs, reverse=True):
            del robot[0][ex]

        robot[0].append(contact)
        robot[0].append(inertial)

        with open(urdf_path, 'w') as handle:
            handle.write(robot.urdf())

        return urdf_path

    def _load_hand(self):
        pb_robot.utils.set_pbrobot_clientid(self.pb_client_id)
        hand = pb_robot.panda.PandaHand()

        p.changeDynamics(hand.id, 0,
            lateralFriction=1,
            contactStiffness=30000,
            contactDamping=1000,
            spinningFriction=0.01,
            frictionAnchor=1,
            physicsClientId=self.pb_client_id)
        p.changeDynamics(hand.id, 1, lateralFriction=1, contactStiffness=30000, contactDamping=1000, spinningFriction=0.01, frictionAnchor=1, physicsClientId=self.pb_client_id)

        return hand

    def _load_mesh(self):
        visual_data = p.getVisualShapeData(self.body_ids[0], physicsClientId=self.pb_client_id)[0]

        scale = visual_data[3][0]
        mesh_fname = visual_data[4]
        dataset, object_id = self.graspable_bodies[0].object_name.split('::')
        if 'ShapeNet' == dataset:
            object_id = object_id.split('_')[-1]
            mesh_fname = os.path.join(self.shapenet_root, 'visual_models', f'{object_id}_centered.obj')

        pb_mesh = pb_robot.meshes.read_obj(mesh_fname, decompose=False)
        t_mesh = trimesh.Trimesh(pb_mesh.vertices, pb_mesh.faces, face_colors=[[150, 150, 150, 150]]*len(pb_mesh.faces))
        t_mesh.fix_normals()
        t_mesh = t_mesh.apply_scale(scale)
        # import IPython
        # IPython.embed()

        mesh_pos = visual_data[5]
        mesh_orn = visual_data[6]
        mesh_tform = pb_robot.geometry.tform_from_pose((mesh_pos, mesh_orn))

        return t_mesh, mesh_tform, mesh_fname

    def pb_draw_contacts(self, pb_point1, pb_point2):
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.rightVisualShapeId,
                            basePosition=pb_point1, useMaximalCoordinates=True, physicsClientId=self.pb_client_id)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.leftVisualShapeId,
            basePosition=pb_point2, useMaximalCoordinates=True, physicsClientId=self.pb_client_id)

    def pb_check_grasp_collision(self, grasp_pose):
        self.hand.set_base_link_pose(grasp_pose)
        # self.hand.set_base_link_pose(pb_robot.geometry.invert(grasp_pose))
        result = p.getClosestPoints(bodyA=self.hand.id,
                            bodyB=self.body_id,
                            distance=0,
                            physicsClientId=self.pb_client_id)
        collision = len(result) != 0
        # print(result)
        # if len(result) != 0:
        #     pb_robot.viz.draw_point(result[0][5])
        #     pb_robot.viz.draw_point(result[0][6])

        # pb_robot.viz.remove_all_debug()
        return collision

    def pb_get_poses(self):
       return [p.getBasePositionAndOrientation(body_id, physicsClientId=self.pb_client_id) for body_id in self.body_ids]

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
        grasp_left = multiply(grasp.ee_relpose, Pose(Point(y=-dist/2)))[0]
        grasp_right =multiply(grasp.ee_relpose, Pose(Point(y=dist/2)))[0]
        grasp_left = apply_transform(object2mesh, grasp_left)
        grasp_right = apply_transform(object2mesh, grasp_right)

        grasp_arrow = trimesh.load_path([grasp_left, grasp_right], colors=[color])
        if np.linalg.norm(grasp_left-tm_point1) < np.linalg.norm(grasp_left-tm_point2):
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

    def tm_show_grasps(self, grasps, labels=None, fname=''):
        grasp_arrows = []
        for gx, g in enumerate(grasps):
            if labels is not None:
                color = [0, 255, 0, 255] if labels[gx] == 1 else [255, 0, 0, 255]
            else:
                color = [0, 0, 255, 255]

            if isinstance(g, Grasp):
                grasp_arrows += self._get_trimesh_grasp_viz(g, color)
            else:
                grasp_arrows += self._get_trimesh_grasp_array_viz(g, color)
        axis = self._get_tm_com_axis()
        scene = trimesh.scene.Scene([self.mesh, axis] + grasp_arrows)

        if len(fname) > 0:
            for angles, name in zip([(0, 0, 0), (np.pi/2, 0, 0), (np.pi/2, 0, np.pi/2)], ['z', 'y', 'x']):
                scene.set_camera(angles=angles, distance=0.6, center=self.mesh.centroid)
                with open(fname.replace('.png', '_%s.png' % name), 'wb') as handle:
                    handle.write(scene.save_image())
        else:
            scene.set_camera(angles=(np.pi/2, 0, np.pi/4), distance=0.5, center=self.mesh.centroid)
            scene.show()

    def tm_get_aabb(self, pose):
        tform = pb_robot.geometry.tform_from_pose(pose)
        return self.mesh.apply_transform(tform).bounds




class GraspSimulationClient:

    def __init__(self, graspable_body, show_pybullet, urdf_directory):
        """ Support both PyBullet and Trimesh for simulations and visualizations.
        """
        self.shapenet_root = '/home/seiji/Research/object_models/shapenet-sem/'

        self.pb_client_id = pb_robot.utils.connect(use_gui=True if show_pybullet else False)
        if self.pb_client_id > 5:
            print('[ERROR] Too many pybullet clients open.')
            sys.exit()

        pb_robot.utils.set_pbrobot_clientid(self.pb_client_id)
        p.setGravity(0, 0, 0, physicsClientId=self.pb_client_id)
        self.graspable_body = graspable_body
        self.urdf_directory = urdf_directory
        if not os.path.exists(urdf_directory):
            os.mkdir(urdf_directory)

        urdf_path = self._get_object_urdf(graspable_body)

        # TODO: In the future this information should be read from the URDF, and also
        # try to match up with trimesh primitives in case the urdf geom is a primitive
        self.body_id = p.loadURDF(urdf_path, physicsClientId=self.pb_client_id)
        self.mesh, self.mesh_tform, self.mesh_fname = self._load_mesh()
        #p.changeDynamics(self.body_id, -1, mass=graspable_body.mass, lateralFriction=graspable_body.friction, spinningFriction=0.005, physicsClientId=self.pb_client_id)

        self.rightVisualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=0.005, physicsClientId=self.pb_client_id)
        self.leftVisualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0, 1, 0, 1], radius=0.005, physicsClientId=self.pb_client_id)

        self.RED = [255, 0, 0, 255]
        self.show_pybullet = show_pybullet

        self.hand, self.hand_control = self._load_hand()


    def _get_object_urdf(self, graspable_body):
        # TODO: modify to include the simple blocks case
        """ First copy the YCB Object to a new folder then modify its URDF to include 
        the specified intrinsic properties.

        TODO: Right now we only support loading ycb_objects.
        """
        object_dataset, object_name = graspable_body.object_name.split('::')
        if object_dataset.lower() == 'ycb':
            # Copy all files for the object to a temporary location.
            src_path = os.path.join(ycb_objects.getDataPath(), object_name)
            dst_object_name = '%s_%.2fm_%.2ff_%.2fcx_%.2fcy_%.2fcz' % (graspable_body.object_name, graspable_body.mass, graspable_body.friction, graspable_body.com[0], graspable_body.com[1], graspable_body.com[2])

            dst_path = os.path.join(self.urdf_directory, dst_object_name)
            if not os.path.exists(dst_path):
                shutil.copytree(src=src_path, dst=dst_path)

            # Update the URDF parameters.
            urdf_path = os.path.join(dst_path, 'model.urdf')
        elif object_dataset.lower() == 'shapenet':
            src_path = os.path.join(self.shapenet_root, 'urdfs', f'{object_name}.urdf')
            dst_object_name = '%s_%.2fm_%.2ff_%.2fcx_%.2fcy_%.2fcz.urdf' % (graspable_body.object_name, graspable_body.mass, graspable_body.friction, graspable_body.com[0], graspable_body.com[1], graspable_body.com[2])
            dst_path = os.path.join(self.urdf_directory, dst_object_name)
            shutil.copy(src_path, dst_path)
            urdf_path = dst_path

        with open(urdf_path, 'r') as handle:
            urdf = urdf_to_odio(handle.read())

        locals_dict = {}
        exec('robot = ' + urdf[1:], globals(), locals_dict)
        robot = locals_dict['robot']

        assert(len(robot) == 1)  # Right now we only support single link objects.

        contact = Contact(
            Friction_anchor(),
            Lateral_friction(value=graspable_body.friction),
            Rolling_friction(0.),
            Spinning_friction(0.005), # 0.005
        )

        remove_ixs = []
        for ex, element in enumerate(robot[0]):
            if isinstance(element, Inertial):
                for subelement in element:
                    if isinstance(subelement, Origin):
                        origin = subelement
                remove_ixs.append(ex)
            elif isinstance(element, Contact):
                remove_ixs.append(ex)

        origin.xyz = '%f %f %f' % graspable_body.com
        inertial = Inertial(
            Mass(value=graspable_body.mass),
            origin,
            Inertia(ixx=1e-3, iyy=1e-3, izz=1e-3, ixy=0, ixz=0, iyz=0)
        )

        for ex in sorted(remove_ixs, reverse=True):
            del robot[0][ex]

        robot[0].append(contact)
        robot[0].append(inertial)

        with open(urdf_path, 'w') as handle:
            handle.write(robot.urdf())

        return urdf_path

    def _load_hand(self):
        pb_robot.utils.set_pbrobot_clientid(self.pb_client_id)
        hand = pb_robot.panda.PandaHand()
        init_pos, init_orn = [0.1, -0.115, 0.5], [0, 0, 0, 1]
        hand_control = pb_robot.panda_controls.FloatingHandControl(hand, init_pos, init_orn, client_id=self.pb_client_id)
        hand_control.open()

        p.changeDynamics(hand.id, 0, lateralFriction=1, contactStiffness=30000, contactDamping=1000, spinningFriction=0.01, frictionAnchor=1, physicsClientId=self.pb_client_id)
        p.changeDynamics(hand.id, 1, lateralFriction=1, contactStiffness=30000, contactDamping=1000, spinningFriction=0.01, frictionAnchor=1, physicsClientId=self.pb_client_id) # spinning_friction 0.01

        return hand, hand_control

    def _load_mesh(self):
        visual_data = p.getVisualShapeData(self.body_id, physicsClientId=self.pb_client_id)[0]

        scale = visual_data[3][0]
        mesh_fname = visual_data[4]
        dataset, object_id = self.graspable_body.object_name.split('::')
        if 'ShapeNet' == dataset:
            object_id = object_id.split('_')[-1]
            mesh_fname = os.path.join(self.shapenet_root, 'visual_models', f'{object_id}_centered.obj')

        # THIS DEPENDS ON GETTING MESH FROM URDF - we may have to make simple primitive meshes
        pb_mesh = pb_robot.meshes.read_obj(mesh_fname, decompose=False)
        t_mesh = trimesh.Trimesh(pb_mesh.vertices, pb_mesh.faces, face_colors=[[150, 150, 150, 150]]*len(pb_mesh.faces))
        t_mesh.fix_normals()
        t_mesh = t_mesh.apply_scale(scale)
        # import IPython
        # IPython.embed()

        mesh_pos = visual_data[5]
        mesh_orn = visual_data[6]
        mesh_tform = pb_robot.geometry.tform_from_pose((mesh_pos, mesh_orn))

        return t_mesh, mesh_tform, mesh_fname

    def pb_draw_contacts(self, pb_point1, pb_point2):
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.rightVisualShapeId,
                            basePosition=pb_point1, useMaximalCoordinates=True, physicsClientId=self.pb_client_id)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.leftVisualShapeId,
            basePosition=pb_point2, useMaximalCoordinates=True, physicsClientId=self.pb_client_id)

    def pb_check_grasp_collision(self, grasp_pose):
        self.hand.set_base_link_pose(grasp_pose)
        # self.hand.set_base_link_pose(pb_robot.geometry.invert(grasp_pose))
        result = p.getClosestPoints(bodyA=self.hand.id,
                            bodyB=self.body_id,
                            distance=0,
                            physicsClientId=self.pb_client_id)
        collision = len(result) != 0
        # print(result)
        # if len(result) != 0:
        #     pb_robot.viz.draw_point(result[0][5])
        #     pb_robot.viz.draw_point(result[0][6])

        # pb_robot.viz.remove_all_debug()
        return collision

    def pb_get_pose(self):
       return p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.pb_client_id)

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
        grasp_left = multiply(grasp.ee_relpose, Pose(Point(y=-dist/2)))[0]
        grasp_right =multiply(grasp.ee_relpose, Pose(Point(y=dist/2)))[0]
        grasp_left = apply_transform(object2mesh, grasp_left)
        grasp_right = apply_transform(object2mesh, grasp_right)

        grasp_arrow = trimesh.load_path([grasp_left, grasp_right], colors=[color])
        if np.linalg.norm(grasp_left-tm_point1) < np.linalg.norm(grasp_left-tm_point2):
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

    def tm_show_grasps(self, grasps, labels=None, fname=''):
        grasp_arrows = []
        for gx, g in enumerate(grasps):
            if labels is not None:
                color = [0, 255, 0, 255] if labels[gx] == 1 else [255, 0, 0, 255]
            else:
                color = [0, 0, 255, 255]

            if isinstance(g, Grasp):
                grasp_arrows += self._get_trimesh_grasp_viz(g, color)
            else:
                grasp_arrows += self._get_trimesh_grasp_array_viz(g, color)
        axis = self._get_tm_com_axis()
        scene = trimesh.scene.Scene([self.mesh, axis] + grasp_arrows)

        if len(fname) > 0:
            for angles, name in zip([(0, 0, 0), (np.pi/2, 0, 0), (np.pi/2, 0, np.pi/2)], ['z', 'y', 'x']):
                scene.set_camera(angles=angles, distance=0.6, center=self.mesh.centroid)
                with open(fname.replace('.png', '_%s.png' % name), 'wb') as handle:
                    handle.write(scene.save_image())
        else:
            scene.set_camera(angles=(np.pi/2, 0, np.pi/4), distance=0.5, center=self.mesh.centroid)
            scene.show()

    def tm_get_aabb(self, pose):
        tform = pb_robot.geometry.tform_from_pose(pose)
        return self.mesh.apply_transform(tform).bounds

class ParallelGraspStabilityChecker:
    """
    Check stability of a grasp by apply various forces to an object and comparing relative poses of the object. 
    
    Stability checking will happen by applying perturbations in different directions and ensuring the pose 
    does not change beyond a specified threshold. Two modes are currently supported:
    :stability_direction='gravity': Only apply gravity is the -z direction. Note this is unrealistic unless the
        object's initial pose is accurate.
    :stability_direction='all': Apply a perturbation in multiple directions.
    :TODO: Eventually, we might want to perform a shaking motion as in the Acronym dataset because this will be 
        more realistic for collecting data on a real robot (we can't change gravity in the real world).
    
    We also support two types of stability labels:
    :label_type='contact': True if the object remains in the object's gripper.
    'label_type='relpose': True if the object's relative pose with the gripper does not change during the motion.
    """
    def __init__(self, graspable_bodies, stability_direction='all', label_type='relpose', grasp_noise=0.0, show_pybullet=False):
        assert(stability_direction in ['all', 'gravity'])
        assert(label_type in ['relpose', 'contact'])
        self.stability_direction = stability_direction
        self.label_type = label_type
        self.grasp_noise = grasp_noise
        self.show_pybullet = show_pybullet

        self.sim_client = ParallelGraspSimulationClient(graspable_bodies, show_pybullet=show_pybullet, urdf_directory='urdf_models')
        self.reset_poses = []
        for body_id in self.sim_client.body_ids:
            pose = p.getBasePositionAndOrientation(body_id, physicsClientId=self.sim_client.pb_client_id)
            self.reset_poses.append(pose)

    def get_noisy_grasps(self, grasps):
        new_grasps = []
        for grasp in grasps:
            pos, orn = grasp.ee_relpose
            new_pos = np.array(pos) + np.random.randn(3)*self.grasp_noise
            new_grasp = Grasp(graspable_body=grasp.graspable_body,
                pb_point1=grasp.pb_point1, pb_point2=grasp.pb_point2,
                pitch=grasp.pitch, roll=grasp.roll,
                ee_relpose=(new_pos, orn), force=grasp.force)
            new_grasps.append(new_grasp)
        return new_grasps

    def _reset(self):
        for body_id, pose, hand in zip(self.sim_client.body_ids, self.reset_poses, self.sim_client.hands):
            p.resetBasePositionAndOrientation(body_id, pose[0], pose[1], physicsClientId=self.sim_client.pb_client_id)
            p.setGravity(0, 0, 0, physicsClientId=self.sim_client.pb_client_id)
            pb_robot.utils.set_pbrobot_clientid(self.sim_client.pb_client_id)
            hand.Open()

    def show_contact_points(self):
        results = p.getContactPoints(
            bodyA=self.sim_client.hands[0].id,
            bodyB=self.sim_client.body_ids[0],
            physicsClientId=self.sim_client.pb_client_id
        )
        pb_robot.viz.remove_all_debug()
        for rx, result in enumerate(results):
            point1 = result[5]
            point2 = result[6]
            normalDir = result[7]
            end = np.array(point2) + np.array(normalDir)*0.02

            p.addUserDebugLine(point2, end, lineColorRGB=[1,0,0], lineWidth=0.02,
                               lifeTime=0,
                              physicsClientId=self.sim_client.pb_client_id)


    def get_labels(self, grasps):
        assert(len(grasps)==len(self.sim_client.body_ids))

        gravity_vectors = self._get_gravity_vectors(10)
        grasps = self.get_noisy_grasps(grasps)
        labels = [True] * len(grasps)
        for gx in range(gravity_vectors.shape[0]):
            self._reset()

            self.sim_client.hand_control.set_pose([grasp.ee_relpose for grasp in grasps])
            self.sim_client.hand_control.close(forces=[grasp.force for grasp in grasps], wait=self.show_pybullet)

            init_poses = self.sim_client.pb_get_poses()

            freeze_poses = [None]*len(self.sim_client.body_ids)

            self.sim_client.pb_set_gravity(gravity_vectors[gx, :])
            for tx in range(100):
                self.sim_client.hand_control.move_to(
                    [grasp.ee_relpose for grasp in grasps],
                    [grasp.force for grasp in grasps],
                    wait=self.show_pybullet)

                if self.show_pybullet and tx % 5 == 0:
                    self.show_contact_points()

                for hx in range(len(self.sim_client.body_ids)):
                    if freeze_poses[hx] is None:
                        object_pose = p.getBasePositionAndOrientation(self.sim_client.body_ids[hx], physicsClientId=self.sim_client.pb_client_id)
                        pos_diff = np.linalg.norm(np.array(object_pose[0])-np.array(init_poses[hx][0]))
                        angle_diff = pb_robot.geometry.quat_angle_between(object_pose[1], init_poses[hx][1])
                        if pos_diff > 0.02 or angle_diff > 10:
                            freeze_poses[hx] = object_pose
                    else:
                        p.resetBasePositionAndOrientation(self.sim_client.body_ids[hx],
                           freeze_poses[hx][0],
                           freeze_poses[hx][1],
                           physicsClientId=self.sim_client.pb_client_id)


            end_poses = self.sim_client.pb_get_poses()


            for hx, (init_pose, end_pose) in enumerate(zip(init_poses, end_poses)):
                pos_diff = np.linalg.norm(np.array(end_pose[0])-np.array(init_pose[0]))
                angle_diff = pb_robot.geometry.quat_angle_between(end_pose[1], init_pose[1])
                if pos_diff > 0.02 or angle_diff > 10: # 0.01/5
                    stable = False
                    labels[hx] = False
                # else:
                #     stable = True
                #labels.append(stable)

        # print(f'Stable: {stable}\tPos: {pos_diff}\tAngle: {angle_diff}')
        return labels

    def disconnect(self):
        self.sim_client.disconnect()

    def draw_gravity(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        gravity = self._get_gravity_vectors(100)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(gravity[:, 0], gravity[:, 1], gravity[:, 2])
        plt.show()

    def _get_gravity_vectors(self, n_samples):
        # TODO: Get vectors that are aligned with grasp.
        points = []
        for _ in range(1000):
            gravity = np.random.randn(3)
            gravity = 10*gravity/np.linalg.norm(gravity)

            points.append(gravity)

        points = np.array(points)
        points = self._k_farthest_points(points, n_samples)
        return points

    def _k_farthest_points(self, points, k):
        ixs = [0]
        min_distances = np.linalg.norm(points - points[0:1, :], axis=1)
        for _ in range(k-1):
            # Iteratively choose the point that is farthest.
            new_ix = np.argmax(min_distances)
            ixs.append(new_ix)

            dist_to_new = np.linalg.norm(points - points[new_ix:new_ix+1, :], axis=1)
            min_distances = np.stack([min_distances, dist_to_new],-1)
            min_distances = np.min(min_distances, axis=1)

        return points[ixs,...]

class GraspStabilityChecker:
    """
    Check stability of a grasp by apply various forces to an object and comparing relative poses of the object. 
    
    Stability checking will happen by applying perturbations in different directions and ensuring the pose 
    does not change beyond a specified threshold. Two modes are currently supported:
    :stability_direction='gravity': Only apply gravity is the -z direction. Note this is unrealistic unless the
        object's initial pose is accurate.
    :stability_direction='all': Apply a perturbation in multiple directions.
    :TODO: Eventually, we might want to perform a shaking motion as in the Acronym dataset because this will be 
        more realistic for collecting data on a real robot (we can't change gravity in the real world).
    
    We also support two types of stability labels:
    :label_type='contact': True if the object remains in the object's gripper.
    'label_type='relpose': True if the object's relative pose with the gripper does not change during the motion.
    """
    def __init__(self, graspable_body, stability_direction='all', label_type='relpose', grasp_noise=0.0, show_pybullet=False):
        assert(stability_direction in ['all', 'gravity'])
        assert(label_type in ['relpose', 'contact'])
        self.stability_direction = stability_direction
        self.label_type = label_type
        self.grasp_noise = grasp_noise

        self.sim_client = GraspSimulationClient(graspable_body, show_pybullet=show_pybullet, urdf_directory='urdf_models')
        self.reset_pose = p.getBasePositionAndOrientation(self.sim_client.body_id, physicsClientId=self.sim_client.pb_client_id)

        self.show_pybullet = show_pybullet

    def get_noisy_grasp(self, grasp):
        pos, orn = grasp.ee_relpose
        new_pos = np.array(pos) + np.random.randn(3)*self.grasp_noise
        new_grasp = Grasp(graspable_body=grasp.graspable_body,
            pb_point1=grasp.pb_point1, pb_point2=grasp.pb_point2,
            pitch=grasp.pitch, roll=grasp.roll,
            ee_relpose=(new_pos, orn), force=grasp.force)

        return new_grasp

    def _reset(self):
        p.resetBasePositionAndOrientation(self.sim_client.body_id, self.reset_pose[0], self.reset_pose[1], physicsClientId=self.sim_client.pb_client_id)
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
            end = np.array(point2) + np.array(normalDir)*0.02

            p.addUserDebugLine(point2, end, lineColorRGB=[1,0,0], lineWidth=0.02,
                               lifeTime=0,
                              physicsClientId=self.sim_client.pb_client_id)

    def get_label(self, grasp):
        self._reset()
        # input('Reset...')

        grasp = self.get_noisy_grasp(grasp)

        stable = True
        # gravity_vectors = np.concatenate([self._get_gravity_vectors_inplane(grasp, 20), self._get_gravity_vectors(10)])
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
                    input('Next?')
                self.sim_client.hand_control.move_to(grasp.ee_relpose[0], grasp.ee_relpose[1], grasp.force, wait=self.show_pybullet)
            end_pose = self.sim_client.pb_get_pose()

            pos_diff = np.linalg.norm(np.array(end_pose[0])-np.array(init_pose[0]))
            angle_diff = pb_robot.geometry.quat_angle_between(end_pose[1], init_pose[1])
            if pos_diff > 0.02 or angle_diff > 10: # 0.01/5
                stable = False
                break

        print(f'Stable: {stable}\tPos: {pos_diff}\tAngle: {angle_diff}')
        return stable

    def pb_draw_gravity(self, gravity_vectors):
        for v in gravity_vectors:
            pb_robot.viz.draw_point(v/20.)

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
                self.sim_client.hand_control.move_to([x, y, z+0.001*ix], grasp.ee_relpose[1], grasp.force, wait=self.show_pybullet)

            for _ in range(100):
                self.sim_client.hand_control.move_to([x, y, z + 0.001*500], grasp.ee_relpose[1], grasp.force, wait=self.show_pybullet)
        else:
            gravity_vectors = self._get_gravity_vectors(10)
            for gx in range(gravity_vectors.shape[0]):
                self.sim_client.pb_set_gravity(gravity_vectors[gx, :])
                for _ in range(100):
                    self.sim_client.hand_control.move_to(grasp.ee_relpose[0], grasp.ee_relpose[1], grasp.force, wait=self.show_pybullet)
        end_pose = self.sim_client.pb_get_pose()

        pos_diff = np.linalg.norm(np.array(end_pose[0])-np.array(init_pose[0]))
        angle_diff = pb_robot.geometry.quat_angle_between(end_pose[1], init_pose[1])
        if pos_diff > 0.02 or angle_diff > 10: # 0.01/5
            stable = False
        else:
            stable = True

        print(f'Stable: {stable}\tPos: {pos_diff}\tAngle: {angle_diff}')
        return stable

    def disconnect(self):
        self.sim_client.disconnect()

    def _get_gravity_vectors_inplane(self, grasp, n_samples):
        n_pool = n_samples*100
        angles = np.random.uniform(0, 2*np.pi, n_pool)
        points_x = np.cos(angles)*10
        points_z = np.sin(angles)*10
        points_gripper_frame = np.hstack([points_x[:, None], np.zeros((n_pool, 1)), points_z[:, None], np.ones((n_pool, 1))])
        points_gripper_frame = self._k_farthest_points(points_gripper_frame, n_samples)
        # import IPython
        # IPython.embed()
        points_global_frame = tform_from_pose(grasp.ee_relpose)@(points_gripper_frame.T)
        points_global_frame = points_global_frame.T[:, 0:3]
        return points_global_frame

    def _get_gravity_vectors(self, n_samples):
        points = []
        for _ in range(1000):
            gravity = np.random.randn(3)
            gravity = 10*gravity/np.linalg.norm(gravity)

            points.append(gravity)

        points = np.array(points)
        points = self._k_farthest_points(points, n_samples)
        return points

    def _k_farthest_points(self, points, k):
        ixs = [0]
        min_distances = np.linalg.norm(points - points[0:1, :], axis=1)
        for _ in range(k-1):
            # Iteratively choose the point that is farthest.
            new_ix = np.argmax(min_distances)
            ixs.append(new_ix)

            dist_to_new = np.linalg.norm(points - points[new_ix:new_ix+1, :], axis=1)
            min_distances = np.stack([min_distances, dist_to_new],-1)
            min_distances = np.min(min_distances, axis=1)

        return points[ixs,...]

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
                                                show_pybullet=show_pybullet,
                                                urdf_directory=urdf_directory)

        self.gripper_width = 0.08
        self.antipodal_tolerance = np.deg2rad(antipodal_tolerance)
        self.show_pybullet = show_pybullet
        self.graspable_body = graspable_body

    def _sample_antipodal_points_rays(self, max_attempts=5000):
        """ Return two antipodal points in the visual frame (which may not be the object frame).
        """
        points = []
        # for _ in range(max_attempts):
        while True:
            [point1], [index1] = self.sim_client.mesh.sample(1, return_index=True)
            normal1 = np.array(self.sim_client.mesh.face_normals[index1, :])

            intersections = self.sim_client.mesh.ray.intersects_location([point1], [-normal1])
            hit_points, _, hit_faces = intersections

            if hit_points.shape[0]  <= 1:
                # Only found the origin as intersection point.
                continue

            point_intersect, index_intersect = hit_points[1], hit_faces[1]
            face_weight = np.zeros((len(self.sim_client.mesh.faces)))
            face_weight[index_intersect] = 1.

            for _ in range(10):
                [point2], [index2] = self.sim_client.mesh.sample(1, return_index=True, face_weight=face_weight)

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

                # For anitpodal grasps, the angle between the normal and direction vector should be small.
                if (error1 > self.antipodal_tolerance) or (error2 > self.antipodal_tolerance):
                    continue

                return point1, point2
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
                if ax == 0:
                    import IPython
                    IPython.embed()
                error1 = pb_robot.geometry.angle_between(normal1, -direction)
                error2 = pb_robot.geometry.angle_between(normal2, direction)

                # For anitpodal grasps, the angle between the normal and direction vector should be small.
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

        #for _ in range(max_attempts):
        while True:
            tm_point1, tm_point2 = self._sample_antipodal_points_rays()
            if tm_point1 is None:
                return None
            # The visual frame of reference might be different from the object's link frame.
            pb_point1 = apply_transform(self.sim_client.mesh_tform, tm_point1)
            pb_point2 = apply_transform(self.sim_client.mesh_tform, tm_point2)
            #self.sim_client.pb_draw_contacts(pb_point1, pb_point2)
            for _ in range(10):
                # Pitch is the angle of the grasp while roll controls the orientation (flipped gripper or not).
                pitch = random.uniform(-np.pi, np.pi)
                #pitch = random.choice([-np.pi, np.pi])
                # roll = random.choice([0, np.pi])  # Not used. Should be covered by point-ordering.
                roll = 0
                grasp_point = (pb_point1 + pb_point2)/2

                # The contact points define a plane (contact plane).
                normal = (pb_point2 - pb_point1)/pb_robot.geometry.get_length(pb_point2-pb_point1)
                origin = np.zeros(3)

                # Calculate the transform that brings the XY-plane to the contact plane.
                tform = np.linalg.inv(trimesh.points.plane_transform(origin, -normal))
                quat1 = pb_robot.geometry.quat_from_matrix(tform[0:3, 0:3])
                # z-axis in direction of point2 to point1
                pose1 = pb_robot.geometry.Pose(origin, euler=pb_robot.geometry.euler_from_quat(quat1))

                # Project (0, 0, 1) to the contact plane.
                point = np.array((0, 0, 1))
                distance = np.dot(normal, point - np.array(origin))
                projection_world = point - distance*normal

                # This gives us the (x, y) coordinates of the projected point in contact-plane coordinates. 
                projection = pb_robot.geometry.tform_point(pb_robot.geometry.invert(pose1), projection_world)  # Do we need to invert pose1?
                yaw = np.math.atan2(projection[1], projection[0])
                quat2 = pb_robot.geometry.multiply_quats(quat1, pb_robot.geometry.quat_from_euler(pb_robot.geometry.Euler(yaw=yaw)))

                grasp_quat = pb_robot.geometry.multiply_quats(
                    quat2,
                    pb_robot.geometry.quat_from_euler(pb_robot.geometry.Euler(roll=np.pi / 2)),
                    pb_robot.geometry.quat_from_euler(pb_robot.geometry.Euler(pitch=pitch)), # TODO: local pitch or world pitch?
                    # pb_robot.geometry.quat_from_euler(pb_robot.geometry.Euler(roll=roll)),  # Switches fingers
                )

                # pose2 = pb_robot.geometry.Pose(grasp_point, euler=pb_robot.geometry.euler_from_quat(grasp_quat))
                # pb_robot.viz.draw_pose(pose2)
                finger_length=0.1034
                grasp_pose = pb_robot.geometry.Pose(grasp_point, pb_robot.geometry.euler_from_quat(grasp_quat))
                grasp_pose = pb_robot.geometry.multiply(grasp_pose, pb_robot.geometry.Pose(pb_robot.geometry.Point(z=-finger_length))) # FINGER_LENGTH

                collision = self.sim_client.pb_check_grasp_collision(grasp_pose)
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

                    return grasp

    def disconnect(self):
        self.sim_client.disconnect()


class GraspableBodySampler:
    MASS_RANGE = (0.1, 1.0)
    FRICTION_RANGE = (0.1, 1.0)
    # COM is sampled in % of containing bounding box and rejection sampling is used to
    # make sure it lies within the mesh.

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
        sim_client = GraspSimulationClient(tmp_body, show_pybullet=False, urdf_directory='urdf_models')
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
    objects_names = [name for name in os.listdir(ycb_objects.getDataPath()) if 'Ycb' in name]
    # objects_names = ['YCB::YcbCrackerBox']
    #objects_names = ['ShapeNet::Desk_fe2a9f23035580ce239883c5795189ed']
    #objects_names = ['ShapeNet::ComputerMouse_379e93edfd0cb9e4cc034c03c3eb69d']
    #objects_names = ['ShapeNet::Chair_198a3e82b102529c4904d89e9169817b']
    #objects_names = ['ShapeNet::Barstool_55e7dc1021e15181a495c196d4f0cebb']
    #objects_names = ['ShapeNet::Dresser_e9e3f04bce3933a2c62986712894256b']
    #objects_names = ['ShapeNet::MilkCarton_64018b545e9088303dd0d6160c4dfd18']
    objects_names = ['ShapeNet::WineGlass_2d89d2b3b6749a9d99fbba385cc0d41d']
    #objects_names = ['ShapeNet::WallLamp_8be32f90153eb7281b30b67ce787d4d3']
    objects_names = ['ShapeNet::USBStick_ab82d56cf9cc2476d154e1b098031d39']
    objects_names = ['ShapeNet::TV_1a595fd7e7043a06b0d7b0d4230df8ca']

    object_name = random.choice(objects_names)
    graspable_body = GraspableBody(object_name=object_name,
        com=(0.00400301, 0.01275706, 0.02090709),
        mass=0.52519492,
        friction=0.57660351)

    #graspable_body = GraspableBody(object_name=object_name, com=(0.05, -0.02, -0.008), mass=0.93, friction=0.24) # m=0.93
    graspable_body = GraspableBody(object_name=object_name, com=(-0.0586, 0.0019, -0.0419), mass=2.908, friction=1.84) # mass=0.908
    #graspable_body = GraspableBodySampler.sample_random_object_properties(object_name)

    grasp_sampler = GraspSampler(graspable_body=graspable_body, antipodal_tolerance=30, show_pybullet=True)
    grasp_sampler.sim_client.mesh.show()
    n_samples = 100
    grasps = []
    for lx in range(0, n_samples):
        print('Sampling %d/%d...' % (lx, n_samples))
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasps.append(grasp)
    grasp_sampler.disconnect()
    sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
    #sim_client.tm_show_grasps(grasps)#, fname='test.png')
    sim_client.disconnect()

    labeler = GraspStabilityChecker(graspable_body,
        stability_direction='all',
        label_type='relpose',
        grasp_noise=0.0,
        show_pybullet=False)
    labels1 = []
    labels2 = []
    for lx, grasp in enumerate(grasps):
        print('Labeling %d/%d...' % (lx, n_samples))
        labels1.append(labeler.get_label(grasp))
        labels2.append(labeler.get_label(grasp))

        # l1 = labeler.get_label(grasp, show_pybullet=False)
        # l2 = labeler.get_label(grasp, show_pybullet=False)
        # print('Consistant:', l1 == l2)
    labeler.disconnect()

    print('Equals: ', (np.array(labels1) == np.array(labels2)).sum())
    sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
    sim_client.tm_show_grasps(grasps, labels1)
    sim_client.tm_show_grasps(grasps, np.array(labels1) == np.array(labels2))#, fname='test.png')
    sim_client.disconnect()

def main_parallel():
    objects_names = [name for name in os.listdir(ycb_objects.getDataPath()) if 'Ycb' in name]
    # objects_names = ['YCB::YcbCrackerBox']
    #objects_names = ['ShapeNet::Desk_fe2a9f23035580ce239883c5795189ed']
    #objects_names = ['ShapeNet::ComputerMouse_379e93edfd0cb9e4cc034c03c3eb69d']
    #objects_names = ['ShapeNet::Chair_198a3e82b102529c4904d89e9169817b']
    # objects_names = ['ShapeNet::Barstool_55e7dc1021e15181a495c196d4f0cebb']
    #objects_names = ['ShapeNet::Dresser_e9e3f04bce3933a2c62986712894256b']
    #objects_names = ['ShapeNet::MilkCarton_64018b545e9088303dd0d6160c4dfd18']
    #objects_names = ['ShapeNet::WineGlass_2d89d2b3b6749a9d99fbba385cc0d41d']
    #objects_names = ['ShapeNet::WallLamp_8be32f90153eb7281b30b67ce787d4d3']
    #objects_names = ['ShapeNet::Candle_5526503b9089aa12b21e97993de5df16']
    objects_names = ['ShapeNet::USBStick_ab82d56cf9cc2476d154e1b098031d39']
    #objects_names = ['ShapeNet::TV_1a595fd7e7043a06b0d7b0d4230df8ca']

    object_name = random.choice(objects_names)
    graspable_body = GraspableBody(object_name=object_name,
        com=(0.00400301, 0.01275706, 0.02090709),
        mass=0.52519492,
        friction=0.57660351)

    graspable_body = GraspableBody(object_name=object_name, com=(0.05, -0.02, -0.008), mass=0.8, friction=0.4) # m=0.93
    #graspable_body = GraspableBody(object_name=object_name, com=(-0.1386, 0.0019, -0.0419), mass=0.8, friction=0.843) # mass=0.908
    #graspable_body = GraspableBodySampler.sample_random_object_properties(object_name)

    sim_client = ParallelGraspSimulationClient([graspable_body]*5, show_pybullet=False, urdf_directory='object_models')
    sim_client.disconnect()

    grasp_sampler = GraspSampler(graspable_body=graspable_body, antipodal_tolerance=30, show_pybullet=True)
    # grasp_sampler.sim_client.mesh.show()
    n_samples = 100
    grasps = []
    for lx in range(0, n_samples):
        print('Sampling %d/%d...' % (lx, n_samples))
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasps.append(grasp)
    grasp_sampler.disconnect()
    sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
    # sim_client.tm_show_grasps(grasps)#, fname='test.png')
    sim_client.disconnect()

    labeler = ParallelGraspStabilityChecker([graspable_body]*n_samples,
        stability_direction='all',
        label_type='relpose',
        grasp_noise=0.0,
        show_pybullet=False)
    print('Getting first label set...')
    #np.random.seed(10)
    labeler.draw_gravity()
    labels1 = labeler.get_labels(grasps)
    print(labels1)
    # print('Getting second label set...')
    #np.random.seed(10)
    labels2 = labeler.get_labels(grasps)
    print(labels2)
    labeler.disconnect()
    print(labels1==labels2)
    import IPython
    IPython.embed()
    sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
    sim_client.tm_show_grasps(grasps, labels1)
    sim_client.tm_show_grasps(grasps, np.array(labels1)==np.array(labels2))#, fname='test.png')
    sim_client.disconnect()

    labeler = ParallelGraspStabilityChecker([graspable_body],
        stability_direction='all',
        label_type='relpose',
        grasp_noise=0.0,
        show_pybullet=True)
    for gx in range(len(labels1)):
        if labels1[gx] != labels2[gx]:
            labeler.get_labels([grasps[gx]])

            input('Next grasp?')

def vary_object_properties():
    # object_name = 'ShapeNet::USBStick_ab82d56cf9cc2476d154e1b098031d39'
    # graspable_body = GraspableBody(object_name=object_name, com=(0.05, -0.02, -0.008), mass=0.93, friction=0.24) # m=0.93

    object_name = 'ShapeNet::TV_1a595fd7e7043a06b0d7b0d4230df8ca'
    graspable_body = GraspableBody(object_name=object_name, com=(-0.1386, 0.0019, -0.0419), mass=0.908, friction=0.84) # mass=0.908

    n_samples = 100
    grasps = []
    grasp_sampler = GraspSampler(graspable_body=graspable_body, antipodal_tolerance=30, show_pybullet=False)

    for lx in range(0, n_samples):
        print('Sampling %d/%d...' % (lx, n_samples))
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasps.append(grasp)
    grasp_sampler.disconnect()

    for m in np.linspace(0.1, 1.0, 10):
        for f in np.linspace(0.1, 1.0, 10):
            graspable_body = GraspableBody(object_name=object_name, com=(-0.1386, 0.0019, -0.0419), mass=m, friction=f)



            # sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
            # #sim_client.tm_show_grasps(grasps)#, fname='test.png')
            # sim_client.disconnect()

            labeler = GraspStabilityChecker(graspable_body,
                stability_direction='all',
                label_type='relpose',
                grasp_noise=0.0,
                show_pybullet=False)
            labels = []
            for lx, grasp in enumerate(grasps):
                print('Labeling %d/%d...' % (lx, n_samples))
                labels.append(labeler.get_label(grasp))
            labeler.disconnect()

            # print('Equals: ', (np.array(labels1) == np.array(labels2)).sum())
            sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
            sim_client.tm_show_grasps(grasps, labels, fname='/home/seiji/Pictures/grasp_data_inspection/%s_%.2fm_%.2ff_new.png' % (object_name, m, f))
            # sim_client.tm_show_grasps(grasps, np.array(labels1) == np.array(labels2))#, fname='test.png')
            sim_client.disconnect()

def display_object():
    masses = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    frictions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    from mpl_toolkits.axes_grid1 import ImageGrid
    from PIL import Image

    import matplotlib.pyplot as plt
    plt.clf()
    fig = plt.figure(figsize=(40., 40.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(len(masses), len(frictions)),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    #object_name = 'ShapeNet::USBStick_ab82d56cf9cc2476d154e1b098031d39'
    object_name = 'ShapeNet::TV_1a595fd7e7043a06b0d7b0d4230df8ca'
    for mx, mass in enumerate(masses):
        for fx, friction in enumerate(frictions):
            fname = f'/home/seiji/Pictures/grasp_data_inspection/{object_name}_{mass:.2f}m_{friction:.2f}f_new_y.png'
            with open(fname, 'rb') as handle:
                im = np.array(Image.open(handle))

            # Iterating over the grid returns the Axes.
            ax = grid[mx*len(frictions) + fx]
            ax.imshow(im)
            ax.text(50, 375, f'm={mass: .2f}\nf={friction: .2f}', bbox=dict(fill=False, edgecolor='black', linewidth=1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig('/home/seiji/Pictures/all_monitor_new.png')

if __name__ == '__main__':
    # main_parallel()
    main_serial()
    #vary_object_properties()
    #display_object()
