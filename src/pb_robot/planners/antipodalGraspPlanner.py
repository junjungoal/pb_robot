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

from pybullet_object_models import ycb_objects
from pb_robot.geometry import multiply, Point, Pose


def apply_transform(tform, point):
    vec = np.array([[point[0], point[1], point[2], 1]]).T
    return (tform@vec)[0:3, 0]

# TODO: Support ShapeNet objects.
# Note: CoM, mass, and friction properties of the urdf will be overwritten before loading into PyBullet.
GraspableBody = namedtuple('GraspableBody', ['ycb_name', 'com', 'mass', 'friction'])
Grasp = namedtuple('Grasp', ['graspable_body', 'pb_point1', 'pb_point2', 'pitch', 'roll', 'ee_relpose', 'force'])


class GraspSimulationClient:

    def __init__(self, graspable_body, show_pybullet, urdf_directory):
        """ Support both PyBullet and Trimesh for simulations and visualizations.
        """
        self.pb_client_id = pb_robot.utils.connect(use_gui=True if show_pybullet else False)
        p.setGravity(0, 0, 0, physicsClientId=self.pb_client_id)

        self.graspable_body = graspable_body
        self.urdf_directory = urdf_directory
        if not os.path.exists(urdf_directory):
            os.mkdir(urdf_directory)

        urdf_path = self._get_object_urdf(graspable_body)
        self.body_id = p.loadURDF(urdf_path, physicsClientId=self.pb_client_id)
        self.mesh, self.mesh_tform, self.mesh_fname = self._load_mesh()
        # TODO: In the future this information should be read from the URDF.
        #p.changeDynamics(self.body_id, -1, mass=graspable_body.mass, lateralFriction=graspable_body.friction, spinningFriction=0.005, physicsClientId=self.pb_client_id)
        # import IPython
        # IPython.embed()

        self.visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 1, 1, 1], radius=0.005, physicsClientId=self.pb_client_id)
        self.RED = [255, 0, 0, 255]
        self.show_pybullet = show_pybullet

        self.hand, self.hand_control = self._load_hand()
    
    def _get_object_urdf(self, graspable_body):
        """ First copy the YCB Object to a new folder then modify its URDF to include 
        the specified intrinsic properties.

        TODO: Right now we only support loading ycb_objects.
        """
        # Copy all files for the object to a temporary location.
        src_path = os.path.join(ycb_objects.getDataPath(), graspable_body.ycb_name)
        dst_object_name = '%s_%.2fm_%.2ff_%.2fcx_%.2fcy_%.2fcz' % (graspable_body.ycb_name, graspable_body.mass, graspable_body.friction, graspable_body.com[0], graspable_body.com[1], graspable_body.com[2])
        dst_path = os.path.join(self.urdf_directory, dst_object_name)
        if not os.path.exists(dst_path):    
            shutil.copytree(src=src_path, dst=dst_path)

        # Update the URDF parameters.
        urdf_path = os.path.join(dst_path, 'model.urdf')
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
            Spinning_friction(0.005),
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
        hand = pb_robot.panda.PandaHand()
        init_pos, init_orn = [0.1, -0.115, 0.5], [0, 0, 0, 1]
        hand_control = pb_robot.panda_controls.FloatingHandControl(hand, init_pos, init_orn)
        hand_control.open()

        p.changeDynamics(hand.id, 0, lateralFriction=1, contactStiffness=30000, contactDamping=1000, spinningFriction=0.01, frictionAnchor=1, physicsClientId=self.pb_client_id)
        p.changeDynamics(hand.id, 1, lateralFriction=1, contactStiffness=30000, contactDamping=1000, spinningFriction=0.01, frictionAnchor=1, physicsClientId=self.pb_client_id)

        return hand, hand_control

    def _load_mesh(self):
        visual_data = p.getVisualShapeData(self.body_id, physicsClientId=self.pb_client_id)[0]
        mesh_fname = visual_data[4]
        pb_mesh = pb_robot.meshes.read_obj(mesh_fname, decompose=False)
        t_mesh = trimesh.Trimesh(pb_mesh.vertices, pb_mesh.faces, face_colors=[[150, 150, 150, 150]]*len(pb_mesh.faces))
        t_mesh.fix_normals()

        mesh_pos = visual_data[5]
        mesh_orn = visual_data[6]
        mesh_tform = pb_robot.geometry.tform_from_pose((mesh_pos, mesh_orn))

        return t_mesh, mesh_tform, mesh_fname

    def pb_draw_contacts(self, pb_point1, pb_point2):
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visualShapeId,
                            basePosition=pb_point1, useMaximalCoordinates=True, physicsClientId=self.pb_client_id)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visualShapeId,
            basePosition=pb_point2, useMaximalCoordinates=True, physicsClientId=self.pb_client_id)

    def pb_check_grasp_collision(self, grasp_pose):
        self.hand.set_base_link_pose(grasp_pose)
        # self.hand.set_base_link_pose(pb_robot.geometry.invert(grasp_pose))

        collision = len(p.getClosestPoints(bodyA=self.hand.id, 
                            bodyB=self.body_id, 
                            distance=0,
                            physicsClientId=self.pb_client_id)) != 0
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

    def disconnect(self):
        p.disconnect(self.pb_client_id)

    def tm_show_grasps(self, grasps, labels=None, fname=''):
        grasp_arrows = []
        for gx, g in enumerate(grasps):
            if labels is not None:
                color = [0, 255, 0, 255] if labels[gx] == 1 else [255, 0, 0, 255]
            else:
                color = [0, 0, 255, 255]

            grasp_arrows += self._get_trimesh_grasp_viz(g, color)  
        axis = self._get_tm_com_axis()
        scene = trimesh.scene.Scene([self.mesh, axis] + grasp_arrows)

        if len(fname) > 0:
            for angles, name in zip([(0, 0, 0), (np.pi/2, 0, 0), (np.pi/2, 0, np.pi/2)], ['z', 'y', 'x']):
                scene.set_camera(angles=angles, distance=0.5, center=self.mesh.centroid)
                with open(fname.replace('.png', '_%s.png' % name), 'wb') as handle:
                    handle.write(scene.save_image())
        else:
            scene.set_camera(angles=(np.pi/2, 0, np.pi/4), distance=0.5, center=self.mesh.centroid)
            scene.show()   
       

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
    def __init__(self, stability_direction='all', label_type='relpose'):
        assert(stability_direction in ['all', 'gravity'])
        assert(label_type in ['relpose', 'contact'])
        self.stability_direction = stability_direction
        self.label_type = label_type
    
    def get_label(self, grasp, show_pybullet=False):

        sim_client = GraspSimulationClient(grasp.graspable_body, show_pybullet=show_pybullet, urdf_directory='urdf_models')

        sim_client.hand_control.set_pose(grasp.ee_relpose[0], grasp.ee_relpose[1])
        sim_client.hand_control.close(force=grasp.force, wait=show_pybullet)

        init_pose = sim_client.pb_get_pose()
        if self.stability_direction == 'gravity':
            sim_client.pb_set_gravity((0, 0, -10))

            x, y, z = grasp.ee_relpose[0]
            for ix in range(0, 500):
                sim_client.hand_control.move_to([x, y, z+0.001*ix], grasp.ee_relpose[1], grasp.force, wait=show_pybullet)

            for _ in range(100):
                sim_client.hand_control.move_to([x, y, z + 0.001*500], grasp.ee_relpose[1], grasp.force, wait=show_pybullet)
        else:
            sim_client.pb_set_gravity((0, 0, -10))
            for _ in range(100):
                sim_client.hand_control.move_to(grasp.ee_relpose[0], grasp.ee_relpose[1], grasp.force, wait=show_pybullet)
            # TODO: Apply gravitational force in multiple directions.
        end_pose = sim_client.pb_get_pose()

        pos_diff = np.linalg.norm(np.array(end_pose[0])-np.array(init_pose[0]))
        angle_diff = pb_robot.geometry.quat_angle_between(end_pose[1], init_pose[1])
        if pos_diff > 0.01 or angle_diff > 5:
            stable = False
        else:
            stable = True

        print('Stable:', stable)
        sim_client.disconnect()
        return stable
        

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

    def _sample_antipodal_points(self):
        """ Return two antipodal points in the visual frame (which may not be the object frame).
        """
        while True:
            [point1, point2], [index1, index2] = self.sim_client.mesh.sample(2, return_index=True)
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
            error1 = pb_robot.geometry.angle_between(normal1, -direction)
            error2 = pb_robot.geometry.angle_between(normal2, direction)

            # For anitpodal grasps, the angle between the normal and direction vector should be small.
            if (error1 > self.antipodal_tolerance) or (error2 > self.antipodal_tolerance):
                continue
            
            return point1, point2

    def sample_grasp(self, force, show_trimesh=False, max_attempts=100):

        for _ in range(max_attempts):
            tm_point1, tm_point2 = self._sample_antipodal_points()
            # The visual frame of reference might be different from the object's link frame.
            pb_point1 = apply_transform(self.sim_client.mesh_tform, tm_point1)
            pb_point2 = apply_transform(self.sim_client.mesh_tform, tm_point2)

            for _ in range(10):
                # Pitch is the angle of the grasp while roll controls the orientation (flipped gripper or not).
                pitch = random.uniform(-np.pi, np.pi)
                #pitch = random.choice([-np.pi, np.pi])
                roll = random.choice([0, np.pi])  # Not used. Should be covered by point-ordering.

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
                    pb_robot.geometry.quat_from_euler(pb_robot.geometry.Euler(roll=roll)),  # Switches fingers
                )

                # pose2 = pb_robot.geometry.Pose(grasp_point, euler=pb_robot.geometry.euler_from_quat(grasp_quat))
                # pb_robot.viz.draw_pose(pose2)
                finger_length=0.1034
                grasp_pose = pb_robot.geometry.Pose(grasp_point, pb_robot.geometry.euler_from_quat(grasp_quat))
                grasp_pose = pb_robot.geometry.multiply(grasp_pose, pb_robot.geometry.Pose(pb_robot.geometry.Point(z=-finger_length))) # FINGER_LENGTH

                collision = self.sim_client.pb_check_grasp_collision(grasp_pose)
                
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

    
if __name__ == '__main__':
    ycb_objects_names = [name for name in os.listdir(ycb_objects.getDataPath()) if 'Ycb' in name] 
    ycb_objects_names = ['YcbPowerDrill']
    labeler = GraspStabilityChecker(stability_direction='all', label_type='relpose')

    ycb_object_name = random.choice(ycb_objects_names)
    graspable_body = GraspableBody(ycb_name=ycb_object_name, com=(0.05, 0.05, 0), mass=1., friction=0.5)

    grasp_sampler = GraspSampler(graspable_body=graspable_body, antipodal_tolerance=30, show_pybullet=False)
    n_samples = 10
    grasps = []
    for lx in range(0, n_samples):
        print('Sampling %d/%d...' % (lx, n_samples))
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasps.append(grasp)
    grasp_sampler.disconnect()
    # show_grasps(grasps)
    
    labels = []
    for lx, grasp in enumerate(grasps):
        print('Labeling %d/%d...' % (lx, n_samples))
        labels.append(labeler.get_label(grasp, show_pybullet=False))

    sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
    sim_client.tm_show_grasps(grasps, labels)#, fname='test.png')

    import IPython
    IPython.embed()
