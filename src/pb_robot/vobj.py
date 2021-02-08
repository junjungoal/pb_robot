import pb_robot
import numpy
import time
from pb_robot.transformations import quaternion_from_matrix
import sys
from pb_robot.tsrs.panda_box import ComputePrePose
from pb_robot.planners.util import cspaceLength

class BodyPose(object):
    def __init__(self, body, pose):
        self.body = body
        self.pose = pose
    def __repr__(self):
        return 'p{}'.format(id(self) % 1000)

class RelativePose(object):
    # For cap and bottle, cap is body1, bottle is body2
    #body1_body2F = numpy.dot(numpy.linalg.inv(body1.get_transform()), body2.get_transform())
    #relative_pose = pb_robot.vobj.RelativePose(body1, body2, body1_body2F)
    def __init__(self, body1, body2, pose):
        self.body1 = body1
        self.body2 = body2
        self.pose = pose #body1_body2F
    def computeB1GivenB2(self, body2_pose):
        return numpy.linalg.inv(numpy.dot(self.pose, numpy.linalg.inv(body2_pose)))
    def __repr__(self):
        return 'rp{}'.format(id(self) % 1000)

class BodyGrasp(object):
    def __init__(self, body, grasp_objF, manip, r=0.0085, mu=None, N=40):
        self.body = body
        self.grasp_objF = grasp_objF #Tform
        self.manip = manip
        self.r = r
        self.mu = mu
        self.N = N
    def simulate(self, timestep):
        if self.body.get_name() in self.manip.grabbedObjects:
            # Object grabbed, need to release
            self.manip.hand.Open()
            self.manip.Release(self.body)
        else:
            # Object not grabbed, need to grab
            #self.manip.hand.Close()
            self.manip.hand.MoveTo(0.01)
            self.manip.Grab(self.body, self.grasp_objF)
    def execute(self, realRobot=None):
        hand_pose = realRobot.hand.joint_positions()
        if hand_pose['panda_finger_joint1'] < 0.039: # open pose
            realRobot.hand.open()
        else:
            realRobot.hand.grasp(0.02, self.N, epsilon_inner=0.1, epsilon_outer=0.1)
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class ViseGrasp(object):
    def __init__(self, body, grasp_objF, hand, N=60):
        self.body = body
        self.grasp_objF = grasp_objF #Tform
        self.hand = pb_robot.wsg50_hand.WSG50Hand(hand.id)
        self.N = N
    def simulate(self):
        if self.body.get_name() in self.hand.grabbedObjects:
            # Object grabbed, need to release
            self.hand.Open()
            self.hand.Release(self.body)
        else:
            # Object not grabbed, need to grab
            #self.hand.Close()
            self.hand.MoveTo(-0.04, 0.04)
            self.hand.Grab(self.body, self.grasp_objF)
    def execute(self, realRobot=None):
        # This is a bad work-around
        realhand = pb_robot.wsg50_hand.WSG50HandReal()
        if realhand.get_width < realhand.openValue:
            realhand.open()
        else:
            realhand.grasp(80, self.N)
    def __repr__(self):
        return 'vg{}'.format(id(self) % 1000)

class BodyConf(object):
    def __init__(self, manip, configuration):
        self.manip = manip
        self.configuration = configuration
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

class BodyWrench(object):
    def __init__(self, body, ft):
        self.body = body
        self.ft_objF = ft
    def __repr__(self):
        return 'w{}'.format(id(self) % 1000)

class JointSpacePath(object):
    def __init__(self, manip, path, speed=0.3):
        self.manip = manip
        self.path = path
        self.speed = speed
    def simulate(self, timestep):
        self.manip.ExecutePositionPath(self.path, timestep=timestep)
    def execute(self, realRobot=None):
        realRobot.set_joint_position_speed(self.speed)
        dictPath = [realRobot.convertToDict(q) for q in self.path]
        realRobot.execute_position_path(dictPath)
    def __repr__(self):
        return 'j_path{}'.format(id(self) % 1000)

class MoveToTouch(object):
    def __init__(self, manip, start, end, grasp, block, use_wrist_camera=False):
        self.manip = manip
        self.start = start
        self.end = end
        self.use_wrist_camera = use_wrist_camera
        self.block_name = block.readableName
        self.block = block
        self.grasp = grasp

    def simulate(self, timestep):
        self.manip.ExecutePositionPath([self.start, self.end], timestep=timestep)

    def get_pose_from_wrist(self):
        import rospy
        from panda_vision.srv import GetBlockPosesWrist
        rospy.wait_for_service('get_block_poses_wrist')
        _get_block_poses_wrist = rospy.ServiceProxy('get_block_poses_wrist', GetBlockPosesWrist)
        try:
            poses = _get_block_poses_wrist().poses
        except:
            print('Service call to get block poses failed during approach. Exiting.')
            sys.exit()
        for named_pose in poses:
            if named_pose.block_id in self.block_name:
                pose = named_pose.pose.pose
                position = (pose.position.x, pose.position.y, pose.position.z) # self.block.get_dimensions()[2]/2.
                orientation = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
                print('Block estimated position:', position)
                return (position, orientation)
        print('Desired block not found. Exiting.')
        sys.exit()

    def recalculate_qs(self, realRobot, pose):
        """ Given that the object is at a new pose, recompute the approac
        configurations. Throw an error if the new pose is significantly
        different from the old one. """
        obj_worldF = pb_robot.geometry.tform_from_pose(pose)
        grasp_worldF = numpy.dot(obj_worldF, self.grasp.grasp_objF)
        approach_tform = ComputePrePose(grasp_worldF, [0, 0, -0.08], 'gripper')

        for _ in range(3):
            start_q = realRobot.convertToList(realRobot.joint_angles())
            start_tform = self.manip.ComputeFK(start_q)
            q_approach = self.manip.ComputeIK(approach_tform, seed_q=start_q)
            if (q_approach is None): continue
            q_grasp = self.manip.ComputeIK(grasp_worldF, seed_q=q_approach)
            if (q_grasp is None): continue
            print('Start Transform')
            print(start_tform)

            print('Approach Transform')
            print(approach_tform)

            print('Grasp Transform')
            print(grasp_worldF)
            return q_approach, q_grasp

    def execute(self, realRobot=None):
        if self.use_wrist_camera:
            pose = self.get_pose_from_wrist()
            self.start, self.end = self.recalculate_qs(realRobot, pose)
            print('Moving to corrected approach.')
            realRobot.move_to_joint_positions(realRobot.convertToDict(self.start))
        print('Moving to corrected grasp.')
        length = cspaceLength([self.start, self.end])
        print('CSpaceLength:', length)
        input('Move?')
        realRobot.move_to_touch(realRobot.convertToDict(self.end))
    def __repr__(self):
        return 'moveToTouch{}'.format(id(self) % 1000)

class MoveFromTouch(object):
    def __init__(self, manip, end, speed=0.3):
        self.manip = manip
        self.end = end
        self.speed = speed
    def simulate(self, timestep):
        start = self.manip.GetJointValues()
        self.manip.ExecutePositionPath([start, self.end], timestep=timestep)
    def execute(self, realRobot=None):
        realRobot.set_joint_position_speed(self.speed)
        realRobot.move_from_touch(realRobot.convertToDict(self.end))
    def __repr__(self):
        return 'moveFromTouch{}'.format(id(self) % 1000)

class FrankaQuat(object):
    def __init__(self, quat):
        self.x = quat[0]
        self.y = quat[1]
        self.z = quat[2]
        self.w = quat[3]
    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.x, self.y, self.z, self.w)


class CartImpedPath(object):
    def __init__(self, manip, start_q, ee_path, stiffness=None, timestep=0.1):
        if stiffness is None:
            stiffness = [400, 400, 400, 40, 40, 40]
        self.manip = manip
        self.ee_path = ee_path
        self.start_q = start_q
        self.stiffness = stiffness
        self.timestep = timestep
    def simulate(self):
        q = self.manip.GetJointValues()
        if numpy.linalg.norm(numpy.subtract(q, self.start_q)) > 1e-3:
            raise IOError("Incorrect starting position")
        # Going to fake cartesian impedance control
        for i in range(len(self.ee_path)):
            q = self.manip.ComputeIK(self.ee_path[i], seed_q=q)
            self.manip.SetJointValues(q)
            time.sleep(self.timestep)
    def execute(self, realRobot=None):
        #FIXME adjustment based on current position..? Need to play with how execution goes.
        sim_start = self.ee_path[0, 0:3, 3]
        real_start = realRobot.endpoint_pose()['position']
        sim_real_diff = numpy.subtract(sim_start, real_start)

        poses = []
        for transform in self.ee_path:
            #quat = FrankaQuat(pb_robot.geometry.quat_from_matrix(transform[0:3, 0:3]))
            quat = quaternion_from_matrix(transform[0:3,0:3])
            xyz = transform[0:3, 3] - sim_real_diff
            poses += [{'position': xyz, 'orientation': quat}]
        realRobot.execute_cart_impedance_traj(poses, stiffness=self.stiffness)

    def __repr__(self):
        return 'ci_path{}'.format(id(self) % 1000)
