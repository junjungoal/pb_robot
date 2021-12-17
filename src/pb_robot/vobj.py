import pb_robot
import numpy as np
import time
import sys
from pb_robot.transformations import quaternion_from_matrix
from pb_robot.tsrs.panda_box import ComputePrePose
from pb_robot.planners.util import cspaceLength

from scipy.spatial.transform import Rotation as Rot

class BodyPose(object):
    def __init__(self, body, pose):
        self.body = body
        self.pose = pose
    def __repr__(self):
        return 'p{}'.format(id(self) % 1000)

class RelativePose(object):
    # For cap and bottle, cap is body1, bottle is body2
    #body1_body2F = np.dot(np.linalg.inv(body1.get_transform()), body2.get_transform())
    #relative_pose = pb_robot.vobj.RelativePose(body1, body2, body1_body2F)
    def __init__(self, body1, body2, pose):
        self.body1 = body1
        self.body2 = body2
        self.pose = pose #body1_body2F
    def computeB1GivenB2(self, body2_pose):
        return np.linalg.inv(np.dot(self.pose, np.linalg.inv(body2_pose)))
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
    def simulate(self, timestep, obstacles=[]):
        if self.body.get_name() in self.manip.grabbedObjects:
            # Object grabbed, need to release
            self.manip.hand.Open()
            self.manip.Release(self.body)
        else:
            # Object not grabbed, need to grab
            #self.manip.hand.Close()
            self.manip.hand.MoveTo(0.01)
            self.manip.Grab(self.body, self.grasp_objF)
    def execute(self, realRobot=None, obstacles=[]):
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
    def __init__(self, manip, path, speed=0.6):
        self.manip = manip
        self.path = path
        self.speed = speed
    def simulate(self, timestep, obstacles=[], control=False):
        curr_q = self.manip.GetJointValues()
        start_q = self.path[0]
        for q1, q2 in zip(curr_q, start_q):
            if np.abs(q1-q2) > 0.01:
                print('Actual:', curr_q)
                print('From planner:', start_q)
                input('ERROR')
        self.manip.ExecutePositionPath(self.path, timestep=timestep)
    def execute(self, realRobot=None, obstacles=[]):
        print('Setting speed:', self.speed)
        realRobot.set_joint_position_speed(self.speed)
        dictPath = [realRobot.convertToDict(q) for q in self.path]
        realRobot.execute_position_path(dictPath)
    def __repr__(self):
        return 'j_path{}'.format(id(self) % 1000)

class JointSpacePushPath(object):
    def __init__(self, manip, path, speed=0.6):
        self.manip = manip
        self.path = path
        self.speed = speed
    def simulate(self, timestep, obstacles=[], control=False):
        curr_q = self.manip.GetJointValues()
        start_q = self.path[0]
        for q1, q2 in zip(curr_q, start_q):
            if np.abs(q1-q2) > 0.01:
                print('Actual:', curr_q)
                print('From planner:', start_q)
                input('ERROR')
        self.manip.ExecutePositionPath(self.path, timestep=timestep, control=control, duration=10)
    def execute(self, realRobot=None, obstacles=[]):
        # TODO: when generating a push path, verify that the distance between each
        # individual joint configuration is close. Don't want to break the panda
        # by having it jump around too far in a single time step
        input('You are about to execute a Push Path on the robot. Have you verified that \
        the given joint space trajectory is smooth??')
        print('Setting speed:', self.speed)
        realRobot.set_joint_position_speed(self.speed)
        dictPath = [realRobot.convertToDict(q) for q in self.path]
        realRobot.execute_position_path(dictPath)
    def __repr__(self):
        return 'push_path{}'.format(id(self) % 1000)

class MoveToTouch(object):
    def __init__(self, manip, start, end, grasp, block, use_wrist_camera=False):
        self.manip = manip
        self.start = start
        self.end = end
        self.use_wrist_camera = use_wrist_camera
        self.block_name = block.readableName
        self.block = block
        self.grasp = grasp

    def get_pose_from_wrist(self):
        import rospy
        from panda_vision.srv import GetBlockPosesWrist
        rospy.wait_for_service('get_block_poses_wrist')
        _get_block_poses_wrist = rospy.ServiceProxy('get_block_poses_wrist', GetBlockPosesWrist)
        try:
            poses = _get_block_poses_wrist().poses
        except:
            print('[MoveToTouch]: Service call to get block poses failed during approach. Exiting.')
            sys.exit()
        for named_pose in poses:
            if named_pose.block_id in self.block_name.split('_')[-1]:
                pose = named_pose.pose.pose
                position = (pose.position.x, pose.position.y, pose.position.z) # self.block.get_dimensions()[2]/2.
                orientation = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
                print('[MoveToTouch]: Block estimated position:', position)
                return (position, orientation)
        print('[MoveToTouch]: Desired block not found. Exiting.')
        return None

    def recalculate_qs(self, start_q, pose, obstacles):
        """ Given that the object is at a new pose, recompute the approac
        configurations. Throw an error if the new pose is significantly
        different from the old one. """
        obj_worldF = pb_robot.geometry.tform_from_pose(pose)
        grasp_worldF = np.dot(obj_worldF, self.grasp.grasp_objF)
        approach_tform = ComputePrePose(grasp_worldF, [0, 0, -0.1], 'gripper')

        for _ in range(10):
            start_tform = self.manip.ComputeFK(start_q)
            q_approach = self.manip.ComputeIK(approach_tform, seed_q=start_q)
            if (q_approach is None):
                print('[MoveToTouch] Failed to find approach IK.')
                continue
            if not self.manip.IsCollisionFree(q_approach, debug=True, obstacles=obstacles):
                print('[MoveToTouch] Approach IK in collision.')
                continue

            q_grasp = self.manip.ComputeIK(grasp_worldF, seed_q=q_approach)
            if (q_grasp is None):
                print('[MoveToTouch] Failed to find grasp IK.')
                continue
            if not self.manip.IsCollisionFree(q_grasp, debug=True, obstacles=obstacles):
                print('[MoveToTouch] Grasp IK in collision.')
                continue
            path1 = self.manip.snap.PlanToConfiguration(self.manip, start_q, q_approach, obstacles=obstacles)
            path2 = self.manip.snap.PlanToConfiguration(self.manip, q_approach, q_grasp, obstacles=obstacles)
            if path1 is None:
                print(f'[MoveToTouch]: Adjust trajectory invalid.')
                continue
            if path2 is None:
                print(f'[MoveToTouch]: Approach trajectory invalid.')
                continue
            # approach_dist = cspaceLength([q_approach, q_grasp])
            # # This should be a short path.
            # if adjust_dist > 1.5 or approach_dist > 1.5:
            #     print(f'[MoveToTouch]: Trajectory too long. Adjust: {adjust_dist}\t Approach: {approach_dist}')
            #     continue

            print('[MoveToTouch]: Start Transform')
            print(start_tform)

            print('[MoveToTouch]: Approach Transform')
            print(approach_tform)

            print('[MoveToTouch]: Grasp Transform')
            print(grasp_worldF)
            return q_approach, q_grasp

        print('[MoveToTouch]: Could not find adjusted IK solution.')
        return None

    def simulate(self, timestep, obstacles=[], sim_noise=False):
        # When use_wrist_camera is enabled in simulation there is no vision
        # system, so we sample a perturbation of the current block pose
        if self.use_wrist_camera and sim_noise:
            # sample a new pose for the object with 1cm of position noise
            # and 10 degrees of rotation noise about the vertical axis
            pos, orn = self.block.get_pose()
            new_pos = pos + np.random.randn(3) * 0.0
            new_orn = Rot.from_quat(orn) * Rot.from_euler('z', np.random.randn() * 0.0)
            new_pose = (new_pos, new_orn.as_quat())
            print('NEW POSE', new_pose)
            # get the current position of the bot and calculate the new pregrasp pose
            start_q = self.manip.GetJointValues()
            result = self.recalculate_qs(start_q, new_pose, obstacles=obstacles)
            if result is None:
                from tamp.misc import ExecutionFailure
                reason = '[MoveToTouch] Failed to find locate and pick up block.'
                print(reason)
                raise ExecutionFailure(reason=reason, fatal=False)
            else:
                self.start, self.end = result
                print('Moving to corrected approach.')
        '''
        import pybullet as p
        while True:
            ans = input('a: for approach, c: for contact, q: to continue')
            if ans in ['a', 'c']:
                if ans == 'a':
                    self.manip.SetJointValues(self.start)
                elif ans == 'c':
                    self.manip.SetJointValues(self.end)
                print('pausing (make sure you are visualizing the correct robot)\nCTRL+C to continue')
                try:
                    while True:
                        p.stepSimulation()
                except KeyboardInterrupt:
                    #import pdb; pdb.set_trace()
                    pass
            else:
                break
        '''
        length = cspaceLength([self.start, self.end])
        print('CSpaceLength:', length)
        self.manip.ExecutePositionPath([self.start, self.end], timestep=timestep)

    def execute(self, realRobot=None, obstacles=[]):
        if self.use_wrist_camera:
            success = False
            for ix in range(3):
                print(f'[MoveToTouch] Attempt {ix+1} to localize block.')
                pose = self.get_pose_from_wrist()
                if pose is None:
                    continue

                start_q = realRobot.convertToList(realRobot.joint_angles())
                result = self.recalculate_qs(start_q, pose, obstacles=obstacles)
                if result is None:
                    continue
                else:
                    self.start, self.end = result
                    success = True
                    break
            if not success:
                from tamp.misc import ExecutionFailure
                reason = '[MoveToTouch] Failed to find locate and pick up block.'
                print(reason)
                raise ExecutionFailure(reason=reason, fatal=False)
                # sys.exit(0)

            print('[MoveToTouch]: Moving to corrected approach.')
            realRobot.set_joint_position_speed(0.2)
            realRobot.move_to_joint_positions(realRobot.convertToDict(self.start))
        print('[MoveToTouch]: Moving to corrected grasp.')
        realRobot.move_to_touch(realRobot.convertToDict(self.end))
    def __repr__(self):
        return 'moveToTouch{}'.format(id(self) % 1000)

class MoveFromTouch(object):
    def __init__(self, manip, end, speed=0.3, use_wrist_camera=False):
        self.manip = manip
        self.end = end
        self.speed = speed
        self.use_wrist_camera = use_wrist_camera
    def simulate(self, timestep, obstacles=[]):
        start = self.manip.GetJointValues()
        self.manip.ExecutePositionPath([start, self.end], timestep=timestep)
    def recompute_backoff(self, realRobot, obstacles):
        curr_q = realRobot.convertToList(realRobot.joint_angles())
        grasp_tform = self.manip.ComputeFK(curr_q)
        backoff_tform = ComputePrePose(grasp_tform, [0, 0, 0.1], 'global')

        print('[MoveFromTouch] Computing new backoff position')
        for ax in range(10):
            print(f'[MoveFromTouch] Attempt {ax} for backoff.')
            backoff_q = self.manip.ComputeIK(backoff_tform, seed_q=curr_q)
            if (backoff_q is None):
                print('[MoveFromTouch] Failed to find backoff IK.')
                continue
            if not self.manip.IsCollisionFree(backoff_q, debug=True):
                print('[MoveFromTouch] Backoff IK in collision.')
                continue
            path1 = self.manip.snap.PlanToConfiguration(self.manip, curr_q, backoff_q, obstacles=obstacles, check_upwards=True)
            path2 = self.manip.snap.PlanToConfiguration(self.manip, backoff_q, self.end, obstacles=obstacles)
            if path1 is None:
                print(f'[MoveFromTouch]: Backoff trajectory invalid.')
                continue
            if path2 is None:
                print(f'[MoveFromTouch]: Readjust trajectory invalid.')
                continue
            realRobot.move_from_touch(realRobot.convertToDict(backoff_q))
            return

    def execute(self, realRobot=None, obstacles=[]):
        realRobot.set_joint_position_speed(self.speed)
        if self.use_wrist_camera:
            self.recompute_backoff(realRobot, obstacles)
        realRobot.move_from_touch(realRobot.convertToDict(self.end))

        # Check if grasp was missed by checking gripper opening distances
        if self.use_wrist_camera:
            min_gripper_width = 0.015
            gripper_pos = realRobot.hand.joint_positions()
            grip_width = 0.5*(
                gripper_pos['panda_finger_joint1'] +
                gripper_pos['panda_finger_joint2'])
            if grip_width < min_gripper_width:
                from tamp.misc import ExecutionFailure
                realRobot.hand.open()
                raise ExecutionFailure(
                    reason="No block detected in gripper",
                    fatal=False)

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
    def simulate(self, timestep):
        q = self.manip.GetJointValues()
        if np.linalg.norm(np.subtract(q, self.start_q)) > 1e-3:
            raise IOError("Incorrect starting position")
        # Going to fake cartesian impedance control
        for i in range(len(self.ee_path)):
            q = self.manip.ComputeIK(self.ee_path[i], seed_q=q)
            self.manip.SetJointValues(q)
            time.sleep(self.timestep)
    def execute(self, realRobot=None):
        import quaternion
        #FIXME adjustment based on current position..? Need to play with how execution goes.
        sim_start = self.ee_path[0, 0:3, 3]
        real_start = realRobot.endpoint_pose()['position']
        sim_real_diff = np.subtract(sim_start, real_start)
        input('Cartesian path?')
        poses = []
        for transform in self.ee_path:
            #quat = FrankaQuat(pb_robot.geometry.quat_from_matrix(transform[0:3, 0:3]))
            quat = quaternion_from_matrix(transform[0:3,0:3])
            xyz = transform[0:3, 3] - sim_real_diff
            poses += [{'position': xyz, 'orientation': quat}]
        realRobot.execute_cart_impedance_traj(poses, stiffness=self.stiffness)

    def __repr__(self):
        return 'ci_path{}'.format(id(self) % 1000)
