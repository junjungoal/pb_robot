import pb_robot
import numpy
import time

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
    def __init__(self, body, grasp_objF, manip, r=0.0085, mu=None):
        self.body = body
        self.grasp_objF = grasp_objF #Tform
        self.manip = manip
        self.r = r
        self.mu = mu
    def simulate(self):
        if self.body.get_name() in self.manip.grabbedObjects:
            # Object grabbed, need to release
            self.manip.hand.Open()
            self.manip.Release(self.body)
        else:
            # Object not grabbed, need to grab
            self.manip.hand.Close()
            self.manip.Grab(self.body, self.grasp_objF)
    def execute(self, realRobot=None, realHand=None):
        #FIXME need to confirm logic and widths
        hand_pose = realHand.joint_positions()
        if hand_pose['panda_finger_joint_1'] < 0.08:
            realHand.Open()
        else:
            # For now, grasp with fixed N (40, used in mechanics)
            realHand.grasp(0.01, 40)
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class ViseGrasp(object):
    def __init__(self, body, grasp_objF, hand):
        self.body = body
        self.grasp_objF = grasp_objF #Tform
        self.hand = pb_robot.wsg50_hand.WSG50Hand(hand.id)
    def simulate(self):
        if self.body.get_name() in self.hand.grabbedObjects:
            # Object grabbed, need to release
            self.hand.Open()
            self.hand.Release(self.body)
        else:
            # Object not grabbed, need to grab
            self.hand.Close()
            self.hand.Grab(self.body, self.grasp_objF)
    def execute(self, realRobot=None, realHand=None):
        raise NotImplementedError('Havent conected WSG50 hand yet')
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
    def __init__(self, manip, path):
        self.manip = manip
        self.path = path
    def simulate(self):
        self.manip.ExecutePositionPath(self.path)
    def execute(self, realRobot=None, realHand=None):
        #FIXME in execute_position_path compare time_so_far to timeout
        dictPath = [realRobot.convertToDict(q) for q in path]
        realRobot.execute_position_path(dict_path)
    def __repr__(self):
        return 'j_path{}'.format(id(self) % 1000)

class MoveToTouch(object):
    def __init__(self, manip, start, end):
        self.manip = manip
        self.start = start
        self.end = end
    def simulate(self):
        self.manip.ExecutePositionPath([self.start, self.end])
    def execute(self, realRobot=None, realHand=None):
        realRobot.move_to_touch(realRobot.convertToDict(self.start))
    def __repr__(self):
        return 'move_touch{}'.format(id(self) % 1000)

class FrankaQuata(object):
    def __init__(self, quat):
        self.x = quat[0]
        self.y = quat[1]
        self.z = quat[2]
        self.w = quat[3]


class CartImpedPath(object):
    def __init__(self, manip, start_q, ee_path, stiffness=None, timestep=0.05):
        if stiffness is None: 
            stiffness = [400, 400, 400, 40, 40, 40]
        elif isinstance(stiffness, int) or isinstance(stiffness, float):
            stiffness = [stiffness]*6
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
        for i in xrange(len(self.ee_path)):
            q = self.manip.ComputeIK(self.ee_path[i], seed_q=q)
            self.manip.SetJointValues(q)
            time.sleep(self.timestep)
    def execute(self, realRobot=None, realHand=None):
        #FIXME adjustment based on current position..? Need to play with how execution goes.
        poses = []
        for transform in ee_path:
            quat = FrankaQuat(pb_robot.geometry.quat_from_matrix(transform[0:3, 0:3]))
            poses += [{'position': transform[0:3, 3], 'orientation': quat}]
        realRobot.execute_cart_impedance_traj(poses, stiffness=self.stiffness)

    def __repr__(self):
        return 'ci_path{}'.format(id(self) % 1000)
