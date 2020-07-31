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
    def __init__(self, body, grasp_objF, manip, r=0.02, mu=0.5):
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
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class ViseGrasp(object):
    def __init__(self, body, grasp_objF, hand):
        self.body = body
        self.grasp_objF = grasp_objF #Tform
        self.hand = pb_robot.panda.PandaHand(hand.id)
    def simulate(self):
        if self.body.get_name() in self.hand.grabbedObjects:
            # Object grabbed, need to release
            self.hand.Open()
            self.hand.Release(self.body)
        else:
            # Object not grabbed, need to grab
            self.hand.Close()
            self.hand.Grab(self.body, self.grasp_objF)
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
    def __repr__(self):
        return 'j_path{}'.format(id(self) % 1000)

class MoveToTouch(object):
    #TODO do I want to change the input?
    def __init__(self, manip, start, end):
        self.manip = manip
        self.start = start
        self.end = end
    def simulate(self):
        self.manip.ExecutePositionPath([self.start, self.end])
    def __repr__(self):
        return 'move_touch{}'.format(id(self) % 1000)

class CartImpedPath(object):
    def __init__(self, manip, start_q, ee_path, stiffness=None, timestep=0.05):
        if stiffness is None:
            stiffness = [400]*6
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
    def __repr__(self):
        return 'ci_path{}'.format(id(self) % 1000)


class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = body.get_movable_joints()
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments
    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])
    def iterator(self):
        for i, configuration in enumerate(self.path):
            self.body.arm.SetJointValues(configuration)
            for grasp in self.attachments:
                #grasp.assign()
                #TODO move this to only grasp once. Also, need to be able to release
                self.body.arm.Grab(grasp.body, pb_robot.geometry.tform_from_pose(grasp.grasp_pose))
            yield i
    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            pb_robot.utils.enable_real_time()
        else:
            pb_robot.utils.disable_real_time()
        for values in self.path:
            for _ in pb_robot.utils.joint_controller(self.body, self.joints, values):
                pb_robot.utils.enable_gravity()
                if not real_time:
                    pb_robot.utils.step_simulation()
                time.sleep(dt)
    def refine(self, num_steps=0):
        return self.__class__(self.body, pb_robot.planning.refine_path(self.body, self.joints, self.path, num_steps), self.joints, self.attachments)
    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)
    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path), len(self.attachments))

class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link
    def bodies(self):
        return {self.body, self.robot}
    def iterator(self, **kwargs):
        return []
    def refine(self, **kwargs):
        return self
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)

class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        pb_robot.grasp.add_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Detach(self.body, self.robot, self.link)

class Detach(ApplyForce):
    def control(self, **kwargs):
        pb_robot.grasp.remove_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Attach(self.body, self.robot, self.link)

class Command(object):
    def __init__(self, body_paths):
        self.body_paths = body_paths

    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                raw_input(msg)
                #print(msg)

    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            raw_input("next path?")
            for j in body_path.iterator():
                #time.sleep(time_step)
                pb_robot.utils.wait_for_duration(time_step)

    def control(self, real_time=False, dt=0): # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)

    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])

    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])

    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)
