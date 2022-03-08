from datetime import timedelta
import time
import numpy
import pybullet as p
import pb_robot

class PandaControls(object):
    def __init__(self, arm):
        self.arm = arm
        print("Set up")

    def clampTorque(self, tau_d):
        tau_limit = [87, 87, 87, 87, 50, 50, 50] #robot.arm.torque_limits
        for i in range(len(tau_d)):
            tau_d[i] = pb_robot.helper.clip(tau_d[i], -tau_limit[i], tau_limit[i])
        return tau_d

    def positionControl(self, q, threshold=0.1):
        n = len(q)
        while True:
            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID,
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=q,
                            targetVelocities=[0]*n,
                            forces=self.arm.torque_limits,
                            positionGains=[0.1]*n,
                            velocityGains=[1]*n)
            p.stepSimulation()
            time.sleep(0.01)

            if numpy.linalg.norm(numpy.subtract(self.arm.GetJointValues(), q)) < threshold:
                break

    def moveToTouch(self, q_desired):
        n = len(q_desired)
        p.stepSimulation()
        ft_past = p.getJointState(self.arm.bodyID, 8)[2]
        i = 0

        while True:
            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID,
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=q_desired,
                        targetVelocities=[0]*n,
                        forces=self.arm.torque_limits,
                        positionGains=[0.1]*n,
                        velocityGains=[1]*n)
            p.stepSimulation()
            time.sleep(0.01)

            ft = self.arm.GetFTWristReading() #p.getJointState(robot.id, robot.ft_joint.jointID)[2]
            # So the in collision check seems incredibly accurate
            # The FT sensor check produces false positives (and stops trigging once in collision
            # So for now its check FT and then confirm with is collision. 
            if (numpy.linalg.norm(numpy.subtract(ft, ft_past)) > 100):
                if not self.arm.IsCollisionFree(self.arm.GetJointValues()):
                    break

            if numpy.linalg.norm(numpy.subtract(self.arm.GetJointValues(), q_desired)) < 0.01:
                if self.arm.IsCollisionFree(self.arm.GetJointValues()):
                    raise RuntimeError("MoveToTouch did not end in contact")

            ft_past = copy.deepcopy(ft)

    def forceControl(self, wrench_target, timeExerted):
        # wrench target needs to be a column, not row vector
        # Time exerted is in seconds
        wrench_desired = numpy.zeros((6, 1))
        gain = 0.1 #Was 0.01
        p.setGravity(0, 0, -9.81)
        # To go down, z should be negative! 

        fts = [0]*1000
        jfs = [0]*1000
        counts = int(timeExerted / 0.01) # 0.01 is rate

        for i in range(counts):
            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID, p.VELOCITY_CONTROL,
                                        targetVelocities=[0]*7, forces=[35]*7)

            jacobian = self.arm.GetJacobian(self.arm.GetJointValues())
            # Feedforward. Commented out PI control because gains were 0 
            tau_d = numpy.matmul(numpy.transpose(jacobian), wrench_desired)
            tau_cmd = self.clampTorque(tau_d)

            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID,
                        controlMode=p.TORQUE_CONTROL,
                        forces=tau_cmd)

            wrench_desired = gain * wrench_target + (1 - gain) * wrench_desired
            p.stepSimulation()
            time.sleep(0.01)

            fts[i] = self.arm.GetFTWristReading()[2]
            jfs[i] = numpy.matmul(self.arm.GetJacobian(self.arm.GetJointValues()), self.arm.GetJointTorques())[2]
            # Large difference between commanded torque and observed torque (with or without gravity)
            # Is this true on the real robot? (i dont know!)

    def jointImpedance(self, q_d_target, stiffness_params):
        # Seems to work but requires huuuge stiffness coefficients to get to pose (much higher than on real robot)
        p.setGravity(0, 0, -9.81)

        dq_d_target = numpy.zeros(len(q_d_target))
        stiffness_target = numpy.diag(stiffness_params)
        damping_target = numpy.diag(2.0*numpy.sqrt(stiffness_params))
        stiffness = numpy.eye(7)
        damping = numpy.eye(7)
        gain = 0.1

        q_d = self.arm.GetJointValues()
        dq_d = self.arm.GetJointVelocities()
        N = len(q_d)

        prev_q = self.arm.GetJointValues()
        diff_count = 0
        eps = 1e-4

        while True:
            diff = numpy.linalg.norm(numpy.subtract(self.arm.GetJointValues(), prev_q), ord=2)
            if diff < eps:
                diff_count += 1
            else:
                diff_count = 0
            if diff_count > 20:
                break

            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID, p.VELOCITY_CONTROL,
                                        targetVelocities=[0]*7, forces=[40]*7)

            tau_d = [0]*N
            prev_q = self.arm.GetJointValues()
            q = self.arm.GetJointValues()
            dq = self.arm.GetJointVelocities()
            coriolis = self.arm.GetCoriolosMatrix(q, dq)

            for i in range(N):
                tau_d[i] = coriolis[i] + (stiffness[i,i]*(q_d[i] - q[i])) + (damping[i,i]*(dq_d[i] - dq[i]))

            tau_cmd = self.clampTorque(tau_d)
            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau_cmd)

            stiffness = gain * stiffness_target + (1 - gain) * stiffness
            damping = gain * damping_target + (1 - gain) * damping
            q_d = gain * q_d_target + (1 - gain) * q_d
            dq_d = gain * dq_d_target + (1 - gain) * dq_d

            p.stepSimulation()
            time.sleep(0.01)

    def cartImpedance(self, pose_d_target, stiffness_params):
        p.setGravity(0, 0, -9.81)

        position_d_target = pose_d_target[0:3, 3]
        ori_d_target = pb_robot.geometry.quat_from_matrix(pose_d_target[0:3, 0:3])
        stiffness_target = numpy.diag(stiffness_params)
        damping_target = numpy.diag(2.0*numpy.sqrt(stiffness_params))

        stiffness = numpy.eye(6)
        damping = numpy.eye(6)
        position_d = self.arm.GetEETransform()[0:3, 3]
        ori_d = pb_robot.geometry.quat_from_matrix(self.arm.GetEETransform()[0:3, 0:3])
        gain = 0.1

        prev_q = self.arm.GetJointValues()
        diff_count = 0
        eps = 1e-4

        while True:
            diff = numpy.linalg.norm(numpy.subtract(self.arm.GetJointValues(), prev_q), ord=2)
            if diff < eps:
                diff_count += 1
            else:
                diff_count = 0
            if diff_count > 20:
                break

            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID, p.VELOCITY_CONTROL,
                                        targetVelocities=[0]*7, forces=[30]*7)

            # Compute Pose Error
            error = numpy.zeros((6))
            current_pose = self.arm.GetEETransform()
            position_error = current_pose[0:3, 3] - position_d
            error[0:3] = position_error

            current_ori = pb_robot.geometry.quat_from_matrix(current_pose[0:3, 0:3])
            # Compute different quaternion
            error_ori_quat = pb_robot.transformations.quaternion_multiply(current_ori, pb_robot.transformations.quaternion_inverse(ori_d))
            # Convert to axis angle
            (error_ori_angle, error_ori_axis) = pb_robot.geometry.quatToAxisAngle(error_ori_quat)
            ori_error = numpy.multiply(error_ori_axis, error_ori_angle)
            error[3:6] = ori_error

            q = self.arm.GetJointValues()
            prev_q = self.arm.GetJointValues()
            dq = self.arm.GetJointVelocities()
            jacobian = self.arm.GetJacobian(q)
            tau_task = (numpy.transpose(jacobian)).dot(-stiffness.dot(error) - damping.dot(jacobian.dot(dq)))
            coriolis = self.arm.GetCoriolosMatrix(q, dq)
            tau_d = numpy.add(tau_task, coriolis)
            tau_cmd = self.clampTorque(tau_d)

            p.setJointMotorControlArray(self.arm.bodyID, self.arm.jointsID,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=tau_cmd)

            # Update parameters
            stiffness = gain * stiffness_target + (1 - gain) * stiffness
            damping = gain * damping_target + (1 - gain) * damping
            position_d = gain * position_d_target + (1 - gain) * position_d

            (ori_d_angle, ori_d_axis) = pb_robot.geometry.quatToAxisAngle(ori_d)
            (ori_d_target_angle, ori_d_target_axis) = pb_robot.geometry.quatToAxisAngle(ori_d_target)
            ori_d_axis = gain * ori_d_target_axis + (1 - gain) * ori_d_axis
            ori_d_angle = gain * ori_d_target_angle + (1 - gain) * ori_d_angle
            ori_d = pb_robot.geometry.quat_from_axis_angle(ori_d_axis, ori_d_angle)

            p.stepSimulation()
            time.sleep(0.01)

    def positionControlPath(self, path):
        for i in range(len(path)):
            self.positionControl(path[i])

    def jointImpedancePath(self, path, stiffness_params):
        for i in range(len(path)):
            self.jointImpedance(path[i], stiffness_params)

    def cartImpedancePath(self,path, stiffness_params):
        for i in range(len(path)):
            self.cartImpedance(path[i], stiffness_params)


class FloatingHandControl(object):
    """
    This class is meant to control a floating hand that
    can move through the environment.
    """
    def __init__(self, hand, init_pos, init_orn):
        self.hand = hand
        hand.Open()

        p.resetBasePositionAndOrientation(hand.id, init_pos, init_orn)
        self.cid = p.createConstraint(parentBodyUniqueId=hand.id,
                                      parentLinkIndex=-1,
                                      childBodyUniqueId=-1,
                                      childLinkIndex=-1,
                                      jointType=p.JOINT_FIXED,
                                      jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0],
                                      parentFrameOrientation=[0, 0, 0, 1],
                                      childFramePosition=init_pos,
                                      childFrameOrientation=init_orn)

        self.force_dir = 1  # -1 is close, 1 is open

        p.setJointMotorControl2(hand.id, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(hand.id, 1, p.VELOCITY_CONTROL, force=0)

        c = p.createConstraint(hand.id,
                       1,
                       hand.id,
                       0,
                       jointType=p.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=10000)
        # p.changeDynamics(hand.id, 0, linearDamping=0, angularDamping=0)
        # p.changeDynamics(hand.id, 1, linearDamping=0, angularDamping=0)

    def open(self, wait=False):
        self.force_dir = 1
        self._actuate_fingers(max_force=2, wait=wait)

    def close(self, force, wait=False):
        self.force_dir = -1
        self._actuate_fingers(max_force=force, wait=wait)

    def _actuate_fingers(self, max_force, wait=False):
        exit_count = 0
        while True:
            p.setJointMotorControlArray(bodyUniqueId=self.hand.id,
                                        jointIndices=[0, 1],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[self.force_dir*0.01, self.force_dir*0.01],
                                        forces=[max_force, max_force])

            force0, force1 = p.getJointState(self.hand.id, 0)[3], p.getJointState(self.hand.id, 1)[3]
            p.stepSimulation()
            if wait: 
                time.sleep(0.01)
            # print('Force:', force0, force1)
            if numpy.abs(force0) + 0.01 >= max_force:
                exit_count += 1
            
            if exit_count >= 20:
                break

    def move_to(self, hand_pos, hand_orn, force, wait=False):
        # Use force control to maintain gripper strength.
        p.setJointMotorControl2(self.hand.id, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.hand.id, 1, p.VELOCITY_CONTROL, force=0)
        while True:
            p.setJointMotorControlArray(bodyUniqueId=self.hand.id,
                                        jointIndices=[0, 1],
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=[self.force_dir*force, self.force_dir*force])
            p.changeConstraint(self.cid, hand_pos, jointChildFrameOrientation=hand_orn, maxForce=100)
            p.stepSimulation()
            force = p.getJointState(self.hand.id, 1)[3]
            # print('Grip Force:', force)
            if wait:
                time.sleep(0.01)

            if numpy.linalg.norm(numpy.subtract(self.hand.get_base_link_point(), hand_pos)) < 0.005:
                break

    def set_pose(self, hand_pos, hand_orn):
        p.resetBasePositionAndOrientation(self.hand.id, hand_pos, hand_orn)
        p.changeConstraint(self.cid, hand_pos, jointChildFrameOrientation=hand_orn, maxForce=50)