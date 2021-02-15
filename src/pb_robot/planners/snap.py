#/usr/bin/env python
# -*- coding: utf-8 -*-

'''Snap Planner between two configurations. Straight line in configuration space'''

import numpy
from . import util

class SnapPlanner(object):
    '''Snap Planner - maintaining class structure because may be useful later when all formatting'''
    def __init__(self):
        self.checkRate = 0.05

    def PlanToConfiguration(self, manip, start_q, goal_q, obstacles=None, check_upwards=False):
        '''Plan from one joint location (start) to another (goal_config)
        optional constraints.
        @param manip Robot arm to plan with
        @param start_q Joint pose to start from
        @param goal_q Joint pose to plan to
        @return joint trajectory or None if plan failed'''

        # Check if start and goal are collision-free
        if (not manip.IsCollisionFree(start_q, obstacles=obstacles)) or (not manip.IsCollisionFree(goal_q, obstacles=obstacles)):
            return None

        # Check intermediate points for collisions
        cdist = util.cspaceLength([start_q, goal_q])
        count = int(cdist / self.checkRate) # Check every 0.1 distance (a little arbitrary)
        # This should be a short path.
        if cdist > 1.5:
            print('[Snap Planner] Too long:', cdist)
            return None

        # linearly interpolate between that at some step size and check all those points
        interp = [numpy.linspace(start_q[i], goal_q[i], count+1).tolist() for i in range(len(start_q))]
        middle_qs = numpy.transpose(interp)[1:-1] # Remove given points
        if not  all((manip.IsCollisionFree(m, obstacles=obstacles) for m in middle_qs)):
            return None

        if check_upwards:
            start_z = manip.ComputeFK(start_q)[2, 3]
            for mq in middle_qs:
                mid_z = manip.ComputeFK(mq)[2, 3]
                if mid_z < start_z:
                    print('[Snap Planner] Trajectory goes down first.')
                    return None

        # Have collision-free path. For now just return two points
        return [start_q, goal_q]
