import numpy, math
import pb_robot
from tsr.tsrlibrary import TSRFactory
from tsr.tsr import TSR, TSRChain

def grasp(box, 
          push_distance=0.0,
          width_offset=0.0,
          **kw_args):
    """
    @param box The box to grasp
    @param push_distance The distance to push before grasping
    """
    gripper_width = 0.05  # TODO: Verify this number.

    dimensions = box.get_dimensions()
    ee_to_palm_distance = 0.098
    lateral_offset = ee_to_palm_distance + dimensions[0]/2

    p0_w = box.get_base_link_pose()
    T0_w = pb_robot.geometry.tform_from_pose(p0_w)
    chain_list = []

    # Base of box (opposite side of head)
    Tw_e_front1 = numpy.array([[0., 0., -1., lateral_offset],
                               [0., 1., 0., 0.0],
                               [1., 0., 0., 0.0],
                               [0., 0., 0., 1.]])

    Tw_e_front2 = numpy.array([[0., 0., 1., -lateral_offset],
                               [0., -1., 0., 0.0],
                               [1., 0., 0., 0.0],
                               [0., 0., 0., 1.]])

    Tw_e_front3 = numpy.array([[0., 0., -1., lateral_offset],
                               [1., 0., 0., 0.0],
                               [0., -1., 0., 0.0],
                               [0., 0., 0., 1.]])

    Tw_e_front4 = numpy.array([[0., 0., 1., -lateral_offset],
                               [-1., 0., 0., 0.0],
                               [0., -1., 0., 0.0],
                               [0., 0., 0., 1.]])
    Bw_yz = numpy.zeros((6, 2))
    #Bw_yz[2, :] = [-dimensions[2]/2, dimensions[2]/2]
    front_tsr1 = TSR(T0_w=T0_w, Tw_e=Tw_e_front1, Bw=Bw_yz)
    grasp_chain_front1 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=front_tsr1)

    front_tsr2 = TSR(T0_w=T0_w, Tw_e=Tw_e_front2, Bw=Bw_yz)
    grasp_chain_front2 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=front_tsr2)

    front_tsr3 = TSR(T0_w=T0_w, Tw_e=Tw_e_front3, Bw=Bw_yz)
    grasp_chain_front3 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=front_tsr3)

    front_tsr4 = TSR(T0_w=T0_w, Tw_e=Tw_e_front4, Bw=Bw_yz)
    grasp_chain_front4 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=front_tsr4)

    if dimensions[1] < gripper_width:
        chain_list += [grasp_chain_front1, grasp_chain_front2]
    if dimensions[2] < gripper_width:
        chain_list += [grasp_chain_front3, grasp_chain_front4]
 
    # Top and Bottom sides 
    vertical_offset = ee_to_palm_distance + dimensions[2]/2

    Tw_e_side1 = numpy.array([[ 1., 0.,  0., 0.0],
                              [ 0.,-1.,  0., 0.0],
                              [ 0., 0., -1., vertical_offset], # Added tmp.
                              [ 0., 0.,  0., 1.]])

    Tw_e_side2 = numpy.array([[ 1., 0., 0., 0.0],
                              [ 0., 1., 0., 0.0],
                              [ 0., 0., 1., -vertical_offset],
                              [ 0., 0., 0., 1.]])

    Tw_e_side3 = numpy.array([[ 0., 1.,  0., 0.0],
                              [ 1., 0.,  0., 0.0],
                              [ 0., 0., -1., vertical_offset],
                              [ 0., 0.,  0., 1.]])

    Tw_e_side4 = numpy.array([[ 0., 1., 0., 0.0],
                              [-1., 0., 0., 0.0],
                              [ 0., 0., 1., -vertical_offset],
                              [ 0., 0., 0., 1.]])
    Bw_side = numpy.zeros((6,2))
    # Bw_side[1,:] = [-width_offset, width_offset]
    # Bw_side[1,:] = [-dimensions[1]/2, dimensions[1]/2]
    side_tsr1 = TSR(T0_w = T0_w, Tw_e = Tw_e_side1, Bw = Bw_side)
    grasp_chain_side1 = TSRChain(sample_start=False, sample_goal=True,
                                constrain=False, TSR=side_tsr1)

    side_tsr2 = TSR(T0_w = T0_w, Tw_e = Tw_e_side2, Bw = Bw_side)
    grasp_chain_side2 = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=side_tsr2)
    side_tsr3 = TSR(T0_w = T0_w, Tw_e = Tw_e_side3, Bw = Bw_side)
    grasp_chain_side3 = TSRChain(sample_start=False, sample_goal=True,
                                constrain=False, TSR=side_tsr3)
    side_tsr4 = TSR(T0_w = T0_w, Tw_e = Tw_e_side4, Bw = Bw_side)
    grasp_chain_side4 = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=side_tsr4)

    if dimensions[1] < gripper_width:
        chain_list += [grasp_chain_side1, grasp_chain_side2]

    if dimensions[0] < gripper_width:
        chain_list += [grasp_chain_side3, grasp_chain_side4]

    # Two side faces
    lateral_offset = ee_to_palm_distance + dimensions[1]/2
    Tw_e_bottom1 = numpy.array([[ 0., -1.,  0., 0.],
                                [ 0.,  0., -1., lateral_offset],
                                [ 1.,  0.,  0., 0.],
                                [ 0.,  0.,  0., 1.]])

    Tw_e_bottom2 = numpy.array([[ 0.,  1.,  0., 0.],
                                [ 0.,  0.,  1., -lateral_offset],
                                [ 1.,  0.,  0., 0.],
                                [ 0.,  0.,  0., 1.]])

    Tw_e_bottom3 = numpy.array([[ -1., 0.,  0., 0.],
                                [ 0.,  0., -1., lateral_offset],
                                [ 0.,  -1.,  0., 0.],
                                [ 0.,  0.,  0., 1.]])

    Tw_e_bottom4 = numpy.array([[ 1.,  0.,  0., 0.],
                                [ 0.,  0.,  1., -lateral_offset],
                                [ 0.,  -1.,  0., 0.],
                                [ 0.,  0.,  0., 1.]])
    Bw_topbottom = numpy.zeros((6,2))
    #Bw_topbottom[2,:] = [-dimensions[2]/2,dimensions[2]/2]
    bottom_tsr1 = TSR(T0_w = T0_w, Tw_e = Tw_e_bottom1, Bw = Bw_topbottom)
    grasp_chain_bottom1 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=bottom_tsr1)

    bottom_tsr2 = TSR(T0_w = T0_w, Tw_e = Tw_e_bottom2, Bw = Bw_topbottom)
    grasp_chain_bottom2 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=bottom_tsr2)

    bottom_tsr3 = TSR(T0_w = T0_w, Tw_e = Tw_e_bottom3, Bw = Bw_topbottom)
    grasp_chain_bottom3 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=bottom_tsr3)

    bottom_tsr4 = TSR(T0_w = T0_w, Tw_e = Tw_e_bottom4, Bw = Bw_topbottom)
    grasp_chain_bottom4 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=bottom_tsr4)

    if dimensions[0] < gripper_width:
        chain_list += [grasp_chain_bottom1, grasp_chain_bottom2]

    if dimensions[2] < gripper_width:
        chain_list += [grasp_chain_bottom3, grasp_chain_bottom4]

    # Each chain in the list can also be rotated by 180 degrees around z
    rotated_chain_list = []
    for c in chain_list:
        rval = numpy.pi
        R = numpy.array([[numpy.cos(rval), -numpy.sin(rval), 0., 0.],
                         [numpy.sin(rval),  numpy.cos(rval), 0., 0.],
                         [             0.,               0., 1., 0.],
                         [             0.,               0., 0., 1.]])
        tsr = c.TSRs[0]
        Tw_e = tsr.Tw_e
        Tw_e_new = numpy.dot(Tw_e, R)
        tsr_new = TSR(T0_w = tsr.T0_w, Tw_e=Tw_e_new, Bw=tsr.Bw)
        tsr_chain_new = TSRChain(sample_start=False, sample_goal=True, constrain=False,
                                     TSR=tsr_new)
        rotated_chain_list += [ tsr_chain_new ]
    return chain_list + rotated_chain_list
