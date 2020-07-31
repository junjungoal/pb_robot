import numpy, math
from tsr.tsrlibrary import TSRFactory
from tsr.tsr import TSR, TSRChain

def handle_grasp(tool, push_distance=0.0,
                width_offset=0.008,
                toolLength=0.08,
                **kw_args):
    """
    @param tool The tool to grasp
    @param push_distance The distance to push before grasping
    """
    ee_to_palm_distance = 0.1 #0.098 
    lateral_offset=ee_to_palm_distance + push_distance

    T0_w = tool.get_transform()
    chain_list = []

    # Base of tool - horizontal (opposite side of head)
    Tw_e_horiz = numpy.array([[ 0., 0., -1.,  lateral_offset],
                              [ 0., 1.,  0., 0.0],
                              [ 1., 0.,  0., 0.0], 
                              [ 0., 0.,  0., 1.]])
    Bw_h = numpy.zeros((6,2))
    Bw_h[0, :] = [-0.03, 0]
    Bw_h[1, :] = [-width_offset, width_offset]
    Bw_h[2, :] = [-width_offset, width_offset]
    Bw_h[4, :] = [-math.pi/2.0, math.pi/2.0] 
    horiz_tsr = TSR(T0_w=T0_w, Tw_e=Tw_e_horiz, Bw=Bw_h)
    grasp_chain_horiz = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=horiz_tsr)
    chain_list += [ grasp_chain_horiz ] 

    # Base of tool - vertical (opposite side of head)
    Tw_e_vert = numpy.array([[ 0.,  0., -1., lateral_offset],
                             [ 1.,  0.,  0., 0.0],
                             [ 0., -1.,  0., 0.0],
                             [ 0.,  0.,  0., 1.]])
    Bw_v = numpy.zeros((6,2))
    Bw_v[0, :] = [-0.03, 0]
    Bw_v[1, :] = [-width_offset, width_offset]
    Bw_v[2, :] = [-width_offset, width_offset]
    Bw_v[5, :] = [-math.pi/2.0, math.pi/2.0]
    vert_tsr = TSR(T0_w=T0_w, Tw_e=Tw_e_vert, Bw = Bw_v)
    grasp_chain_vert = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=vert_tsr)
    chain_list += [ grasp_chain_vert ]
 
    # Top and Bottom sides
    Tw_e_side1 = numpy.array([[ 1., 0.,  0., 0.0],
                              [ 0.,-1.,  0., 0.0],
                              [ 0., 0., -1., lateral_offset],
                              [ 0., 0.,  0., 1.]])

    Tw_e_side2 = numpy.array([[ 1., 0., 0., 0.0],
                              [ 0., 1., 0., 0.0],
                              [ 0., 0., 1., -lateral_offset],
                              [ 0., 0., 0., 1.]])
    Bw_side = numpy.zeros((6,2))
    Bw_side[0, :] = [-toolLength, 0.0]
    Bw_side[2, :] = [-width_offset, width_offset]
    Bw_side[4,:] = [-math.pi/4.0, math.pi/4.0]
    side_tsr1 = TSR(T0_w = T0_w, Tw_e = Tw_e_side1, Bw = Bw_side)
    grasp_chain_side1 = TSRChain(sample_start=False, sample_goal=True,
                                constrain=False, TSR=side_tsr1)
    chain_list += [ grasp_chain_side1 ]
    side_tsr2 = TSR(T0_w = T0_w, Tw_e = Tw_e_side2, Bw = Bw_side)
    grasp_chain_side2 = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=side_tsr2)
    chain_list += [ grasp_chain_side2 ] 


    # Two side faces
    Tw_e_bottom1 = numpy.array([[ 1., 0.,  0., 0.],
                                [ 0., 0., -1., lateral_offset],
                                [ 0., 1.,  0., 0.0],
                                [ 0., 0.,  0., 1.]])

    Tw_e_bottom2 = numpy.array([[ 1.,  0.,  0., 0.],
                                [ 0.,  0.,  1., -lateral_offset],
                                [ 0., -1.,  0., 0.0],
                                [ 0.,  0.,  0., 1.]])
    Bw_topbottom = numpy.zeros((6,2))
    Bw_topbottom[0, :] = [-toolLength, 0.0]
    Bw_topbottom[2, :] = [-width_offset, width_offset]
    Bw_topbottom[5, :] = [-math.pi/4.0, math.pi/4.0]
    bottom_tsr1 = TSR(T0_w = T0_w, Tw_e = Tw_e_bottom1, Bw = Bw_topbottom)
    grasp_chain_bottom1 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=bottom_tsr1)
    chain_list += [ grasp_chain_bottom1 ]

    bottom_tsr2 = TSR(T0_w = T0_w, Tw_e = Tw_e_bottom2, Bw = Bw_topbottom)
    grasp_chain_bottom2 = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=bottom_tsr2)
    chain_list += [ grasp_chain_bottom2 ]

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

def cylinder_handle_grasp(tool, push_distance=0.0,
                width_offset=0.0,
                **kw_args):
    """
    @param tool The tool to grasp
    @param push_distance The distance to push before grasping
    """
    ee_to_palm_distance = 0.098 
    lateral_offset=ee_to_palm_distance + push_distance

    T0_w = tool.get_transform()
    chain_list = []

    # Base of tool
    Tw_e_head = numpy.array([[ 0., 0., -1.,  lateral_offset],
                             [ 0., 1.,  0., 0.0],
                             [ 1., 0.,  0., 0.0], 
                             [ 0., 0.,  0., 1.]])
    Bw_yz = numpy.zeros((6,2))
    Bw_yz[0, :] = [-0.03, 0]
    Bw_yz[1, :] = [-width_offset, width_offset]
    Bw_yz[2, :] = [-width_offset, width_offset]
    Bw_yz[3, :] = [-math.pi, math.pi]
    Bw_yz[4, :] = [-math.pi/2.0, math.pi/2.0]
    head_tsr = TSR(T0_w = T0_w, Tw_e = Tw_e_head, Bw = Bw_yz)
    grasp_chain_head = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=head_tsr)
    chain_list += [ grasp_chain_head ] 


    # Sides
    Tw_e_side = numpy.array([[ 1., 0.,  0., 0.0],
                             [ 0.,-1.,  0., 0.0],
                             [ 0., 0., -1., lateral_offset],
                             [ 0., 0.,  0., 1.]])
    Bw_side = numpy.zeros((6,2))
    Bw_side[0, :] = [-0.08, 0.0]
    Bw_side[1, :] = [-width_offset, width_offset]
    Bw_side[3, :] = [-math.pi, math.pi]
    Bw_side[4,:] = [-math.pi/4.0, math.pi/4.0]
    side_tsr = TSR(T0_w = T0_w, Tw_e = Tw_e_side, Bw = Bw_side)
    grasp_chain_side = TSRChain(sample_start=False, sample_goal=True,
                                constrain=False, TSR=side_tsr)
    chain_list += [ grasp_chain_side ]

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
