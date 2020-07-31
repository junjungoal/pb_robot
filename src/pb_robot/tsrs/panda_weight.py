import numpy, math
from tsr.tsrlibrary import TSRFactory
from tsr.tsr import TSR, TSRChain

def weight_grasp(nut, push_distance=0.0,
                width_offset=0.0,
                **kw_args):
    """
    @param nut The nut to grasp
    @param push_distance The distance to push before grasping
    """
    epsilon = 0.005
    ee_to_palm_distance = 0.105 
    lateral_offset=ee_to_palm_distance + push_distance

    T0_w = nut.get_transform()
    chain_list = []
    
    # Top and Bottom sides
    Tw_e_side1 = numpy.array([[ 1., 0.,  0., 0.0],
                              [ 0.,-1.,  0., 0.0],
                              [ 0., 0., -1., lateral_offset+0.13],
                              [ 0., 0.,  0., 1.]])
    Bw_side = numpy.zeros((6,2))

    Bw_side[0, :] = [-0.05, 0.05]
    side_tsr1 = TSR(T0_w = T0_w, Tw_e = Tw_e_side1, Bw = Bw_side)
    grasp_chain_side1 = TSRChain(sample_start=False, sample_goal=True,
                                constrain=False, TSR=side_tsr1)
    chain_list += [ grasp_chain_side1 ] 

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
