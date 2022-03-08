import sys
sys.path.append('src')
import pb_robot
import IPython
import os
import pybullet as p

if __name__ == '__main__':
    pb_robot.utils.connect(use_gui=True)
    pb_robot.utils.disable_real_time()
    pb_robot.utils.set_default_camera()

    # Create robot object 
    robot = pb_robot.panda.Panda() 
 
    # Add floor object 
    floor_file = 'models/short_floor.urdf'
    floor = pb_robot.body.createBody(floor_file)

    cup_file = 'models/dinnerware/cup/cup_small.urdf'
    cup = pb_robot.body.createBody(cup_file)
    cup.set_base_link_point((0.5, 0, pb_robot.placements.stable_z(cup, floor)))

    pan_file = 'models/dinnerware/pan_tefal.urdf'
    pan = pb_robot.body.createBody(pan_file)
    pan.set_base_link_point((0.5, -0.2, pb_robot.placements.stable_z(pan, floor)))

    plate_file = 'models/dinnerware/plate.urdf'
    plate = pb_robot.body.createBody(plate_file)
    plate.set_base_link_point((0.5, 0.2, pb_robot.placements.stable_z(plate, floor)))

    test_model_file = 'src/pb_robot/models/shapenet/test_model/newsdf.sdf'
    test_model = p.loadSDF(test_model_file)
    #test_model = pb_robot.body.createBody(test_model_file)
    #p.resetBasePositionAndOrientation(test_model[0], (0, 0, 0), (0, 0, 0, 1))
    #test_model.set_base_link_point((0.5, 0.4, 0.5))
    #test_model.set_base_link_point((0.5, 0.4, pb_robot.placements.stable_z(test_model, floor)))


    IPython.embed()

    pb_robot.utils.wait_for_user()
    pb_robot.utils.disconnect()