import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt  
from IK_controller import IK_controller
from utilities import criterias, rmse, circle


# шаг в течение которого будут интегрироваться скорости врашения сочленений
integration_dt: float = 0.05
# постоянная демпфирования для DLS алгоритма
damping: float = 0.0001 
#максимальное изменение углов соединения между возвращаемыми точками траектории  
max_delta = 0.6
# максимальная угловая скорость вращения соединений, значение взято из документации
max_ang_vel =  np.pi
# скорость движения концевого эффектора
velocity = 0.6
# радиус окружности
radius = 0.08
# координаты центра окружности
x_center = 0.3
y_center = 0.3

x_current = []
y_current = []
x_ref = []
y_ref = []
err_pose = []
err_vel = []
model_time = []

model = mujoco.MjModel.from_xml_path("unitree_z1/scene.xml")
data = mujoco.MjData(model)
gravity_compensation: bool = True
model.body_gravcomp[:] = float(gravity_compensation) 
model.opt.timestep = 0.002

site_name = "attachment_site"
site_id = model.site(site_name).id
key_name = "home"
key_id = model.key(key_name).id
# mocap тело, созданное в модели, двигается по референсной траектории (не отрисовывается)
mocap_name = "target"
mocap_id = model.body(mocap_name).mocapid[0]

controller = IK_controller(model, data, integration_dt, damping, velocity, site_id, max_delta, max_ang_vel, "DL")

with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    viewer.cam.distance = 1.1
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    
    while viewer.is_running():
        step_start = time.time()   
        # координата z задана в файле scene.xml, при желании, ее можно изменить
        data.mocap_pos[mocap_id, 0:2] = circle(data.time, radius, x_center, y_center, velocity)         
        x, y = data.mocap_pos[mocap_id, 0:2]
        controller.set_pose(data.site(site_id).xpos, data.mocap_pos[mocap_id])
        controller.set_vel(data.sensor("velocimeter").data.copy())
        controller.calculate_twist()
        controller.calculate_joint_velocity()
        joint_angles = controller.calculate_joint_angles()
        data.ctrl[0:6] = joint_angles[0:6]
        mujoco.mj_step(model, data)
        err_pose.append(controller.get_error_pos())
        err_vel.append(controller.get_error_vel())
       
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # при необходимости можно построить графики траекторий и использовать утилиты
        # из файла utilities для расчета rmse
        
        # x_current.append(data.site(site_id).xpos[0])
        # y_current.append(data.site(site_id).xpos[1])
        # x_ref.append(x)
        # y_ref.append(y)
        model_time.append(data.time)

plt.subplot(2,1,1)
plt.ylabel('position error, m', fontsize=12)
plt.xlabel('time, s', fontsize=12)
plt.plot(model_time, err_pose)
plt.grid()
plt.subplot(2,1,2) 
plt.ylabel('velocity_error, m/s', fontsize=12)
plt.xlabel('time, s', fontsize=12)
plt.plot(model_time, err_vel)
plt.grid()
plt.show()
