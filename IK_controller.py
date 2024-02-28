import mujoco
import numpy as np


class IK_controller():
    """
    Класс реализует систему управления положением конечного эффектора на основе алгоритмов 
    дифференциальной обратной кинематики (Damped Least Squares(DLS) и Transpose)
    """

    def __init__(self, model, data, integration_step, damping, desired_velocity, site_id, max_delta, max_ang_vel, mode) -> None:
        """
        Args:
            mode: выбор алгоритма управления: DLS или Transpose 
            damping: константа демпфирования алгоритма DLS
            max_delta: максимальное изменение углов соединения между возвращаемыми точками траектории
            max_ang_vel: максимальная угловая скорость вращения соединений 
            desired_velocity: желаемая скорость движения концевого эффектора
            site_id: id концевого эффектора в модели
        """
        self.model = model
        self.data = data
        self.integration_step = integration_step
        self.damping = damping
        self.site_id = site_id
        self.max_delta = max_delta
        self.max_ang_vel = max_ang_vel
        self.mode = mode
        self.desired_velocity = desired_velocity
        # инициализация необходимых переменных
        self.current_pose = np.zeros(3)
        self.desired_pose = np.zeros(3)
        self.twist = np.zeros(6)
        self.jac = np.zeros((6, model.nv))
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

    def calculate_twist(self) -> np.array:
        """
        Функция расчета "twist" матрицы скоростей
        Returns: 
            twist: twist матрица
        """
        pos_error = self.desired_pose - self.current_pose
        self.twist[:3] = pos_error / self.integration_step
        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(
            self.error_quat, self.data.mocap_quat[0], self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] /= self.integration_step

        return self.twist

    def calculate_joint_velocity(self) -> np.array:
        """
        Функция расчета скоростей сочленений выбранным алгоритмом
        Returns: 
            dq: матрица угловых скоростей сочленений
        """
        mujoco.mj_jacSite(self.model, self.data,
                          self.jac[:3], self.jac[3:], self.site_id)

        if self.mode == "DLS":
            self.dq = np.linalg.solve(
                self.jac.T @ self.jac + self.damping ** 2 * np.eye(6), self.jac.T @ self.twist)
        elif self.mode == "Transpose":
            jjte = self.jac @ self.jac.T @ self.twist
            alpha = np.dot(self.twist, jjte) / np.dot(jjte, jjte)
            self.dq = alpha * self.jac.T  @ self.twist
        else:
            print("Invalid mode!")
            return

        return self.dq

    def calculate_joint_angles(self) -> np.array:
        """
        Функция расчета требуемых углов сочленений для достижения концевым эффектором желаемого положения
        Returns:
            next_angles: матрица углов сочленений, необходимых для достижения концевым эффектором
            желаемого положения
        """
        # масштабирование шага из-за линейной аппроксимации Якобианом функции прямой кинематики
        delta = self.max_delta / max(self.max_delta, np.max(np.abs(self.dq)))
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, self.dq, self.integration_step)
        np.clip(q, *self.model.jnt_range.T, out=q)
        next_angles = np.array(
            [angle + delta * dth for angle, dth in zip(q, self.dq)])

        return next_angles

    def set_pose(self, current_pose, desired_pose) -> None:
        """
        Args: 
            current_pose: текущее положение концевого эффектора
            desired_pose: желаемое положение концевого эффектора
        """
        self.current_pose = current_pose
        self.desired_pose = desired_pose

    def set_vel(self, current_vel) -> None:
        """
        Args: 
            current_vel: текущая скорость концевого эффектра
        """
        self.current_velocity = current_vel

    def get_error_pos(self) -> float:
        """
        Функция расчета траекторной ошибки концевого эффектора
        Returns:
            err: величина текущей траекторной ошибки
        """
        err_pose = np.sqrt((self.current_pose[0] - self.desired_pose[0])**2 + (
            self.current_pose[1] - self.desired_pose[1])**2)
        return err_pose

    def get_error_vel(self) -> float:
        """
        Функция расчета ошибки линейной скорости концевого эффектора
        Returns:
            err_vel: величина текущей ошибки линейной скорости
        """
        err_vel = np.sqrt(
            (self.current_velocity[0]**2 + self.current_velocity[1]**2)) - self.desired_velocity
        return err_vel
