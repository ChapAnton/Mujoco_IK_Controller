import numpy as np


def circle(time, r, x_center, y_center, velocity) -> np.ndarray:
    """
    Функция, описывающая референсную окружность движения конечного эффектора
    Args:
        time: время
        r: радиус 
        x_center: x координата центра оружности
        y_center: y координата центра оружности
        velocity: уставка линейной скорости концевого эффектора
    Returns:
        np.array([x, y]) - x,y координаты точки траектории
    """
    x = r * np.cos(2 * np.pi * time * (velocity / r)/(2 * np.pi)) + x_center
    y = r * np.sin(2 * np.pi * time * (velocity / r)/(2 * np.pi)) + y_center

    return np.array([x, y])

def criterias(error, delta):
    """
    Функция для расчета начала установившегося режима
    Args:
        error: текущая величина ошибки
        delta: величина ошибки в установившемся режиме 
    Returns: 
        i: номер первого индекса, с которого начинается установившийся режим
    """
    for i in range(0, len(error)):
        if abs(error[i]) <= delta:
            return i

def rmse(err):
    """
    Функция рассчитывает среднюю квадратичную ошибку (rmse)
    Args: 
        err: величина ошибки алгоритма
    Returns:
        rmse: величина среднеквадратичной ошибки
    """
    return np.sqrt(((err) ** 2).mean())
