import numpy as np

def get_rpy_matrix(roll, pitch, yaw):
    roll_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch), 0],
            [0, np.sin(pitch), np.cos(pitch), 0],
            [0, 0, 0, 1.0],
        ])
    
    pitch_matrix = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw), 0],
            [0, 1, 0, 0],
            [-np.sin(yaw), 0, np.cos(yaw), 0],
            [0, 0, 0, 1.0],
        ])
    yaw_matrix = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0, 0],
            [np.sin(roll), np.cos(roll), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1.0],
        ]

    )

    return yaw_matrix @ pitch_matrix @ roll_matrix