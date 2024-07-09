import torch
from gsplat import rasterization


from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QScrollArea, QScrollBar, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

import numpy as np
import json
import tyro

def main(input_path: str):

    input_file = open(input_path, "r")
    metadata = json.load(input_file)
    checkpoint_path = metadata["checkpoint"]
    model_params, _ = torch.load(checkpoint_path)

    splats = {
        "active_sh_degree": model_params[0], 
        "xyz": model_params[1], 
        "features_dc": model_params[2], 
        "features_rest": model_params[3],
        "scaling": model_params[4], 
        "rotation": model_params[5], 
        "opacity": model_params[6].squeeze(1),
    }


    viewmat = torch.tensor([[-0.9961, -0.0325,  0.0822, -0.0713],
            [ 0.0790,  0.0903,  0.9928, -0.8718],
            [-0.0397,  0.9954, -0.0873, -0.2474],
            [ 0.0000,  0.0000,  0.0000,  1.0000]], device='cuda:0')
    K = torch.tensor(
        [
            [1000, 0, 500],
            [0, 1000, 500],
            [0, 0, 1.0]
        ]
    )

    # K = torch.tensor([[1.0548e+03, 0.0000e+00, 6.3350e+02],
    #         [0.0000e+00, 1.0514e+03, 4.1575e+02],
    #         [0.0000e+00, 0.0000e+00, 1.0000e+00]], device='cuda:0')

    device = torch.device("cuda:0")
    viewmat[3,3] = 10
    viewmat = viewmat.to(device)
    K = K.to(device)


    means = splats["xyz"].float()
    opacities = splats["opacity"]
    # colors = splats["sh0"]
    quats = splats["rotation"]
    scales = splats["scaling"].float()

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

    output, _, meta = rasterization(means, quats, scales, opacities, colors, viewmat[None], K[None], width=1267, height=832, sh_degree=3)

    output = output[0].cpu().numpy()
    output = (np.clip(output, 0, 1) * 255).astype(np.uint8)





    app = QApplication([])

    window = QMainWindow()
    window.setWindowTitle("GSplat Viewer")
    window.setGeometry(100, 100, 640, 480)
    window.setCentralWidget(QWidget())
    layout = QVBoxLayout(window.centralWidget())
    label = QLabel()
    scroll_area = QScrollArea()
    layout.addWidget(scroll_area)


    scroll_area.setWidget(label)
    scroll_area.setWidgetResizable(True)

    scrollbar_roll = QSlider()
    scrollbar_pitch = QScrollBar()
    scrollbar_yaw = QScrollBar()
    x_slider = QSlider()
    y_slider = QSlider()
    z_slider = QSlider()
    for scrollbar in [scrollbar_roll, scrollbar_pitch, scrollbar_yaw]:
        scrollbar.setMinimum(-180)
        scrollbar.setMaximum(180)
        scrollbar.setValue(0)
        scrollbar.setOrientation(1)
        layout.addWidget(scrollbar)

    for slider in [x_slider, y_slider, z_slider]:
        slider.setMinimum(-1000)
        slider.setMaximum(1000)
        slider.setValue(0)
        slider.setOrientation(1)
        layout.addWidget(slider)


    pixmap = QPixmap(output.shape[1], output.shape[0])
    image = QImage(output.data, output.shape[1], output.shape[0], output.shape[1] * 3, QImage.Format_RGB888)
    pixmap.convertFromImage(image)
    label.setPixmap(pixmap)
    window.show()

    timer = QTimer()

    def timeout_handler():
        loop()

    timer.timeout.connect(timeout_handler)
    timer.start(10)

    def loop():
        # global viewmat, K
        roll = scrollbar_roll.value()
        pitch = scrollbar_pitch.value()
        yaw = scrollbar_yaw.value()
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        roll_mat = torch.tensor(
            [
                [np.cos(roll_rad), -np.sin(roll_rad), 0, 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1.0]
            ]).float().to(device)
        pitch_mat = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad), 0],
                [0, np.sin(pitch_rad), np.cos(pitch_rad), 0],
                [0, 0, 0, 1.0]
            ]).float().to(device)
        yaw_mat = torch.tensor(
            [
                [np.cos(yaw_rad), 0, np.sin(yaw_rad), 0],
                [0, 1, 0, 0],
                [-np.sin(yaw_rad), 0, np.cos(yaw_rad), 0],
                [0, 0, 0, 1.0]
            ]).float().to(device)
        viewmat = yaw_mat @ pitch_mat @ roll_mat 
        viewmat[0,3] = x_slider.value() /100.0
        viewmat[1,3] = y_slider.value() /100.0
        viewmat[2,3] = z_slider.value() /100.0
        nonlocal K, means, quats, scales, opacities, colors
        output, _, meta = rasterization(means, quats, scales, opacities, colors, viewmat[None], K[None], width=1267, height=832, sh_degree=3)
        output = output[0].cpu().numpy()
        output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
        image = QImage(output.data, output.shape[1], output.shape[0], output.shape[1] * 3, QImage.Format_RGB888)
        pixmap.convertFromImage(image)
        label.setPixmap(pixmap)
        
    exit(app.exec_())

if __name__ == "__main__":
    tyro.cli(main)