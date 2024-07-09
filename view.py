import torch
from gsplat import rasterization


from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QScrollArea, QScrollBar, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

import numpy as np
import json
import tyro

device = torch.device("cuda:0")

def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d

def load_gaussian_splats_from_input_file(input_path: str):
    with open(input_path, "r") as f:
        metadata = json.load(f)
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

    _detach_tensors_from_dict(splats)

    return splats, metadata

def main(input_path: str):
    splats, metadata = load_gaussian_splats_from_input_file(input_path)


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

    viewmat[3,3] = 10
    viewmat = viewmat.to(device)
    K = K.to(device)

    if "intrinsics" in metadata:
        intrinsics = metadata["intrinsics"]
        K = torch.tensor(
            [[intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1.0]]
        ).float().to(device)

    # metadata["height"] = 83

    if "width" not in metadata:
        metadata["width"] = 1267
        # width = metadata["width"]
    if "height" not in metadata:
        metadata["height"] = 832


    means = splats["xyz"].float()
    opacities = splats["opacity"]
    quats = splats["rotation"]
    scales = splats["scaling"].float()

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

    output, _, meta = rasterization(means, quats, scales, opacities, colors, viewmat[None], K[None], width=1267, height=83, sh_degree=3)

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

    slider_roll = QSlider()
    slider_pitch = QSlider()
    slider_yaw = QSlider()
    x_slider = QSlider()
    y_slider = QSlider()
    z_slider = QSlider()
    for slider in [slider_roll, slider_pitch, slider_yaw]:
        slider.setMinimum(-180)
        slider.setMaximum(180)
        slider.setValue(0)
        slider.setOrientation(1)
        layout.addWidget(slider)

    for slider in [x_slider, y_slider, z_slider]:
        slider.setMinimum(-1000)
        slider.setMaximum(1000)
        slider.setValue(0)
        slider.setOrientation(1)
        layout.addWidget(slider)

    scaling_slider = QSlider()
    scaling_slider.setMinimum(0)
    scaling_slider.setMaximum(100)
    scaling_slider.setValue(100)
    scaling_slider.setOrientation(1)
    layout.addWidget(scaling_slider)

    explosion_slider = QSlider()
    explosion_slider.setMinimum(0)
    explosion_slider.setMaximum(200)
    explosion_slider.setValue(100)
    explosion_slider.setOrientation(1)
    layout.addWidget(explosion_slider)


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
        roll = slider_roll.value()
        pitch = slider_pitch.value()
        yaw = slider_yaw.value()
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
        scaling_modifier = scaling_slider.value() / 100.0
        explosion = explosion_slider.value() / 100.0
        output, _, meta = rasterization(means  * explosion, quats, scales * scaling_modifier, opacities, colors, viewmat[None], K[None], width=metadata["width"], height=metadata["height"], sh_degree=3)
        output = output[0].cpu().numpy()
        output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
        image = QImage(output.data, output.shape[1], output.shape[0], output.shape[1] * 3, QImage.Format_RGB888)
        pixmap.convertFromImage(image)
        label.setPixmap(pixmap)
        
    exit(app.exec_())

if __name__ == "__main__":
    tyro.cli(main)