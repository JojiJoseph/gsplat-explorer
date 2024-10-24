import torch
from gsplat import rasterization


from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QScrollBar,
    QSlider,
    QGridLayout,
    QFrame,
    QTextEdit,
    QLineEdit,
    QHBoxLayout,
)
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QColor,
    QColorConstants,
    QDoubleValidator,
)
from PyQt5.QtCore import QTimer

import numpy as np
import json
import tyro

from utils import get_rpy_matrix

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
    K = torch.tensor([[1000, 0, 500], [0, 1000, 500], [0, 0, 1.0]])
    K = K.to(device)

    if "intrinsics" in metadata:
        intrinsics = metadata["intrinsics"]
        K = (
            torch.tensor(
                [
                    [intrinsics["fx"], 0, intrinsics["cx"]],
                    [0, intrinsics["fy"], intrinsics["cy"]],
                    [0, 0, 1.0],
                ]
            )
            .float()
            .to(device)
        )


    if "width" not in metadata:
        metadata["width"] = 1267
    if "height" not in metadata:
        metadata["height"] = 832

    means = splats["xyz"].float()
    opacities = splats["opacity"]
    quats = splats["rotation"]
    scales = splats["scaling"].float()

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

    

    class GaussianSplatViewer(QApplication):
        def __init__(self, argv: torch.List[str]) -> None:
            super().__init__(argv)
            self._create_window()
            self._create_menu()
            self._create_viewport()
            self._create_sliders()
            self._create_intrinsic_panel()
            self._create_extents_panel()
            self._create_world_offset_panel()

        def _create_world_offset_panel(self):
            self.world_offset_frame = QFrame()
            self.sidebar.addWidget(self.world_offset_frame)
            self.world_offset_layout = QGridLayout(self.world_offset_frame)
            self.world_offset_frame.setLayout(self.world_offset_layout)
            self.world_offset_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
            
            roll_label = QLabel("Roll")
            pitch_label = QLabel("Pitch")
            yaw_label = QLabel("Yaw")
            x_label = QLabel("X")
            y_label = QLabel("Y")
            z_label = QLabel("Z")
            roll_slider = QSlider()
            pitch_slider = QSlider()
            yaw_slider = QSlider()
            x_slider = QSlider()
            y_slider = QSlider()
            z_slider = QSlider()

            self.world_offset_roll_slider = roll_slider
            self.world_offset_pitch_slider = pitch_slider
            self.world_offset_yaw_slider = yaw_slider
            self.world_offset_x_slider = x_slider
            self.world_offset_y_slider = y_slider
            self.world_offset_z_slider = z_slider

            row = 0
            for slider, label, type_ in zip([roll_slider, pitch_slider, yaw_slider, x_slider, y_slider, z_slider], [roll_label, pitch_label, yaw_label, x_label, y_label, z_label], ["roll", "pitch", "yaw", "x", "y", "z"]):
                slider.setMinimum(-180)
                slider.setMaximum(180)
                if type_ in ["x", "y", "z"]:
                    slider.setMinimum(-1000)
                    slider.setMaximum(1000)
                    slider.setValue(0)
                slider.setValue(0)
                slider.setOrientation(1)
                label.setText(f"{type_}: {slider.value()}")
                self.world_offset_layout.addWidget(label, row, 0)
                self.world_offset_layout.addWidget(slider, row, 1)
                slider.valueChanged.connect(lambda value, label=label, type_=type_: label.setText(f"{type_}: {value}"))
                row += 1

        def _create_window(self):
            window = QMainWindow()
            window.setWindowTitle("GSplat Explorer")
            window.setGeometry(100, 100, 640, 480)
            window.setCentralWidget(QWidget())
            self.window = window
            layout = QVBoxLayout(self.window.centralWidget())
            self.layout = layout

            self.show_anaglyph = False

        def _toggle_anaglyph(self):
            
            self.show_anaglyph = not self.show_anaglyph

        def _create_menu(self):
            file_menu = self.window.menuBar().addMenu("File")
            action = file_menu.addAction("Quit")
            action.triggered.connect(self.window.close)
            action.setShortcut("Ctrl+Q")

            view_menu = self.window.menuBar().addMenu("View")
            toggle_anaglyph_action = view_menu.addAction("Anaglyph 3D")
            toggle_anaglyph_action.setCheckable(True)
            toggle_anaglyph_action.triggered.connect(lambda: self._toggle_anaglyph())
            toggle_anaglyph_action.setShortcut("Ctrl+3")


        def _create_viewport(self):
            self.viewport = QLabel()
            scroll_area = QScrollArea()
            self.sidebar = QVBoxLayout()
            viewport_and_sidebar = QHBoxLayout()
            self.layout.addLayout(viewport_and_sidebar)
            viewport_and_sidebar.addWidget(scroll_area)
            scroll_area.setWidget(self.viewport)
            scroll_area.setWidgetResizable(True)
            scroll_area_2 = QScrollArea()
            viewport_and_sidebar.addWidget(scroll_area_2)
            scroll_area_2.setLayout(self.sidebar)

        def _create_intrinsic_panel(self):
            self.intrinsics_frame = QFrame()
            self.sidebar.addWidget(self.intrinsics_frame)
            self.intrinsics_layout = QGridLayout(self.intrinsics_frame)
            self.intrinsics_frame.setLayout(self.intrinsics_layout)
            self.intrinsics_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
            fx_label = QLabel("fx")
            fy_label = QLabel("fy")
            cx_label = QLabel("cx")
            cy_label = QLabel("cy")
            width_label = QLabel("width")
            height_label = QLabel("height")
            self.fx_input = QLineEdit()
            self.fy_input = QLineEdit()
            self.cx_input = QLineEdit()
            self.cy_input = QLineEdit()
            self.width_input = QLineEdit()
            self.height_input = QLineEdit()
            self.fx_input.setText(str(K[0, 0].item()))
            self.fy_input.setText(str(K[1, 1].item()))
            self.cx_input.setText(str(K[0, 2].item()))
            self.cy_input.setText(str(K[1, 2].item()))
            self.width_input.setText(str(metadata["width"]))
            self.height_input.setText(str(metadata["height"]))
            for input_ in [
                self.fx_input,
                self.fy_input,
                self.cx_input,
                self.cy_input,
                self.width_input,
                self.height_input,
            ]:
                input_.setValidator(QDoubleValidator(input_))
            self.intrinsics_layout.addWidget(fx_label, 0, 0)
            self.intrinsics_layout.addWidget(fy_label, 1, 0)
            self.intrinsics_layout.addWidget(cx_label, 2, 0)
            self.intrinsics_layout.addWidget(cy_label, 3, 0)
            self.intrinsics_layout.addWidget(width_label, 4, 0)
            self.intrinsics_layout.addWidget(height_label, 5, 0)
            self.intrinsics_layout.addWidget(self.fx_input, 0, 1)
            self.intrinsics_layout.addWidget(self.fy_input, 1, 1)
            self.intrinsics_layout.addWidget(self.cx_input, 2, 1)
            self.intrinsics_layout.addWidget(self.cy_input, 3, 1)
            self.intrinsics_layout.addWidget(self.width_input, 4, 1)
            self.intrinsics_layout.addWidget(self.height_input, 5, 1)

        def _create_extents_panel(self):
            self.extents_frame = QFrame()
            self.sidebar.addWidget(self.extents_frame)
            self.extents_layout = QGridLayout(self.extents_frame)
            self.extents_frame.setLayout(self.extents_layout)
            self.extents_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
            x_minus_label = QLabel("X-")
            x_plus_label = QLabel("X+")
            y_minus_label = QLabel("Y-")
            y_plus_label = QLabel("Y+")
            z_minus_label = QLabel("Z-")
            z_plus_label = QLabel("Z+")
            
            self.x_minus_slider = QSlider()
            self.x_plus_slider = QSlider()
            self.y_minus_slider = QSlider()
            self.y_plus_slider = QSlider()
            self.z_minus_slider = QSlider()
            self.z_plus_slider = QSlider()

            row = 0
            for slider, label, type_ in zip([self.x_minus_slider, self.x_plus_slider, self.y_minus_slider, self.y_plus_slider, self.z_minus_slider, self.z_plus_slider], [x_minus_label, x_plus_label, y_minus_label, y_plus_label, z_minus_label, z_plus_label], ["x-", "x+", "y-", "y+", "z-", "z+"]):
                slider.setMinimum(-2000)
                slider.setMaximum(2000)
                if type_[-1] == "+":
                    slider.setValue(2000)
                else:
                    slider.setValue(-2000)
                slider.setOrientation(1)
                label.setText(f"{type_}: {slider.value()/100}")
                self.extents_layout.addWidget(label, row, 0)
                self.extents_layout.addWidget(slider, row, 1)
                slider.valueChanged.connect(lambda value, label=label, type_=type_: label.setText(f"{type_}: {value/100}"))
                row += 1

        def _create_sliders(self):
            label_roll = QLabel("Roll")
            label_pitch = QLabel("Pitch")
            label_yaw = QLabel("Yaw")
            slider_roll = QSlider()
            slider_pitch = QSlider()
            slider_yaw = QSlider()
            x_slider = QSlider()
            y_slider = QSlider()
            z_slider = QSlider()
            self.slider_grid = QGridLayout()
            self.layout.addLayout(self.slider_grid)

            row_ = 0
            for slider, label, type_ in zip(
                [slider_roll, slider_pitch, slider_yaw],
                [label_roll, label_pitch, label_yaw],
                ["roll", "pitch", "yaw"],
            ):
                slider.setMinimum(-180)
                slider.setMaximum(180)
                slider.setValue(0)
                slider.setOrientation(1)
                # self.layout.addWidget(slider)
                label.setText(f"{type_}: {slider.value()}")
                self.slider_grid.addWidget(label, row_, 0)
                self.slider_grid.addWidget(slider, row_, 1)
                slider.valueChanged.connect(
                    lambda value, label=label, type_=type_: label.setText(
                        f"{type_}: {value}"
                    )
                )
                row_ += 1

            label_x = QLabel("X")
            label_y = QLabel("Y")
            label_z = QLabel("Z")

            for slider, label, type_ in zip(
                [x_slider, y_slider, z_slider],
                [label_x, label_y, label_z],
                ["x", "y", "z"],
            ):
                slider.setMinimum(-1000)
                slider.setMaximum(1000)
                slider.setValue(0)
                slider.setOrientation(1)
                # self.layout.addWidget(slider)
                label.setText(f"{type_}: {slider.value()}")
                self.slider_grid.addWidget(label, row_, 0)
                self.slider_grid.addWidget(slider, row_, 1)
                slider.valueChanged.connect(
                    lambda value, label=label, type_=type_: label.setText(
                        f"{type_}: {value/100}"
                    )
                )
                row_ += 1

            scaling_label = QLabel("Scaling: 1.0")

            scaling_slider = QSlider()
            scaling_slider.setMinimum(0)
            scaling_slider.setMaximum(100)
            scaling_slider.setValue(100)
            scaling_slider.setOrientation(1)
            self.slider_grid.addWidget(scaling_label, row_, 0)
            self.slider_grid.addWidget(scaling_slider, row_, 1)
            scaling_slider.valueChanged.connect(
                lambda value, label=scaling_label: label.setText(
                    f"Scaling: {value/100}"
                )
            )
            row_ += 1

            explosion_label = QLabel("Explosion: 1.0")

            explosion_slider = QSlider()
            explosion_slider.setMinimum(0)
            explosion_slider.setMaximum(200)
            explosion_slider.setValue(100)
            explosion_slider.setOrientation(1)
            self.slider_grid.addWidget(explosion_label, row_, 0)
            self.slider_grid.addWidget(explosion_slider, row_, 1)
            explosion_slider.valueChanged.connect(
                lambda value, label=explosion_label: label.setText(
                    f"Explosion: {value/100}"
                )
            )

            self.slider_roll = slider_roll
            self.slider_pitch = slider_pitch
            self.slider_yaw = slider_yaw
            self.x_slider = x_slider
            self.y_slider = y_slider
            self.z_slider = z_slider
            self.scaling_slider = scaling_slider
            self.explosion_slider = explosion_slider

        def run(self):
            self.window.show()

            timer = QTimer()
            timer.timeout.connect(self.loop)
            timer.start(10)
            exit(self.exec_())

        def _get_viewmat_from_sliders(self):
            viewmat = torch.eye(4, device=device)
            roll = self.slider_roll.value()
            pitch = self.slider_pitch.value()
            yaw = self.slider_yaw.value()
            roll_rad = np.deg2rad(roll)
            pitch_rad = np.deg2rad(pitch)
            yaw_rad = np.deg2rad(yaw)
            viewmat = (
                torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
                .float()
                .to(device)
            )
            viewmat[0, 3] = self.x_slider.value() / 100.0
            viewmat[1, 3] = self.y_slider.value() / 100.0
            viewmat[2, 3] = self.z_slider.value() / 100.0
            return viewmat

        def _get_K_from_panel(self):
            fx = float(self.fx_input.text() or "0")
            fy = float(self.fy_input.text() or "0")
            cx = float(self.cx_input.text() or "0")
            cy = float(self.cy_input.text() or "0")
            width = int(self.width_input.text() or "0")
            height = int(self.height_input.text() or "0")

            # Clamp values and update input fields
            fx = np.clip(fx, 100, 3000)
            fy = np.clip(fy, 100, 3000)
            cx = np.clip(cx, -3000, 3000)
            cy = np.clip(cy, -3000, 3000)
            width = np.clip(width, 1, 3000)
            height = np.clip(height, 1, 3000)

            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]).float().to(device)
            return K, width, height

        def _torch_to_qimage(self, output):
            output = output.cpu().numpy()
            output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
            image = QImage(
                output.data,
                output.shape[1],
                output.shape[0],
                output.shape[1] * 3,
                QImage.Format_RGB888,
            )
            return image

        def _get_offset_mtx_from_panel(self):
            roll_offset = self.world_offset_roll_slider.value()
            pitch_offset = self.world_offset_pitch_slider.value()
            yaw_offset = self.world_offset_yaw_slider.value()
            x_offset = self.world_offset_x_slider.value() / 100
            y_offset = self.world_offset_y_slider.value() / 100
            z_offset = self.world_offset_z_slider.value() / 100

            roll_offset = np.deg2rad(roll_offset)
            pitch_offset = np.deg2rad(pitch_offset)
            yaw_offset = np.deg2rad(yaw_offset)

            world_offset_mtx = get_rpy_matrix(roll_offset, pitch_offset, yaw_offset)
            world_offset_mtx[0, 3] = x_offset
            world_offset_mtx[1, 3] = y_offset
            world_offset_mtx[2, 3] = z_offset
            return world_offset_mtx
        def loop(self):
            nonlocal K, means, quats, scales, opacities, colors

            pixmap = QPixmap()
            viewmat = self._get_viewmat_from_sliders()
            K, width, height = self._get_K_from_panel()

            scaling_modifier = self.scaling_slider.value() / 100.0
            explosion = self.explosion_slider.value() / 100.0

            x_minus = self.x_minus_slider.value() / 100
            x_plus = self.x_plus_slider.value() / 100
            y_minus = self.y_minus_slider.value() / 100
            y_plus = self.y_plus_slider.value() / 100
            z_minus = self.z_minus_slider.value() / 100
            z_plus = self.z_plus_slider.value() / 100

            

            
            world_offset_mtx = self._get_offset_mtx_from_panel()
            to_new_world = np.linalg.inv(world_offset_mtx)
  
            new_means = torch.tensor(to_new_world[:3,:3]).float().cuda() @ means.T
            new_means = new_means.T + torch.tensor(to_new_world[:3,3]).float().cuda().T
            # new_means = new_means.T

            mask = (new_means[:, 0] > x_minus) & (new_means[:, 0] < x_plus) #& (new_means[:, 1] > y_minus) & (new_means[:, 1] < y_plus) & (new_means[:, 2] > z_minus) & (new_means[:, 2] < z_plus)
            
            means_filtered = means[mask]
            quats_filtered = quats[mask]
            scales_filtered = scales[mask]
            opacities_filtered = opacities[mask]
            colors_filtered = colors[mask]

            output, _, meta = rasterization(
                means_filtered * explosion,
                quats_filtered,
                scales_filtered * scaling_modifier,
                opacities_filtered,
                colors_filtered,
                viewmat[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            if self.show_anaglyph:
                viewmat[0, 3] = viewmat[0, 3] - 0.05
                output_left = output
                output_right, _, meta = rasterization(
                    means_filtered * explosion,
                    quats_filtered,
                    scales_filtered * scaling_modifier,
                    opacities_filtered,
                    colors_filtered,
                    viewmat[None],
                    K[None],
                    width=width,
                    height=height,
                    sh_degree=3,
                )
                output_left[..., 1:] = 0
                output_right[...,0] = 0
                output = output_left + output_right
            image = self._torch_to_qimage(output[0])
            pixmap.convertFromImage(image)

            painter = QPainter(pixmap)
            self._draw_gizmos(
                painter, viewmat.detach().cpu().numpy(), K.detach().cpu().numpy()
            )

            self.viewport.setPixmap(pixmap)

        def _draw_gizmos(self, painter, viewmat_np, K_np):

            self._draw_world_axes(painter, viewmat_np, K_np)
            painter.end()

        def _draw_world_axes(self, painter, viewmat_np, K_np):
            points_3d = np.array(
                [
                    [0, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                    [0, 0, 1, 1],
                ]
            ).astype(np.float32)

            roll_offset = self.world_offset_roll_slider.value()
            pitch_offset = self.world_offset_pitch_slider.value()
            yaw_offset = self.world_offset_yaw_slider.value()
            x_offset = self.world_offset_x_slider.value() / 100
            y_offset = self.world_offset_y_slider.value() / 100
            z_offset = self.world_offset_z_slider.value() / 100

            roll_offset = np.deg2rad(roll_offset)
            pitch_offset = np.deg2rad(pitch_offset)
            yaw_offset = np.deg2rad(yaw_offset)


            world_offset_mtx = get_rpy_matrix(roll_offset, pitch_offset, yaw_offset)
            world_offset_mtx[0, 3] = x_offset
            world_offset_mtx[1, 3] = y_offset
            world_offset_mtx[2, 3] = z_offset

            points_3d_wrt_camera = viewmat_np @ world_offset_mtx @ points_3d.T
            points_2d = K_np @ points_3d_wrt_camera[:3, :]
            points_2d /= points_2d[2, :]
            points_2d = points_2d[:2, :].T
            points_2d = points_2d.astype(np.int32)

            # Draw x-axis
            painter.setPen(QColorConstants.Red)
            painter.drawLine(
                points_2d[0, 0], points_2d[0, 1], points_2d[1, 0], points_2d[1, 1]
            )
            # Draw y-axis
            painter.setPen(QColorConstants.Green)
            painter.drawLine(
                points_2d[0, 0], points_2d[0, 1], points_2d[2, 0], points_2d[2, 1]
            )
            # Draw z-axis
            painter.setPen(QColorConstants.Blue)
            painter.drawLine(
                points_2d[0, 0], points_2d[0, 1], points_2d[3, 0], points_2d[3, 1]
            )

    app = GaussianSplatViewer([])
    app.run()


if __name__ == "__main__":
    tyro.cli(main)
