# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
import torch
from gsplat import rasterization
import cv2


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

    show_anaglyph = False

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

    cv2.namedWindow("GSplat Explorer", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Roll", "GSplat Explorer", 0, 180, lambda x: None)
    cv2.createTrackbar("Pitch", "GSplat Explorer", 0, 180, lambda x: None)
    cv2.createTrackbar("Yaw", "GSplat Explorer", 0, 180, lambda x: None)
    cv2.createTrackbar("X", "GSplat Explorer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Y", "GSplat Explorer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Z", "GSplat Explorer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Scaling", "GSplat Explorer", 100, 100, lambda x: None)

    cv2.setTrackbarMin("Roll", "GSplat Explorer", -180)
    cv2.setTrackbarMax("Roll", "GSplat Explorer", 180)
    cv2.setTrackbarMin("Pitch", "GSplat Explorer", -180)
    cv2.setTrackbarMax("Pitch", "GSplat Explorer", 180)
    cv2.setTrackbarMin("Yaw", "GSplat Explorer", -180)
    cv2.setTrackbarMax("Yaw", "GSplat Explorer", 180)
    cv2.setTrackbarMin("X", "GSplat Explorer", -1000)
    cv2.setTrackbarMax("X", "GSplat Explorer", 1000)
    cv2.setTrackbarMin("Y", "GSplat Explorer", -1000)
    cv2.setTrackbarMax("Y", "GSplat Explorer", 1000)
    cv2.setTrackbarMin("Z", "GSplat Explorer", -1000)
    cv2.setTrackbarMax("Z", "GSplat Explorer", 1000)

    width = metadata["width"]
    height = metadata["height"]

    while True:
        roll = cv2.getTrackbarPos("Roll", "GSplat Explorer")
        pitch = cv2.getTrackbarPos("Pitch", "GSplat Explorer")
        yaw = cv2.getTrackbarPos("Yaw", "GSplat Explorer")

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        viewmat = (
            torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
            .float()
            .to(device)
        )
        
        viewmat[0, 3] = cv2.getTrackbarPos("X", "GSplat Explorer") / 100.0
        viewmat[1, 3] = cv2.getTrackbarPos("Y", "GSplat Explorer") / 100.0
        viewmat[2, 3] = cv2.getTrackbarPos("Z", "GSplat Explorer") / 100.0
        output, _, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat[None],
            K[None],
            width=width,
            height=height,
            sh_degree=3,
        )

        output_cv = torch_to_cv(output[0])

        if show_anaglyph:
            viewmat[0, 3] = viewmat[0, 3]-0.05
            output_left = output_cv
            output, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmat[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            output_right = torch_to_cv(output[0])
            output_left[...,:2] = 0
            output_right[...,-1] = 0
            output_cv = output_left + output_right

        cv2.imshow("GSplat Explorer", output_cv)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("3"):
            show_anaglyph = not show_anaglyph


def torch_to_cv(tensor, permute=False):
    if permute:
        tensor = torch.clamp(tensor.permute(1, 2, 0), 0, 1).cpu().numpy()
    else:
        tensor = torch.clamp(tensor, 0, 1).cpu().numpy()
    return (tensor * 255).astype(np.uint8)[..., ::-1]


if __name__ == "__main__":
    tyro.cli(main)
