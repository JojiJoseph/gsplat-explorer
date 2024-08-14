# GSPLAT EXPLORER

This is a tool for exploring Gaussian Splats.

# How to run

```bash
python view.py --input_path path/to/json
```

# File Format
A json file is used to wrap the checkpoint path and intrinsics.

An example file looks like the following.

```json
{
    "type": "gaussian-splatting",
    "checkpoint": "data/teatime/chkpnt30000.pth",
    "intrinsics": {
        "fx": 517.3,
        "fy": 516.5,
        "cx": 318.6,
        "cy": 255.3
    },
    "width": 640,
    "height": 480
}
```
The checkpoint can be obtained with https://github.com/graphdeco-inria/gaussian-splatting

# Sample Output


https://github.com/user-attachments/assets/09da4735-1c2a-4e3a-98a9-eec72355cd1d

