#!/bin/bash

RESOLUTIONS=(
    "180 320"
    "180 416"
    "180 512"
    "180 640"
    "180 800"
    "240 320"
    "240 416"
    "240 512"
    "240 640"
    "240 800"
    "240 960"
    "288 480"
    "288 512"
    "288 640"
    "288 800"
    "288 960"
    "288 1280"
    "320 320"
    "360 480"
    "360 512"
    "360 640"
    "360 800"
    "360 960"
    "360 1280"
    "376 1344"
    "416 416"
    "480 640"
    "480 800"
    "480 960"
    "480 1280"
    "512 512"
    "540 800"
    "540 960"
    "540 1280"
    "640 640"
    "640 960"
    "720 1280"
    "720 2560"
    "1080 1920"
)

for((i=0; i<${#RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[i]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}

    onnxsim superpoint_lightglue_superpoint.onnx superpoint_lightglue_superpoint_1x3x${H}x${W}.onnx \
    --overwrite-input-shape "image0:1,3,${H},${W}" "image1:1,3,${H},${W}"
    onnxsim superpoint_lightglue_superpoint_1x3x${H}x${W}.onnx superpoint_lightglue_superpoint_1x3x${H}x${W}.onnx
    onnxsim superpoint_lightglue_superpoint_1x3x${H}x${W}.onnx superpoint_lightglue_superpoint_1x3x${H}x${W}.onnx
    onnxsim superpoint_lightglue_superpoint_1x3x${H}x${W}.onnx superpoint_lightglue_superpoint_1x3x${H}x${W}.onnx

    # onnxsim superpoint_lightglue_disk.onnx superpoint_lightglue_disk_1x3x${H}x${W}.onnx \
    # --overwrite-input-shape "image0:1,3,${H},${W}" "image1:1,3,${H},${W}"
    # onnxsim superpoint_lightglue_disk_1x3x${H}x${W}.onnx superpoint_lightglue_disk_1x3x${H}x${W}.onnx
    # onnxsim superpoint_lightglue_disk_1x3x${H}x${W}.onnx superpoint_lightglue_disk_1x3x${H}x${W}.onnx
    # onnxsim superpoint_lightglue_disk_1x3x${H}x${W}.onnx superpoint_lightglue_disk_1x3x${H}x${W}.onnx
done
