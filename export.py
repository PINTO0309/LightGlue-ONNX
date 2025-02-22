import argparse

import torch

from lightglue_onnx import DISK, LightGlue, LightGlueEnd2End, SuperPoint
from lightglue_onnx.end2end import normalize_keypoints
from lightglue_onnx.ops import patch_disk_convolution_mode
from lightglue_onnx.utils import load_image, rgb_to_grayscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=512,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers (height width).",
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        default="superpoint",
        required=False,
        help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'. Defaults to 'superpoint'.",
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the feature extractor ONNX model.",
    )
    parser.add_argument(
        "--lightglue_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the LightGlue ONNX model.",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to export an end-to-end pipeline instead of individual models.",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Whether to allow dynamic image sizes."
    )
    return parser.parse_args()


def export_onnx(
    img_size=512,
    extractor_type="superpoint",
    extractor_path=None,
    lightglue_path=None,
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
    end2end=False,
    dynamic=False,
):
    # Handle args
    if extractor_path is not None and end2end:
        raise ValueError(
            "Extractor will be combined with LightGlue when exporting end-to-end model."
        )
    if extractor_path is None:
        extractor_path = f"weights/{extractor_type}.onnx"

    if lightglue_path is None:
        if end2end:
            lightglue_path = f"weights/{extractor_type}_lightglue_end2end.onnx"
        else:
            lightglue_path = f"weights/{extractor_type}_lightglue.onnx"

    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)

    # Models
    extractor_type = extractor_type.lower()
    if extractor_type == "superpoint":
        # SuperPoint works on grayscale images.
        # image0 = rgb_to_grayscale(image0)
        # image1 = rgb_to_grayscale(image1)
        extractor = SuperPoint().eval()
        lightglue = LightGlue(extractor_type).eval()
    elif extractor_type == "disk":
        extractor = DISK().eval()
        lightglue = LightGlue(extractor_type).eval()

        patch_disk_convolution_mode(extractor)  # Fixed in PyTorch >= 2.1
    else:
        raise NotImplementedError(
            f"LightGlue has not been trained on {extractor_type} features."
        )

    # ONNX Export
    if end2end:
        pipeline = LightGlueEnd2End(extractor, lightglue).eval()
        dynamic_axes = {
            "kpts0": {1: "num_keypoints0"},
            "kpts1": {1: "num_keypoints1"},
            "matches0": {1: "num_matches0"},
            "matches1": {1: "num_matches1"},
            "mscores0": {1: "num_matches0"},
            "mscores1": {1: "num_matches1"},
        }
        if dynamic:
            dynamic_axes.update(
                {
                    "image0": {2: "height0", 3: "width0"},
                    "image1": {2: "height1", 3: "width1"},
                }
            )
        torch.onnx.export(
            pipeline,
            (image0[None], image1[None]),
            lightglue_path,
            input_names=["image0", "image1"],
            output_names=[
                "kpts0",
                "kpts1",
                "matches0",
                "matches1",
                "mscores0",
                "mscores1",
            ],
            opset_version=16,
            dynamic_axes=dynamic_axes,
        )
    else:
        # Export Extractor
        dynamic_axes = {
            "keypoints": {1: "num_keypoints"},
            "scores": {1: "num_keypoints"},
            "descriptors": {1: "num_keypoints"},
        }
        if dynamic:
            dynamic_axes.update({"image": {2: "height", 3: "width"}})
        torch.onnx.export(
            extractor,
            image0[None],
            extractor_path,
            input_names=["image"],
            output_names=["keypoints", "scores", "descriptors"],
            opset_version=16,
            dynamic_axes=dynamic_axes,
        )

        # Export LightGlue
        feats0, feats1 = extractor(image0[None]), extractor(image1[None])
        kpts0, scores0, desc0 = feats0
        kpts1, scores1, desc1 = feats1

        kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
        kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])

        torch.onnx.export(
            lightglue,
            (
                kpts0,
                kpts1,
                desc0,
                desc1,
            ),
            lightglue_path,
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["matches0", "matches1", "mscores0", "mscores1"],
            opset_version=16,
            dynamic_axes={
                "kpts0": {1: "num_keypoints0"},
                "kpts1": {1: "num_keypoints1"},
                "desc0": {1: "num_keypoints0"},
                "desc1": {1: "num_keypoints1"},
                "matches0": {1: "num_matches0"},
                "matches1": {1: "num_matches1"},
                "mscores0": {1: "num_matches0"},
                "mscores1": {1: "num_matches1"},
            },
        )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
