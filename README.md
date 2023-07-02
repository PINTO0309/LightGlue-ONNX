# LightGlue ONNX

## PINTO Custom
1. In the process of processing, the feature point narrowing process by score threshold is often used, which causes onnxruntime to terminate abnormally when the number of detected points reaches zero.
2. Abolish the narrowing of feature points by score (heavy use of `NonZero` judgment) and change to a fixed extraction of the top 20 scores.
3. Although the feature points are insufficiently narrowed down by score, only 20 of the inference results need to be filtered by score.
4. The program should determine the score thresholds. For example, use Numpy.
5. Inference performance is only slightly worse because 20 fixed points are needlessly processed.
6. Since it is a pain to write preprocessing of the input image in the program, grayscale conversion is included in the model.
  ![image](https://github.com/PINTO0309/LightGlue-ONNX/assets/33194443/f6980a01-fea9-42f7-a74d-deff7a902cab)

---

Open Neural Network Exchange (ONNX) compatible implementation of [LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue). The ONNX model format allows for interoperability across different platforms with support for multiple execution providers, and removes Python-specific dependencies such as PyTorch.

<p align="center"><a href="https://arxiv.org/abs/2306.13643"><img src="assets/easy_hard.jpg" alt="LightGlue figure" width=80%></a>

## Updates

- **1 July 2023**: Add support for extractor `max_num_keypoints`.
- **30 June 2023**: Add support for DISK extractor.
- **28 June 2023**: Add end-to-end SuperPoint+LightGlue export & inference pipeline.

## ONNX Export

Prior to exporting the ONNX models, please install the [requirements](./requirements.txt) of the original LightGlue repository. ([Flash Attention](https://github.com/HazyResearch/flash-attention) does not need to be installed.)

To convert the DISK or SuperPoint and LightGlue models to ONNX, run [`export.py`](./export.py). We provide two types of ONNX exports: individual standalone models, and a combined end-to-end pipeline (recommended for convenience) with the `--end2end` flag.

```bash
python export.py \
  --img_size 512 \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --dynamic
```

- Exporting individually can be useful when intermediate outputs can be cached or precomputed. On the other hand, the end-to-end pipeline can be more convenient.
- Although dynamic axes have been specified, it is recommended to export your own ONNX model with the appropriate input image sizes of your use case.

## ONNX Inference

With ONNX models in hand, one can perform inference on Python using ONNX Runtime (see [requirements-onnx.txt](./requirements-onnx.txt)).

The LightGlue inference pipeline has been encapsulated into a runner class:

```python
from onnx_runner import LightGlueRunner, load_image, rgb_to_grayscale

image0, scales0 = load_image("assets/sacre_coeur1.jpg", resize=512)
image1, scales1 = load_image("assets/sacre_coeur2.jpg", resize=512)
image0 = rgb_to_grayscale(image0)  # only needed for SuperPoint
image1 = rgb_to_grayscale(image1)  # only needed for SuperPoint

# Create ONNXRuntime runner
runner = LightGlueRunner(
    extractor_path="weights/superpoint.onnx",
    lightglue_path="weights/superpoint_lightglue.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Run inference
m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)
```

Note that the output keypoints have already been rescaled back to the original image sizes.

Alternatively, you can also run [`infer.py`](./infer.py).

```bash
python infer.py \
  --img_paths assets/DSC_0410.JPG assets/DSC_0411.JPG \
  --img_size 512 \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --viz
```

## Caveats

As the ONNX Runtime has limited support for features like dynamic control flow, certain configurations of the models cannot be exported to ONNX easily. These caveats are outlined below.

### Feature Extraction

- Only batch size `1` is currently supported. This limitation stems from the fact that different images in the same batch can have varying numbers of keypoints, leading to non-uniform (a.k.a. *ragged*) tensors.

### LightGlue Keypoint Matching

- Since dynamic control flow has limited support in ONNX tracing, by extension, early stopping and adaptive point pruning (the `depth_confidence` and `width_confidence` parameters) are also difficult to export to ONNX.
- Flash Attention is turned off.
- Mixed precision is turned off.
- Note that the end-to-end version, despite its name, still requires the postprocessing (filtering valid matches) function outside the ONNX model since the `scales` variables need to be passed.

Additionally, the outputs of the ONNX models differ slightly from the original PyTorch models (by a small error on the magnitude of `1e-6` to `1e-5` for the scores/descriptors). Although the cause is still unclear, this could be due to differing implementations or modified dtypes.

## Possible Future Work

- **Support for TensorRT**: Appears to be currently blocked by unsupported Einstein summation operations (`torch.einsum()`) in TensorRT - Thanks to [Shidqiet](https://github.com/Shidqiet)'s investigation.
- **Support for batch size > 1**: Blocked by the fact that different images can have varying numbers of keypoints. Perhaps max-length padding?
- **Support for dynamic control flow**: Investigating *script-mode* ONNX export instead of *trace-mode*.
- **Mixed-precision Support**
- **Quantization Support**

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [LightGlue](https://arxiv.org/abs/2306.13643) and [SuperPoint](https://arxiv.org/abs/1712.07629) and [DISK](https://arxiv.org/abs/2006.13566). Lastly, if the ONNX versions helped you in any way, please also consider starring this repository.

```txt
@inproceedings{lindenberger23lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue}: Local Feature Matching at Light Speed},
  booktitle = {ArXiv PrePrint},
  year      = {2023}
}
```

```txt
@article{DBLP:journals/corr/abs-1712-07629,
  author       = {Daniel DeTone and
                  Tomasz Malisiewicz and
                  Andrew Rabinovich},
  title        = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  journal      = {CoRR},
  volume       = {abs/1712.07629},
  year         = {2017},
  url          = {http://arxiv.org/abs/1712.07629},
  eprinttype    = {arXiv},
  eprint       = {1712.07629},
  timestamp    = {Mon, 13 Aug 2018 16:47:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1712-07629.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```txt
@article{DBLP:journals/corr/abs-2006-13566,
  author       = {Michal J. Tyszkiewicz and
                  Pascal Fua and
                  Eduard Trulls},
  title        = {{DISK:} Learning local features with policy gradient},
  journal      = {CoRR},
  volume       = {abs/2006.13566},
  year         = {2020},
  url          = {https://arxiv.org/abs/2006.13566},
  eprinttype    = {arXiv},
  eprint       = {2006.13566},
  timestamp    = {Wed, 01 Jul 2020 15:21:23 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2006-13566.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
