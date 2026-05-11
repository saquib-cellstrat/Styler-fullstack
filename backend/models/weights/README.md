# ONNX weights

Place the following ONNX files in this directory. None of them are
bundled with the repository - the service will refuse extraction
requests (HTTP 503) until the required files are present.

## Required files

### `face_parser.onnx`

CelebAMask-HQ BiSeNet face parser - the primary source of the hair-only
alpha matte. Class index 17 is `hair`, class 18 is `hat`.

- **Input:** `1 x 3 x 512 x 512` float32 (NCHW), ImageNet-normalized
  (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`).
- **Outputs:** Three 19-class logit tensors; the service uses the first
  (main head) output and ignores the auxiliary heads.
- **Default weight file:** the ResNet34 export from
  [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing/releases/tag/weights)
  (`resnet34.onnx`, ~89 MB). The smaller `resnet18.onnx` (~51 MB) is
  drop-in compatible; rename to `face_parser.onnx`.

### `retinaface.onnx`

RetinaFace MobileNet0.25 export (PyTorch_Retinaface variant) used as
the frontal-face quality gate.

- **Input:** `1 x 3 x 640 x 640` float32 (NCHW), BGR pixels with
  per-channel mean subtraction `[104, 117, 123]`.
- **Outputs:** Three tensors - bbox regressions (`...x4`), softmax
  logits (`...x2`) and 5-point landmarks (`...x10`) at strides
  8 / 16 / 32. Exact tensor order is auto-detected by shape.

The gate uses the 5 landmarks to estimate yaw and rejects donors
whose absolute yaw exceeds `HAIRSWAP_MAX_ABS_YAW_DEG` (default 35°).

## Optional file

### `modnet.onnx`

MODNet portrait matting export. **Optional refinement only** - when
present and `HAIRSWAP_REFINE_HAIR_WITH_PORTRAIT_MATTE=true`, MODNet's
person silhouette is used as a soft gate to suppress stray hair-class
predictions outside the person. Off by default because the BiSeNet
softmax is already a high-quality alpha matte on its own.

- **Input:** `1 x 3 x H x W` float32 (NCHW), normalized as
  `(x - 0.5) / 0.5`.
- **Default weight file:** the FP16 export from
  [Xenova/modnet](https://huggingface.co/Xenova/modnet/tree/main/onnx)
  (~12 MB). FP32 (~25 MB) and quantized (~6.6 MB) variants are
  drop-in compatible.

### `retinaface.onnx`

RetinaFace MobileNet0.25 export (PyTorch_Retinaface variant) used as
the frontal-face quality gate.

- **Input:** `1 x 3 x 640 x 640` float32 (NCHW), BGR pixels with
  per-channel mean subtraction `[104, 117, 123]`.
- **Outputs:** Three tensors — bbox regressions (`...x4`), softmax
  logits (`...x2`) and 5-point landmarks (`...x10`) at strides
  8 / 16 / 32. Exact tensor order is auto-detected by shape.

The gate uses the 5 landmarks to estimate yaw and rejects donors
whose absolute yaw exceeds `HAIRSWAP_MAX_ABS_YAW_DEG` (default 35°).

## Configuration overrides

All filename and threshold defaults come from
[`app/core/settings.py`](../../app/core/settings.py) and can be
overridden with `HAIRSWAP_*` environment variables — for example:

```bash
HAIRSWAP_FACE_PARSER_ONNX_FILENAME=face_parser_resnet18.onnx \
HAIRSWAP_INCLUDE_HAT_IN_HAIR=true \
uv run fastapi dev app/main.py
```
