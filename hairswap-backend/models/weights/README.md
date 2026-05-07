# ONNX weights

Place the following ONNX files in this directory. None of them are
bundled with the repository - they must be supplied locally and the
service will refuse extraction requests (HTTP 503) until both are
present.

## Required files

### `modnet.onnx`

MODNet portrait matting model exported to ONNX.

- **Input:** `1 x 3 x H x W` float32 (NCHW), normalized with the original
  MODNet recipe `(x - 0.5) / 0.5` (i.e. `mean = [0.5, 0.5, 0.5]`,
  `std = [0.5, 0.5, 0.5]`, channels in `[-1, 1]`).
- **Output:** Single alpha channel tensor; the loader accepts both
  `1 x 1 x H x W` and `1 x H x W` shapes and clamps to `[0, 1]`.
- **Default input resolution:** 512 x 512 (override via
  `HAIRSWAP_MODNET_INPUT_HEIGHT` / `HAIRSWAP_MODNET_INPUT_WIDTH`).

> MODNet is a portrait matting model: the alpha channel represents
> the foreground person, not an isolated hair layer. The v1 API
> intentionally exposes that matte as the RGBA alpha channel as
> specified; isolating only hair without generative models is left
> to a future stage.

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
HAIRSWAP_MODNET_ONNX_FILENAME=modnet_p_512.onnx \
HAIRSWAP_RETINAFACE_ONNX_FILENAME=retinaface_mnet0.25.onnx \
uv run fastapi dev app/main.py
```
