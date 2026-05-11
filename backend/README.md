# HairSwap-Backend

Modular FastAPI backend for a non-generative hair-swap pipeline. The
v1 API ships the first stage — high-fidelity hair / portrait extraction
on a configurable ONNX runtime — and the project layout is designed so
warping and blending stages can plug into the same `PipelineStage`
contract later.

## Run with uv

```bash
uv sync
uv run fastapi dev app/main.py
```

The dev server listens on `http://127.0.0.1:8000` and the OpenAPI docs
are available at `/docs`. The service is unauthenticated by default so
it works as a fully local in-process tool. To enforce the `X-Api-Key`
header, set:

```bash
HAIRSWAP_API_KEY_REQUIRED=true HAIRSWAP_API_KEY=your-secret \
uv run fastapi dev app/main.py
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/extract-hair` | Multipart upload (`image`); returns RGBA PNG matte. |
| `GET`  | `/api/v1/pipeline/status` | Health check for the legacy stage scaffolding. |
| `POST` | `/api/v1/pipeline/warp` | Stub for the future warping stage. |
| `POST` | `/api/v1/pipeline/blend` | Stub for the future blending stage. |

`POST /api/v1/extract-hair` decodes the upload, runs a RetinaFace ONNX
quality gate (rejects non-frontal donors with HTTP 422), runs a
CelebAMask-HQ BiSeNet face parser to produce a **hair-only** soft alpha
matte (class 17, optionally combined with class 18 = hat), and streams a
4-channel RGBA PNG back to the caller. The response includes the
diagnostic headers `X-Face-Score` and `X-Face-Yaw-Deg`.

## Required ONNX weights

Place the following files under `models/weights/`:

- `modnet.onnx` — MODNet portrait matting export (input `1x3xHxW`,
  alpha output `1x1xHxW`). Defaults to a 512×512 input; configurable
  via `HAIRSWAP_MODNET_INPUT_*`.
- `retinaface.onnx` — RetinaFace MobileNet0.25 export (PyTorch_Retinaface
  variant) with bbox/conf/landmark outputs at strides 8/16/32.

When weights are missing, the app still boots but `POST /extract-hair`
responds with HTTP 503 until the files are provisioned.

## ONNX Runtime providers

On Apple Silicon (M-series, including the M4 Pro Max target) the runtime
prefers `CoreMLExecutionProvider` and falls back to `CPUExecutionProvider`.
The provider list is configurable via
`HAIRSWAP_ORT_PROVIDER_PRIORITY` (JSON array string), so Linux CI can
force CPU only.

## Test

```bash
uv run pytest
```

Extraction tests substitute MODNet and RetinaFace with deterministic
in-process stubs and don't require any ONNX weight files on disk.

## Project structure

```text
hairswap-backend/
├── app/
│   ├── api/
│   │   ├── router_registry.py         # Auto-includes feature routers
│   │   └── v1/
│   │       ├── router.py              # Aggregates /api/v1/* endpoints
│   │       └── endpoints/
│   │           └── extraction.py      # POST /api/v1/extract-hair
│   ├── bootstrap.py                   # FastAPI factory + lifespan
│   ├── core/
│   │   ├── dependencies.py            # DI providers + ORT registry
│   │   ├── model_registry.py          # InferenceRegistry / LoadedOnnxModel
│   │   ├── onnx_session.py            # CoreML-aware session factory
│   │   ├── security.py                # API key auth
│   │   └── settings.py                # Pydantic-based config
│   ├── features/pipeline/             # Legacy warp/blend stubs
│   ├── schemas/pipeline.py
│   ├── services/
│   │   ├── base.py                    # PipelineStage ABC
│   │   ├── blending/service.py
│   │   ├── extraction/
│   │   │   ├── extraction_service.py  # MODNet + composer
│   │   │   └── quality_gate.py        # RetinaFace frontal-face gate
│   │   └── warping/service.py
│   └── shared/
│       ├── image_processing.py        # ImageProcessor (decode/letterbox/encode)
│       └── math_utils.py
├── models/weights/                    # Place ONNX files here
└── tests/
```

## Strict design rules

- No generative or diffusion models are used.
- All service-layer code is strictly typed (Pydantic for I/O, NumPy
  typing for arrays).
- Edge quality and fine-strand preservation take priority over
  inference latency; alpha mattes are remapped to the original image
  resolution before compositing.
