# Hair Swap Backend Pipeline

This document describes the current backend implementation of the hair swap
pipeline so it can be reviewed by a lead developer. The implementation is a
non-generative image processing pipeline: it detects/aligns both faces, extracts
only the donor hair, warps that donor hair onto the base portrait, and blends the
result into a PNG.

## Entry Point

Primary endpoint: `POST /api/v1/swap-hair`

Implemented in `backend/app/api/v1/endpoints/swap_hair.py`.

Request:

- `base_image`: the target/base portrait, described in the API as the bald user image.
- `donor_image`: the source image whose hair should be transferred.
- Accepted content types: JPEG, JPG, PNG, WEBP.

Response:

- PNG image streamed as `image/png`.
- The output is an RGBA PNG, but the alpha channel is currently forced to fully opaque
  in the final blend stage.
- Telemetry is returned in headers:
  - selected implementations for each stage.
  - per-stage timings.
  - `X-Pipeline-Stage-Details`, a JSON list of stage name, implementation, timing,
    and declared models/algorithms.

Errors:

- `400`: empty upload.
- `415`: unsupported content type.
- `422`: pipeline stage failure, usually propagated from `ValueError`.
- `503`: required inference models were not loaded.

The endpoint constructs a `ProcessingContext` with both image byte arrays, calls
`MasterPipeline.run(...)`, encodes `context.output_rgba`, and returns it.

## Pipeline Orchestration

The orchestrator is `MasterPipeline` in `backend/app/pipeline/master.py`.

Stage order is fixed:

1. `alignment`
2. `extraction`
3. `warp`
4. `harmonization`
5. `blending`

The registry is built in `backend/app/pipeline/builder.py`. Current registered
implementations are:

| Stage | Default implementation | Class |
| --- | --- | --- |
| `alignment` | `retinaface` | `RetinaFaceAlignmentStage` |
| `extraction` | `modnet` | `ModNetExtractionStage` |
| `warp` | `tps` | `TpsWarpStage` |
| `warp` alternative | `mls` | `MlsWarpStage` |
| `harmonization` | `lab_transfer` | `LabColorTransferStage` |
| `blending` | `laplacian` | `LaplacianBlendStage` |

The stage names are configurable through settings such as
`HAIRSWAP_PIPELINE_WARP_IMPL`, but the public endpoint does not currently expose
per-request implementation overrides.

Intermediate data is passed through `ProcessingContext` in
`backend/app/pipeline/context.py`. Important fields include decoded RGB images,
detected faces, scalp anchors, donor hair alpha/RGBA, warped hair RGB/alpha,
optional dense landmarks and cranial hulls, optional occlusion metadata, final
RGBA output, timings, and selected implementations.

## Model Loading

Required model files under `backend/models/weights/`:

- `face_parser.onnx`: BiSeNet/CelebAMask-HQ face parser.
- `retinaface.onnx`: RetinaFace face detector and 5-point landmark model.

Optional model files:

- `modnet.onnx`: portrait matte used only if
  `HAIRSWAP_REFINE_HAIR_WITH_PORTRAIT_MATTE=true`.
- `dense_landmarks.onnx`: loaded if present, but the current dense landmark
  service uses a deterministic heuristic 68-point generator.
- `depth_small.onnx`: loaded if present, but the current depth estimator is a
  deterministic CPU fallback rather than ONNX inference.

ONNX Runtime sessions are created in `backend/app/core/onnx_session.py`.
Provider priority defaults to CoreML first, then CPU:

- `CoreMLExecutionProvider`
- `CPUExecutionProvider`

The dependency loader in `backend/app/core/dependencies.py` requires the face
parser and RetinaFace weights. MODNet, dense landmarks, and depth are optional.

## Phase 1: Alignment And Quality Gate

Code:

- `backend/app/pipeline/stages/alignment.py`
- `backend/app/services/extraction/quality_gate.py`
- `backend/app/services/landmarks/service.py`
- `backend/app/services/cranial_hull/service.py`

What happens:

1. Base and donor bytes are decoded to RGB using Pillow with EXIF orientation
   correction.
2. RetinaFace runs on both images.
3. The highest-scoring face after confidence filtering and NMS is selected.
4. A quality gate rejects faces that are too small or not frontal enough.
5. The stage builds scalp anchors for both base and donor.
6. If experimental geometry is enabled, it also creates heuristic dense
   landmarks and a cranial hull, then appends cranial anchors to the scalp anchors.

RetinaFace details:

- Input is letterboxed to `retinaface_input_size`, default `640`.
- Preprocessing converts RGB to BGR and subtracts `[104, 117, 123]`.
- Outputs are aligned by tensor last dimension:
  - `4`: bounding boxes.
  - `2`: confidence logits.
  - `10`: five facial landmarks.
- Priors use the PyTorch_RetinaFace style min sizes and steps from settings.
- Face score threshold default is `0.5`.
- NMS threshold default is `0.4`.

Quality gate details:

- Inter-eye distance must be at least `min_inter_eye_fraction` of image width,
  default `0.03`.
- Yaw is estimated heuristically from nose offset relative to eye midpoint.
- Absolute yaw must be no more than `max_abs_yaw_deg`, default `35`.
- This is not a full 3D pose estimator.

Scalp anchor details:

- Anchors are built from the five RetinaFace landmarks plus the detected face box.
- The points cover temples, fringe center, crown, nose, sides, cheeks, jaw, chin,
  shoulders, and bottom center.
- The logic uses eye direction, roll correction, face vertical distance, and box
  padding to reduce inward squeezing and preserve long-hair support.
- If `enable_experimental_geometry=true` (default), canonical anchors from the
  cranial hull are appended.

Nuance for review:

- The dense landmark service is named as if it could use a model, but today it is
  heuristic. That means the cranial hull and extra anchors are approximate.
- The same frontal quality gate applies to both base and donor images.
- Multiple faces are not handled beyond picking the highest score.

## Phase 2: Donor Hair Extraction

Code:

- `backend/app/pipeline/stages/extraction.py`
- `backend/app/services/extraction/hair_parser.py`
- `backend/app/shared/image_processing.py`

What happens:

1. Donor RGB is passed to `HairParser`.
2. BiSeNet/CelebAMask-HQ face parsing predicts semantic class logits.
3. The softmax probability for class `17` is used directly as the hair alpha.
4. Class `18` for hat can optionally be added if `include_hat_in_hair=true`.
5. Optional MODNet portrait gating can suppress hair predictions outside the
   person silhouette.
6. The alpha is amplified, morphologically stabilized, and converted to donor
   hair RGBA.
7. The donor hair alpha is decomposed into region masks for later optional
   region-aware warping and occlusion.

BiSeNet details:

- Input size defaults to `512x512`.
- Normalization uses ImageNet-style mean/std values.
- By default, parsing is run on a padded crop around the detected donor face
  rather than the entire image. This preserves more effective resolution for
  hair edges on wide/full-body images.
- Crop padding defaults:
  - top: `1.6 * face height`
  - bottom: `1.0 * face height`
  - horizontal: `0.8 * face width`
- The predicted crop alpha is pasted back into a full-image alpha canvas.

Alpha handling:

- `hair_alpha_gain` defaults to `1.15`.
- `hair_alpha_close_kernel` defaults to `5`.
- Morphological close fills small holes.
- A core mask is eroded from pixels above `0.35`; core pixels are raised to at
  least `hair_alpha_core_min_opacity`, default `1.0`.
- Edge alpha is gamma-adjusted with `hair_alpha_edge_gamma`, default `0.9`.
- RGB pixels where alpha is zero can be set to black to improve PNG compression.

Region masks:

- Legacy region map: `crown`, `fringe`, `sides`.
- `HairRegions` v2:
  - `roots`
  - `forehead_line`
  - `temples`
  - `side_strands`
  - `long_strands`
  - `shoulder_overlap`

Nuance for review:

- The extraction implementation is named `modnet`, but the primary hair mask is
  BiSeNet face parsing. MODNet is optional and disabled by default.
- The direct softmax alpha preserves soft hair boundaries better than argmax,
  but may include uncertain semantic regions.
- The face crop improves resolution, but hair far outside the crop can be missed.
- Region masks are simple normalized-coordinate bands, not anatomically segmented
  regions.

## Phase 3: Warping

Code:

- `backend/app/pipeline/stages/warping.py`
- `backend/app/pipeline/stages/warping_mls.py`
- `backend/app/services/warping/region_tps.py`
- `backend/app/services/warping/mls_rigid.py`

Default behavior:

- `pipeline_warp_impl` defaults to `tps`.
- Donor scalp anchors are mapped to base scalp anchors.
- An inverse thin-plate spline is solved so each base pixel samples a donor
  coordinate.
- `cv2.remap` warps donor hair RGB and donor alpha into the base image size.
- Warped alpha is stabilized with close, blur, and a minimum interior opacity.

TPS details:

- The implementation solves separate TPS systems for x and y.
- Control points are the target/base anchors.
- Values are source/donor x and y coordinates.
- Regularization defaults to `1e-3`.
- The generated `map_x` and `map_y` are full-resolution base-sized remap maps.

MLS alternative:

- `pipeline_warp_impl=mls` uses rigid Moving Least Squares.
- It builds a backward map from base pixels to donor pixels using fixed-point
  inversion.
- Maps are computed on a coarse grid and upsampled for speed.
- Important settings:
  - `mls_map_grid_long_edge`, default `48`.
  - `mls_inverse_max_iterations`, default `12`.
  - `mls_weight_alpha`, default `1.0`.
  - `mls_weight_eps`, default `2.0`.

Region-aware option:

- `enable_region_aware_tps=false` by default.
- If enabled, region stiffness values blend anchors between donor and base
  positions:
  - roots: `0.95`
  - forehead line: `0.90`
  - temples: `0.80`
  - side strands: `0.55`
  - long strands: `0.35`
  - shoulder overlap: `0.50`
- Higher stiffness keeps the donor shape closer to the original; lower stiffness
  adapts more to the base.
- Debug export can write a stiffness map.

Nuance for review:

- Default TPS is global and can over-deform some hairstyles.
- Region-aware TPS/MLS exists but is disabled by default and uses coarse
  heuristic region labeling.
- The region label list in `RegionAwareTpsWarper` has a fixed order and may not
  match every anchor added by experimental geometry.
- Warping has no learned understanding of hair direction, volume, curls, or
  physical layering.

## Phase 4: Harmonization

Code:

- `backend/app/pipeline/stages/harmonization.py`

Current behavior:

- Despite the implementation name `lab_transfer`, this stage currently passes
  the warped donor hair RGB through unchanged.
- The comment says this is intentional to preserve the donor hair lighting, hue,
  and saturation exactly.

Unused helper methods:

- `_build_reference_mask(...)` can build a surrounding ring mask.
- `_masked_stats(...)` can compute LAB channel statistics.

Nuance for review:

- There is no actual LAB color transfer in the current live path.
- If visual mismatch is a concern, this stage is the natural place to add color
  matching, exposure matching, or local tone adaptation.
- Preserving donor color may be desirable for hairstyle preview, but it can make
  lighting direction and camera white balance mismatches obvious.

## Phase 5: Blending And Compositing

Code:

- `backend/app/pipeline/stages/blending.py`
- `backend/app/services/compositing/service.py`
- `backend/app/services/depth/service.py`

What happens:

1. Base RGB, harmonized/warped hair RGB, and warped alpha are converted to float.
2. Alpha is median-blurred and Gaussian-blurred.
3. Optional depth/occlusion logic can modify alpha.
4. A contact shadow is generated from the hair-alpha boundary.
5. The base is darkened under the shadow.
6. A Laplacian pyramid blend combines base and donor hair.
7. Final output is stacked as RGBA with alpha set to `255` everywhere.

Alpha preparation:

- Median blur kernel: `5`.
- Gaussian blur sigma: `1.0`.
- Pixels above `0.55` are raised to at least `0.82`.

Contact shadow:

- Built from a morphological gradient on alpha.
- Shifted downward by 4 pixels.
- Blurred with `contact_shadow_blur_sigma`, default `5.0`.
- Multiplied by `contact_shadow_opacity`, default `0.15`.

Laplacian blending:

- Pyramid levels default to `4`.
- Mask pyramids are generated from the prepared alpha.
- Overlay and base Laplacians are blended at each level and reconstructed.

Optional depth occlusion:

- Disabled by default with `enable_depth_occlusion=false`.
- The current `OptionalDepthEstimator` is deterministic and CPU-friendly:
  grayscale, vertical Sobel gradient, and vertical position bias.
- If enabled, compositor splits front and back hair masks using either region
  masks or a depth quantile.
- Neck and shoulder masks suppress parts of back hair.

Nuance for review:

- The final alpha channel is fully opaque, so this endpoint returns a complete
  composited image rather than a transparent hair overlay.
- The depth estimator is not using a loaded ONNX model today.
- Occlusion is heuristic and likely needs stronger anatomical modeling for long
  hair around neck, ears, and shoulders.

## Debugging And Observability

Settings:

- `debug_export_enabled=false` by default.
- `debug_export_dir=debug/exports`.
- `export_landmarks_json=false`.

Potential debug exports:

- base and donor landmark overlays.
- base and donor cranial hull masks.
- warp stiffness map.
- depth map.
- front/back occlusion masks.

Response telemetry:

- The endpoint reports selected implementations and timings in headers.
- `X-Pipeline-Stage-Details` is the best compact view for frontend or manual
  inspection.

Nuance for review:

- Debug artifacts are written to disk only when enabled.
- There is no persistent request ID, input hash, or structured log tying debug
  outputs to a specific API request.

## Important Settings

The settings class is `backend/app/core/settings.py`; all environment variables
use the `HAIRSWAP_` prefix.

Common review knobs:

| Setting | Default | Purpose |
| --- | --- | --- |
| `pipeline_warp_impl` | `tps` | Selects TPS or MLS warp. |
| `enable_experimental_geometry` | `true` | Adds heuristic dense landmarks and cranial hull anchors. |
| `enable_region_aware_tps` | `false` | Enables region stiffness behavior in warp. |
| `enable_depth_occlusion` | `false` | Enables heuristic depth/occlusion compositing. |
| `refine_hair_with_portrait_matte` | `false` | Uses MODNet as a person-silhouette gate. |
| `face_crop_for_parsing` | `true` | Runs BiSeNet on a detected-face crop. |
| `hair_alpha_gain` | `1.15` | Boosts confident hair alpha. |
| `hair_alpha_close_kernel` | `5` | Fills small alpha holes. |
| `laplacian_pyramid_levels` | `4` | Blend pyramid depth. |
| `contact_shadow_opacity` | `0.15` | Strength of synthetic contact shadow. |
| `max_abs_yaw_deg` | `35.0` | Frontal-face rejection threshold. |

## Tests Covering The Pipeline

Main tests:

- `backend/tests/test_swap_pipeline.py`
- `backend/tests/test_extraction_service.py`
- `backend/tests/test_warping_service.py`
- `backend/tests/test_blending_service.py`
- `backend/tests/test_geometry_services.py`

Current test style:

- Endpoint tests use a stub pipeline.
- Pipeline orchestration is tested with no-op stages.
- TPS and MLS are tested for identity/shape behavior.
- Extraction tests use deterministic in-process model stubs.

Nuance for review:

- Tests validate plumbing and basic numerical behavior, but they do not verify
  visual quality.
- There are no golden-image regression tests for real donor/base pairs.
- There is no benchmark test for large images or worst-case TPS/MLS runtime.

## Current Implementation Risks

These are the main points worth asking a lead developer to review:

1. Stage naming can be misleading:
   - extraction implementation is called `modnet`, but BiSeNet is the main hair
     extractor.
   - harmonization implementation is called `lab_transfer`, but the live path is
     pass-through.
2. Geometry is mostly heuristic:
   - dense landmarks are generated from RetinaFace five-point landmarks and face
     box, not from a true dense landmark model.
   - cranial hull anchors are approximate.
3. Default warp is global TPS:
   - can distort roots and long strands.
   - region-aware stiffness exists but is not default.
4. Occlusion is incomplete:
   - long hair around ears, neck, shoulders, and clothing is not robustly layered.
   - depth is a simple image heuristic, not a learned monocular depth model.
5. The endpoint picks the highest-scoring face only:
   - no user selection or multi-face disambiguation.
6. Final output is a flattened composite:
   - no separate alpha overlay or intermediate artifact response is available.
7. The pipeline has limited production observability:
   - stage timing exists in response headers, but there is no durable trace,
     request ID, model version, or debug artifact manifest in the response.

## Suggested Questions For Lead Developer Review

1. Should the default warp remain TPS, or should MLS or region-aware TPS become
   the default for shape preservation?
2. Should we rename implementation IDs to match reality, for example
   `bisenet_hair_parser` and `passthrough_color`, to avoid confusion?
3. Should MODNet gating be enabled for specific input classes, or kept off unless
   background leakage is detected?
4. Do we need a real dense landmark or face mesh model before investing further
   in cranial hull geometry?
5. Should harmonization preserve donor hair color exactly, or perform local
   exposure/white-balance matching against the base portrait?
6. What acceptance criteria should define visual quality: hairline accuracy,
   strand preservation, no face occlusion, no shoulder artifacts, runtime, or a
   weighted combination?
7. Should the endpoint expose optional debug artifacts or a debug mode for lead
   review and QA?
8. Should we add golden-image tests and perceptual diff thresholds for common
   cases like short hair, long hair, bangs, curly hair, side profiles, and
   full-body donor photos?
