# Current Hair Swap Pipeline

This note captures the pre-refactor baseline and known geometry limitations.

- Detector: RetinaFace with 5 landmarks via `FrontalFaceGate`.
- Alignment: heuristic scalp anchors from eye/nose/mouth landmarks.
- Segmentation: BiSeNet hair parsing with optional MODNet gating.
- Warp: global uniform TPS over scalp anchors.
- Composite: LAB harmonization + Laplacian blend + contact shadow.

Known weaknesses:

- Insufficient cranial geometry causes poor forehead/temple fit.
- Uniform TPS stiffness deforms roots and long strands unnaturally.
- Long-hair occlusion around neck/shoulders is not anatomically ordered.
- No stage-level debug exports for geometry inspection.
