# AGENTS.md

## Purpose

This repository implements a **Metashape Professional Python pipeline** for dual-fisheye video preprocessing, masking, alignment, and post-alignment frame reduction.

The system is designed for:

- **Development on macOS**
- **Implementation with VS Code + Codex**
- **Runtime orchestration inside Metashape Python**
- **Behavior verification on Windows as needed**

---

## Source of Truth

When making changes, use the following priority order:

1. `docs/metashape_dual_fisheye_pipeline_spec.md`
2. `docs/reference/metashape_api_checklist_dual_fisheye.md`
3. `docs/reference/metashape_python_api_2_3_0.pdf`
4. Existing repository structure and code conventions
5. This `AGENTS.md`

Do not silently override the spec or the API checklist.
If a requirement is unclear, add a `TODO:` comment instead of guessing.

---

## Non-Negotiable Rules

These rules must not be broken:

1. **All orchestration must run from Metashape Python**
2. Video extraction must use **ffmpeg / ffprobe**
3. Input video is **2-stream MP4**
4. **Do not keep only one side of a pair**
5. The keep rule must remain:
   - **if either side is acceptable, keep both**
6. Metashape import must use **`MultiplaneLayout`**
7. Sensor type must be **`Metashape.Sensor.Type.Fisheye`**
8. Do not over-prune images during preprocessing
9. Redundant image reduction must happen **after alignment**
10. Keep diffs small and focused
11. Do not edit unrelated files
12. Preserve logging and error handling

---

## Metashape API Rules

- Use the checklist as the **current API usage reference**.
- Use the PDF as the **authoritative reference** when an API detail is uncertain.
- Do not let old research notes or old sample snippets override the checklist.
- If `MultiplaneLayout` behavior, `filegroups`, mask import behavior, or alignment flags are uncertain on the current build, leave a `TODO:` and keep the logic localized.

---

## Required Implementation Behavior

- Preserve front/back timestamp pairing.
- Keep the rule: **if either front or back passes threshold, keep both**.
- Prefer **camera-level mask assignment** using `Metashape.Mask()` plus `mask.load(...)` / `mask.setImage(...)` and `camera.mask = mask` for the initial implementation.
- Treat bulk mask import methods as optional until validated on the current Metashape build.
- Treat `filenames` / `filegroups` layout for `MultiplaneLayout` as **validation-required**, not assumed.
- Use current API argument names for `matchPhotos(...)` and `alignCameras(...)`.

---

## Working Style

- Read the spec first.
- Then read the API checklist.
- Make the smallest reasonable change set.
- Prefer phase-based progress:
  - Phase 1: extraction + blur filtering
  - Phase 2: masking + import
  - Phase 3: alignment + overlap reduction
- Update `README.md` when behavior or setup changes.
- Keep the code debuggable.

---

## What Not to Do

Do **not** do any of the following unless the spec explicitly changes:

- keep only one image from a front/back pair
- replace ffmpeg extraction with OpenCV-only video decoding
- replace `MultiplaneLayout` with another import mode
- aggressively prune images before alignment
- remove logs to simplify code
- change unrelated files
- silently change core behavior
- invent unsupported Metashape API behavior
- let old examples from background notes override the current API checklist

---

## Change Reporting Rules

After each task, report:

1. changed files
2. what changed in each file
3. any unresolved `TODO`
4. any risk or uncertainty
5. how to run or verify the change if behavior changed
