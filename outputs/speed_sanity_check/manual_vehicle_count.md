# Manual Vehicle Count Anchor

> Review date: 2026-05-10  
> Reviewed artifact: `project/outputs/run_bytetrack_smooth_1min/annotated.mp4`  
> Assisted by: `project/outputs/speed_sanity_check/track_review_summary.csv`  
> Purpose: provide a count anchor for interpreting `tracks_created`, not a formal MOT ground truth.

---

## Counting Rule

- Count physical vehicles, not tracker IDs.
- Main count includes vehicles with enough visible trajectory evidence to be confidently treated as real vehicles.
- Very short far-field, edge-only, or partial appearances are marked as uncertain instead of being forced into the main count.
- This count is used to interpret tracking fragmentation direction, not to compute formal MOTA / IDF1.

---

## Count Result

| Category | Count | Notes |
|---|---:|---|
| Confirmed visible vehicles | 42 | Tracks with persistent visual / trajectory evidence, typically >=20 trajectory rows or clear ROI coverage. |
| Uncertain short / edge / far-field candidates | 19 | Brief detections, partial edge appearances, or tiny far-field objects that are hard to confirm as unique physical vehicles. |
| ByteTrack `tracks_created` | 61 | Equals confirmed + uncertain candidates in this review. |
| IoU baseline `tracks_created` | 89 | Higher than both the confirmed count and the ByteTrack track count, indicating more fragmentation. |

---

## Confirmed Vehicle Track IDs

`1, 2, 3, 5, 6, 7, 8, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 32, 33, 35, 36, 38, 39, 40, 42, 43, 45, 47, 48, 49, 51, 53, 54, 56, 57, 59`

## Uncertain Candidate Track IDs

`4, 9, 11, 15, 21, 24, 25, 31, 34, 37, 41, 44, 46, 50, 52, 55, 58, 60, 61`

---

## Interpretation

ByteTrack created **61** total tracks. The manual review found **42 confirmed visible vehicles** plus **19 uncertain short / far-field / edge candidates**. This means ByteTrack's track count is plausible as an upper-bound count for all detected candidates, but it should not be described as an exact physical vehicle count.

Compared with the IoU baseline's **89** created tracks, ByteTrack substantially reduces track fragmentation. The evidence supports the qualitative claim that ByteTrack is more stable than IoU, but it does not replace a formal ground-truth MOT evaluation.

Recommended report wording:

> A manual review identified 42 confidently visible vehicles and 19 additional uncertain short or far-field candidates. ByteTrack produced 61 tracks, which aligns with the review's upper-bound candidate count, while the IoU baseline produced 89 tracks. This supports the conclusion that ByteTrack reduces fragmentation, although formal ID-switch accuracy remains future work.
