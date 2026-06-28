Four new built-in task batteries — `regression`, `image_quality`, `audio`, and
`retrieval` — plus a per-`Task` `update_fn` hook for metrics whose update step is
not `(preds, target)` (used by `retrieval`). LPIPS and PESQ/STOI sit behind the
optional `[image]` and `[audio]` extras. Each battery is exported from `mushin`.
