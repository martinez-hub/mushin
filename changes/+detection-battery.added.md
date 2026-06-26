`compare(task="detection")` — compare trained object detectors across seeds over
the full `torchmetrics.detection` bounding-box family (mean-average-precision plus
the IoU/GIoU/CIoU/DIoU variants), reporting every scalar metric with Holm-corrected
significance. Needs the optional `mushin-py[detection]` extra.
