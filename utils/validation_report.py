"""Human-readable validation results for target-selected source checkpoints."""

import os
from pathlib import Path


def write_source_target_report(path, checkpoint_path, epoch, source_score,
                               source_split, target_split, target_metrics):
    """Replace the report whenever target validation selects a new source model."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    lines = [
        "# Validation of the target-selected source checkpoint",
        "",
        "Selection: highest target-validation mIoU; target labels are used for model selection, not gradient updates.",
        "",
        "- Checkpoint: `{}`".format(checkpoint_path),
        "- Checkpoint epoch: {}".format(epoch),
        "- Source validation set: `{}`".format(source_split),
        "- Source validation mIoU: {:.2f}%".format(source_score * 100),
        "- Target validation set: `{}`".format(target_split),
        "- Target validation mIoU: {:.2f}%".format(target_metrics["miou"] * 100),
        "- Target validation mF1: {:.2f}%".format(target_metrics["mf1"] * 100),
        "",
        "| Class | IoU (%) | F1 (%) |",
        "|---|---:|---:|",
    ]
    for row in target_metrics["classes"]:
        lines.append("| {} | {:.2f} | {:.2f} |".format(
            row["name"], row["iou"] * 100, row["f1"] * 100
        ))
    ignored = target_metrics["ignored_class_ids"]
    if ignored:
        lines.extend(["", "Mean metrics exclude class IDs: {}.".format(
            ", ".join(map(str, ignored))
        )])

    report_path = Path(path)
    temporary_path = report_path.with_name(report_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines) + "\n")
    os.replace(str(temporary_path), str(report_path))
