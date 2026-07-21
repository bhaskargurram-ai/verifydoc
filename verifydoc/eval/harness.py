"""The VerifyDocBench harness: signals x calibrators x metrics -> tables/figures.

Runs the whole §5 metric suite over any benchmark items (documents + schemas +
gold fields), for every confidence signal and calibrator, with a dedicated
calibration split (disjointness asserted), bootstrap CIs, and the conformal
guarantee row. ``scripts/run_benchmark.py`` is a thin CLI over this module;
``make results`` regenerates every table/figure from ``configs/``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from verifydoc.adapters.mock import MockAdapter
from verifydoc.calibration import (
    Calibrator,
    ConformalAbstention,
    HistogramBinning,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
    assert_disjoint,
    split_calibration,
)
from verifydoc.confidence import (
    combined_confidence,
    consensus,
    grounding_confidence,
    token_prob_confidence,
    verbalized_confidence,
)
from verifydoc.eval.calibration import adaptive_ece, brier, ece, nll, reliability_bins
from verifydoc.eval.extraction import score_fields
from verifydoc.eval.grounding import (
    box_grounding_accuracy,
    grounding_conditioned_correctness,
    mean_iou,
    pairwise_ious,
)
from verifydoc.eval.selective import auroc, coverage_at_risk, e_aurc, rc_curve
from verifydoc.eval.stats import bootstrap_ci
from verifydoc.grounding import ground_predictions
from verifydoc.types import FieldPrediction

SIGNALS = ("token_prob", "verbalized", "grounding", "consensus", "combined")
CALIBRATORS: dict[str, Callable[[], Calibrator]] = {
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": IsotonicCalibrator,
    "histogram": HistogramBinning,
}


@dataclass
class SignalData:
    """Per-signal pooled field outcomes across all documents."""

    doc_ids: list[str] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    correct: list[int] = field(default_factory=list)

    def subset(self, ids: set[str]) -> tuple[list[float], list[int]]:
        conf = [c for d, c in zip(self.doc_ids, self.conf) if d in ids]
        corr = [y for d, y in zip(self.doc_ids, self.correct) if d in ids]
        return conf, corr


def collect(bench: Sequence[Any], adapter: MockAdapter, k: int) -> dict[str, Any]:
    """Run the adapter over the benchmark and pool per-signal field outcomes."""
    signals: dict[str, SignalData] = {name: SignalData() for name in SIGNALS}
    extraction_reports = []
    pred_boxes, gold_boxes, box_correct = [], [], []

    for item in bench:
        samples = adapter.extract_samples(item.doc, item.schema, k)
        base = ground_predictions(samples[0], item.doc)
        cons = ground_predictions(consensus(samples), item.doc)

        variants: dict[str, list[FieldPrediction]] = {
            "token_prob": [
                p.model_copy(update={"confidence": token_prob_confidence(p) or 0.5}) for p in base
            ],
            "verbalized": [
                p.model_copy(update={"confidence": verbalized_confidence(p) or 0.5}) for p in base
            ],
            "grounding": [
                p.model_copy(update={"confidence": grounding_confidence(p)}) for p in base
            ],
            "consensus": cons,
            "combined": [
                p.model_copy(
                    update={
                        "confidence": combined_confidence(
                            {
                                "consensus": p.confidence,
                                "grounding": grounding_confidence(p),
                                "token_prob": token_prob_confidence(p),
                                "verbalized": verbalized_confidence(p),
                            }
                        )
                    }
                )
                for p in cons
            ],
        }
        for name, preds in variants.items():
            report = score_fields(preds, item.golds)
            data = signals[name]
            for fs in report.field_scores:
                data.doc_ids.append(item.doc.doc_id)
                data.conf.append(fs.confidence)
                data.correct.append(int(fs.correct))

        base_report = score_fields(base, item.golds)
        extraction_reports.append(base_report)

        gold_box_by_path = {g.path: g.gold_box for g in item.golds}
        for fs in score_fields(cons, item.golds).field_scores:
            gold_grounding = gold_box_by_path.get(fs.path)
            pred_grounding = next((p.grounding for p in cons if p.path == fs.path), None)
            pred_boxes.append(pred_grounding.bbox if pred_grounding else None)
            gold_boxes.append(gold_grounding.bbox if gold_grounding else None)
            box_correct.append(int(fs.correct))

    return {
        "signals": signals,
        "extraction_reports": extraction_reports,
        "boxes": (pred_boxes, gold_boxes, box_correct),
    }


def run_benchmark(cfg: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Execute the full harness from a pinned config; write tables + figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from benchmark.datasets import synthetic

    bench = synthetic.generate(n_docs=cfg.get("n_docs", 40), seed=cfg.get("seed", 0))
    adapter = MockAdapter(
        gold={item.doc.doc_id: item.golds for item in bench},
        error_rate=cfg.get("error_rate", 0.15),
        omit_rate=cfg.get("omit_rate", 0.05),
        hallucinate_rate=cfg.get("hallucinate_rate", 0.05),
        seed=cfg.get("seed", 0),
    )
    pooled = collect(bench, adapter, k=cfg.get("k", 5))
    signals: dict[str, SignalData] = pooled["signals"]

    doc_ids = [item.doc.doc_id for item in bench]
    cal_ids, test_ids = split_calibration(
        doc_ids, cfg.get("calibration_fraction", 0.5), seed=cfg.get("seed", 0)
    )
    assert_disjoint(cal_ids, test_ids)
    cal_set, test_set = set(cal_ids), set(test_ids)

    alphas = cfg.get("alphas", [0.02, 0.05])
    n_boot = cfg.get("n_boot", 300)
    summary: dict[str, Any] = {"n_docs": len(bench), "n_cal": len(cal_ids), "n_test": len(test_ids)}

    calib_rows: list[dict[str, Any]] = []
    select_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []
    figures: dict[str, Any] = {}
    for name in SIGNALS:
        cal_conf, cal_corr = signals[name].subset(cal_set)
        test_conf, test_corr = signals[name].subset(test_set)
        variants: dict[str, list[float]] = {"raw": list(test_conf)}
        for cal_name, factory in CALIBRATORS.items():
            calibrator = factory().fit(cal_conf, cal_corr)
            variants[cal_name] = [float(v) for v in calibrator.transform(test_conf)]
        for variant, conf in variants.items():
            calib_rows.append(
                {
                    "signal": name,
                    "calibrator": variant,
                    "ece": ece(conf, test_corr),
                    "adaptive_ece": adaptive_ece(conf, test_corr),
                    "brier": brier(conf, test_corr),
                    "nll": nll(conf, test_corr),
                }
            )
        row: dict[str, Any] = {
            "signal": name,
            "e_aurc": e_aurc(test_conf, test_corr),
            "auroc": auroc(test_conf, test_corr),
        }
        for alpha in alphas:
            cov, _thr = coverage_at_risk(test_conf, test_corr, alpha)
            row[f"coverage@{int(alpha * 100)}%"] = cov
        ci = bootstrap_ci(
            lambda c, y: float(e_aurc(list(c), list(y.astype(int)))),
            test_conf,
            test_corr,
            n_boot=n_boot,
            seed=cfg.get("seed", 0),
        )
        row["e_aurc_ci"] = f"[{ci.lo:.4f}, {ci.hi:.4f}]"
        select_rows.append(row)

        for alpha in alphas:
            policy = ConformalAbstention(alpha=alpha).fit(cal_conf, cal_corr)
            mask = policy.accept(test_conf)
            achieved = (
                float(1.0 - np.asarray(test_corr, dtype=float)[mask].mean()) if mask.any() else 0.0
            )
            conformal_rows.append(
                {
                    "signal": name,
                    "alpha": alpha,
                    "threshold": policy.threshold_,
                    "test_coverage": float(mask.mean()),
                    "achieved_risk": achieved,
                    "guarantee_held": achieved <= alpha + 1e-9,
                }
            )
        figures[name] = (test_conf, test_corr)

    reports = pooled["extraction_reports"]
    extraction_row = {
        "precision": float(np.mean([r.precision for r in reports])),
        "recall": float(np.mean([r.recall for r in reports])),
        "f1": float(np.mean([r.f1 for r in reports])),
        "exact_match": float(np.mean([r.exact_match_rate for r in reports])),
        "omission_rate": float(np.mean([r.omission_rate for r in reports])),
        "hallucination_rate": float(np.mean([r.hallucination_rate for r in reports])),
    }

    pred_boxes, gold_boxes, box_corr = pooled["boxes"]
    grounding_row = {
        "box_acc@0.5": box_grounding_accuracy(pred_boxes, gold_boxes, 0.5),
        "box_acc@0.7": box_grounding_accuracy(pred_boxes, gold_boxes, 0.7),
        "mean_iou": mean_iou(pred_boxes, gold_boxes),
    }
    gc = grounding_conditioned_correctness(box_corr, pred_boxes, gold_boxes, tau=0.5)
    grounding_row.update(
        {
            "acc_grounded": gc.accuracy_grounded,
            "acc_ungrounded": gc.accuracy_ungrounded,
            "grounding_gap": gc.gap,
        }
    )
    # sanity: every field either has a gold box or was hallucinated
    assert len(pairwise_ious(pred_boxes, gold_boxes)) == len(box_corr)

    _write_tables(out, extraction_row, calib_rows, select_rows, conformal_rows, grounding_row)
    _write_figures(out, figures, signals, cal_set, test_set)

    summary.update(
        {
            "extraction": extraction_row,
            "grounding": grounding_row,
            "best_ece": min(calib_rows, key=lambda r: float(r["ece"])),
            "best_coverage": max(
                select_rows, key=lambda r: float(r[f"coverage@{int(alphas[0] * 100)}%"])
            ),
            "tables": sorted(p.name for p in out.glob("*.md")),
        }
    )
    return summary


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _table(rows: list[dict[str, Any]]) -> str:
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _write_tables(
    out: Path,
    extraction: dict[str, Any],
    calib: list[dict[str, Any]],
    select: list[dict[str, Any]],
    conformal: list[dict[str, Any]],
    grounding: dict[str, Any],
) -> None:
    (out / "extraction.md").write_text(
        "# Extraction quality (single-run baseline)\n\n" + _table([extraction]),
        encoding="utf-8",
    )
    (out / "calibration.md").write_text(
        "# Calibration: signal x calibrator (test split)\n\n" + _table(calib),
        encoding="utf-8",
    )
    (out / "selective.md").write_text(
        "# Selective prediction by signal (raw confidence, test split)\n\n" + _table(select),
        encoding="utf-8",
    )
    (out / "conformal.md").write_text(
        "# Conformal abstention: guarantee vs forced review\n\n" + _table(conformal),
        encoding="utf-8",
    )
    (out / "grounding.md").write_text(
        "# Grounding quality (consensus predictions)\n\n" + _table([grounding]),
        encoding="utf-8",
    )


def _write_figures(
    out: Path,
    figures: dict[str, Any],
    signals: dict[str, SignalData],
    cal_set: set[str],
    test_set: set[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (conf, corr) in figures.items():
        coverage, risk = rc_curve(conf, corr)
        ax.plot(coverage, risk, label=name)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_title("Risk-coverage by confidence signal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rc_curves.png", dpi=150)
    plt.close(fig)

    conf, corr = signals["consensus"].subset(test_set)
    cal_conf, cal_corr = signals["consensus"].subset(cal_set)
    calibrated = [float(v) for v in IsotonicCalibrator().fit(cal_conf, cal_corr).transform(conf)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, data) in zip(
        axes, [("raw consensus", conf), ("isotonic-calibrated", calibrated)]
    ):
        bins = reliability_bins(data, corr, n_bins=10)
        ax.bar(
            [b.mean_confidence for b in bins],
            [b.accuracy for b in bins],
            width=0.08,
            alpha=0.7,
            label="accuracy",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("confidence")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)
