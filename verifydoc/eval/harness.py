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
    GROUP_TAXONOMIES,
    Calibrator,
    ConformalAbstention,
    GroupConformalAbstention,
    GroupPartitionSelector,
    HistogramBinning,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
    assert_disjoint,
    characterize,
    grounded_group,
    split_calibration,
)
from verifydoc.confidence import (
    combined_confidence,
    consensus,
    grounding_confidence,
    token_prob_confidence,
    verbalized_confidence,
)
from verifydoc.eval.calibration import adaptive_ece, brier, ece, nll, reliability_bins, smooth_ece
from verifydoc.eval.extraction import score_fields
from verifydoc.eval.grounding import (
    box_grounding_accuracy,
    grounding_conditioned_correctness,
    mean_iou,
    pairwise_ious,
)
from verifydoc.eval.selective import augrc, auroc, coverage_at_risk, e_aurc, rc_curve
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


def collect(bench: Sequence[Any], adapter: Any, k: int) -> dict[str, Any]:
    """Run the adapter over the benchmark and pool per-signal field outcomes."""
    signals: dict[str, SignalData] = {name: SignalData() for name in SIGNALS}
    extraction_reports = []
    pred_boxes, gold_boxes, box_correct = [], [], []
    learned_ids: list[str] = []
    learned_sigs: list[dict[str, float | None]] = []
    learned_correct: list[int] = []
    # per-field records for grounding-conditioned (Mondrian) conformal: the
    # combined-signal prediction keeps its grounding, so the group taxonomy
    # (grounded vs ungrounded) and the confidence bar share one object.
    grouped_ids: list[str] = []
    grouped_preds: list[FieldPrediction] = []
    grouped_correct: list[int] = []

    for item in bench:
        samples = adapter.extract_samples(item.doc, item.schema, k)
        base = ground_predictions(samples[0], item.doc)
        cons = ground_predictions(consensus(samples), item.doc)

        sig_by_path: dict[str, dict[str, float | None]] = {
            p.path: {
                "consensus": p.confidence,
                "grounding": grounding_confidence(p),
                "token_prob": token_prob_confidence(p),
                "verbalized": verbalized_confidence(p),
            }
            for p in cons
        }
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
                p.model_copy(update={"confidence": combined_confidence(sig_by_path[p.path])})
                for p in cons
            ],
        }
        for name, preds in variants.items():
            report = score_fields(preds, item.golds)
            data = signals[name]
            combined_by_path = {p.path: p for p in preds} if name == "combined" else {}
            for fs in report.field_scores:
                data.doc_ids.append(item.doc.doc_id)
                data.conf.append(fs.confidence)
                data.correct.append(int(fs.correct))
                if name == "combined":
                    learned_ids.append(item.doc.doc_id)
                    learned_sigs.append(sig_by_path[fs.path])
                    learned_correct.append(int(fs.correct))
                    pred = combined_by_path.get(fs.path)
                    if pred is not None:
                        grouped_ids.append(item.doc.doc_id)
                        grouped_preds.append(pred)
                        grouped_correct.append(int(fs.correct))

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
        "learned": (learned_ids, learned_sigs, learned_correct),
        "grouped": (grouped_ids, grouped_preds, grouped_correct),
    }


def run_benchmark(cfg: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Execute the full harness from a pinned config; write tables + figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = cfg.get("dataset", "synthetic")
    if dataset == "synthetic":
        from benchmark.datasets import synthetic

        bench = synthetic.generate(n_docs=cfg.get("n_docs", 40), seed=cfg.get("seed", 0))
    elif dataset == "cord":
        from benchmark.datasets import cord

        bench = cord.load(
            split=cfg.get("split", "validation"),
            limit=cfg.get("limit", 100),
            with_images=cfg.get("extractor", "mock") != "mock",
        )
    elif dataset == "funsd":
        from benchmark.datasets import funsd

        bench = funsd.load(split=cfg.get("split", "testing"), limit=cfg.get("limit"))
    elif dataset == "sroie":
        from benchmark.datasets import sroie

        bench = sroie.load(split=cfg.get("split", "test"), limit=cfg.get("limit", 100))
    else:
        raise ValueError(f"unknown dataset {dataset!r} (available: synthetic, cord, funsd, sroie)")
    extractor = cfg.get("extractor", "mock")
    if extractor == "mock":
        adapter: Any = MockAdapter(
            gold={item.doc.doc_id: item.golds for item in bench},
            error_rate=cfg.get("error_rate", 0.15),
            omit_rate=cfg.get("omit_rate", 0.05),
            hallucinate_rate=cfg.get("hallucinate_rate", 0.05),
            seed=cfg.get("seed", 0),
        )
    else:
        from verifydoc.adapters import get_adapter

        adapter = get_adapter(extractor, **cfg.get("adapter_kwargs", {}))
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
    summary: dict[str, Any] = {
        "dataset": dataset,
        "extractor": extractor,
        "n_docs": len(bench),
        "n_cal": len(cal_ids),
        "n_test": len(test_ids),
    }

    # learned combiner row: logistic fusion fit ONLY on calibration-split fields
    # (# DECISION: its cal-split confidences are in-sample for the conformal
    # threshold fit, same convention as calibrators fit on the same split)
    if cfg.get("learned_combiner", True):
        from verifydoc.confidence.learned import LearnedCombiner

        l_ids, l_sigs, l_corr = pooled["learned"]
        cal_sigs = [s for d, s in zip(l_ids, l_sigs) if d in cal_set]
        cal_y = [y for d, y in zip(l_ids, l_corr) if d in cal_set]
        combiner = LearnedCombiner().fit(cal_sigs, cal_y)
        signals["learned"] = SignalData(
            doc_ids=list(l_ids),
            conf=[float(v) for v in combiner.predict(l_sigs)],
            correct=list(l_corr),
        )

    signal_names = list(SIGNALS) + (["learned"] if "learned" in signals else [])
    calib_rows: list[dict[str, Any]] = []
    select_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []
    figures: dict[str, Any] = {}
    for name in signal_names:
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
                    "smooth_ece": smooth_ece(conf, test_corr),
                    "brier": brier(conf, test_corr),
                    "nll": nll(conf, test_corr),
                }
            )
        row: dict[str, Any] = {
            "signal": name,
            "e_aurc": e_aurc(test_conf, test_corr),
            "augrc": augrc(test_conf, test_corr),
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

    grouped_rows = grouped_conformal_rows(pooled["grouped"], cal_set, test_set, alphas)
    ablation_rows = grouping_ablation_rows(pooled["grouped"], cal_set, test_set, alphas)
    _write_tables(
        out,
        extraction_row,
        calib_rows,
        select_rows,
        conformal_rows,
        grounding_row,
        grouped_rows,
        ablation_rows,
    )
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
    grouped_gains = [r for r in grouped_rows if r["method"] == "grounded-group"]
    if grouped_gains:
        summary["grouped_conformal"] = max(grouped_gains, key=lambda r: float(r["coverage_gain"]))
    if ablation_rows:
        summary["best_taxonomy"] = max(ablation_rows, key=lambda r: float(r["gain_vs_pooled"]))
    # the calibration-split predictor of when grouping helps (paper diagnostic)
    g_ids, g_preds, g_corr = pooled["grouped"]
    cal_g = [(p, y) for d, p, y in zip(g_ids, g_preds, g_corr) if d in cal_set]
    if len(cal_g) >= 4:
        report = characterize(
            [p for p, _ in cal_g], [y for _, y in cal_g], grounded_group, alpha=alphas[0]
        )
        summary["characterization"] = {
            "predicted_gain": report.predicted_gain,
            "error_separation": report.error_separation,
            "within_group_auroc": report.within_group_auroc,
            "recommend_grouped": report.recommend,
        }
    return summary


def grouped_conformal_rows(
    grouped: tuple[list[str], list[FieldPrediction], list[int]],
    cal_set: set[str],
    test_set: set[str],
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    """Pooled (marginal) vs grounding-conditioned conformal, at each risk target.

    The paper's headline selective-prediction result: partitioning fields by
    provenance (grounded vs ungrounded) and fitting a *per-group* conformal
    threshold accepts more fields at the SAME guaranteed risk than one pooled
    threshold. Both policies are fit on the calibration split only and share
    the combined-signal confidence, so the comparison is apples-to-apples and
    isolates the effect of conditioning. Returns two rows per alpha (marginal,
    grounded-group) with the coverage gain and per-group detail.
    """
    g_ids, g_preds, g_corr = grouped
    cal_preds = [p for d, p in zip(g_ids, g_preds) if d in cal_set]
    cal_y = [y for d, y in zip(g_ids, g_corr) if d in cal_set]
    test_preds = [p for d, p in zip(g_ids, g_preds) if d in test_set]
    test_y = np.asarray([y for d, y in zip(g_ids, g_corr) if d in test_set], dtype=float)
    rows: list[dict[str, Any]] = []
    if not cal_preds or test_y.size == 0:
        return rows
    cal_conf = [p.confidence for p in cal_preds]
    test_conf = [p.confidence for p in test_preds]
    groups = np.array([grounded_group(p) for p in test_preds])
    for alpha in alphas:
        marginal = ConformalAbstention(alpha=alpha).fit(cal_conf, cal_y)
        m_mask = marginal.accept(test_conf)
        m_cov = float(m_mask.mean())
        m_risk = float(1.0 - test_y[m_mask].mean()) if m_mask.any() else 0.0

        policy = GroupConformalAbstention(alpha=alpha).fit(cal_preds, cal_y)
        g_mask = policy.accept(test_preds)
        g_cov = float(g_mask.mean())
        g_risk = float(1.0 - test_y[g_mask].mean()) if g_mask.any() else 0.0
        detail = "; ".join(
            f"{g}: {float(g_mask[groups == g].mean()) if (groups == g).any() else 0.0:.0%} acc "
            f"@thr={policy.threshold_for(str(g)):.3f}"
            for g in sorted(set(groups.tolist()))
        )
        rows.append(
            {
                "method": "marginal",
                "alpha": alpha,
                "test_coverage": m_cov,
                "achieved_risk": m_risk,
                "guarantee_held": m_risk <= alpha + 1e-9,
                "coverage_gain": 0.0,
                "detail": f"pooled thr={marginal.threshold_:.3f}",
            }
        )
        rows.append(
            {
                "method": "grounded-group",
                "alpha": alpha,
                "test_coverage": g_cov,
                "achieved_risk": g_risk,
                "guarantee_held": g_risk <= alpha + 1e-9,
                "coverage_gain": g_cov - m_cov,
                "detail": detail,
            }
        )
    return rows


def grouping_ablation_rows(
    grouped: tuple[list[str], list[FieldPrediction], list[int]],
    cal_set: set[str],
    test_set: set[str],
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    """Coverage at each risk target for EVERY provenance taxonomy vs pooled.

    The paper's grouping ablation: which partition (grounded, support-bin,
    value-length, field-type, cross-products) best recovers coverage, plus the
    calibration-split-*selected* partition (:class:`GroupPartitionSelector`,
    validity-preserving via a select/fit sub-split). All fit on calibration only.
    """
    g_ids, g_preds, g_corr = grouped
    cal_preds = [p for d, p in zip(g_ids, g_preds) if d in cal_set]
    cal_y = [y for d, y in zip(g_ids, g_corr) if d in cal_set]
    test_preds = [p for d, p in zip(g_ids, g_preds) if d in test_set]
    test_y = np.asarray([y for d, y in zip(g_ids, g_corr) if d in test_set], dtype=float)
    rows: list[dict[str, Any]] = []
    if len(cal_preds) < 4 or test_y.size == 0:
        return rows
    cal_conf = [p.confidence for p in cal_preds]
    test_conf = [p.confidence for p in test_preds]

    def _cov_risk(mask: np.ndarray) -> tuple[float, float]:
        return (
            float(mask.mean()),
            float(1.0 - test_y[mask].mean()) if mask.any() else 0.0,
        )

    for alpha in alphas:
        marginal = ConformalAbstention(alpha=alpha).fit(cal_conf, cal_y)
        m_cov, m_risk = _cov_risk(marginal.accept(test_conf))
        rows.append(
            {
                "taxonomy": "marginal(pooled)",
                "alpha": alpha,
                "coverage": m_cov,
                "achieved_risk": m_risk,
                "guarantee_held": m_risk <= alpha + 1e-9,
                "gain_vs_pooled": 0.0,
            }
        )
        for name, fn in GROUP_TAXONOMIES.items():
            policy = GroupConformalAbstention(alpha=alpha, group_of=fn).fit(cal_preds, cal_y)
            cov, risk = _cov_risk(policy.accept(test_preds))
            rows.append(
                {
                    "taxonomy": name,
                    "alpha": alpha,
                    "coverage": cov,
                    "achieved_risk": risk,
                    "guarantee_held": risk <= alpha + 1e-9,
                    "gain_vs_pooled": cov - m_cov,
                }
            )
        half = len(cal_preds) // 2
        if half >= 2:
            selector = GroupPartitionSelector(alpha=alpha).fit(
                cal_preds[:half], cal_y[:half], cal_preds[half:], cal_y[half:]
            )
            cov, risk = _cov_risk(selector.accept(test_preds))
            rows.append(
                {
                    "taxonomy": f"selected:{selector.selected_}",
                    "alpha": alpha,
                    "coverage": cov,
                    "achieved_risk": risk,
                    "guarantee_held": risk <= alpha + 1e-9,
                    "gain_vs_pooled": cov - m_cov,
                }
            )
    return rows


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
    grouped: list[dict[str, Any]],
    ablation: list[dict[str, Any]] | None = None,
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
    if grouped:
        (out / "grouped_conformal.md").write_text(
            "# Grounding-conditioned vs marginal conformal (combined signal, test split)\n\n"
            "Per-group conformal accepts more fields at the same guaranteed risk by "
            "applying a lower confidence bar to grounded fields and a stricter bar to "
            "ungrounded ones. `coverage_gain` is grounded-group coverage minus marginal.\n\n"
            + _table(grouped),
            encoding="utf-8",
        )
    if ablation:
        (out / "grouping_ablation.md").write_text(
            "# Provenance-taxonomy ablation (coverage @ risk vs pooled conformal)\n\n"
            "Coverage each grouping taxonomy achieves at the target risk, plus the "
            "calibration-split-selected partition. `gain_vs_pooled` is coverage minus "
            "marginal (pooled) conformal at the same alpha — the effect of conditioning "
            "on that provenance taxonomy.\n\n" + _table(ablation),
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
