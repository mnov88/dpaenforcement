import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from scripts.parser.ingest import segment_records, parse_record
from scripts.parser.enums import build_enum_whitelist
from scripts.clean.wide_output import clean_csv_to_wide
from scripts.clean.long_tables import LongEmitter
from scripts.clean.consistency import run_consistency_checks
from scripts.clean.qa_summary import create_qa_summary
from scripts.export.parquet_export import ParquetExporter
from scripts.export.arrow_export import ArrowExporter
from scripts.export.graph_export import GraphExporter
from scripts.export.stats_export import StatisticalPackageExporter
from scripts.export.ml_export import MLFeatureExporter
from scripts.analysis import fine_reconciliation
from scripts.analysis import build_feature_matrix as feature_matrix_module


DEFAULT_PROMPT = Path("analyzed-decisions/data-extraction-prompt-sent-to-ai.md")
DEFAULT_ENUM_OUT = Path("resources/enum_whitelist.json")
DEFAULT_INPUT_CSV = Path("analyzed-decisions/master-analyzed-data-unclean.csv")
DEFAULT_WIDE_CSV = Path("outputs/cleaned_wide.csv")
DEFAULT_VALIDATION_JSON = Path("outputs/validation_report.json")
DEFAULT_LONG_DIR = Path("outputs/long_tables")
DEFAULT_CONSISTENCY_JSON = Path("outputs/consistency_report.json")
DEFAULT_QA_SUMMARY_CSV = Path("outputs/qa_summary.csv")
DEFAULT_HUMAN_FINES_CSV = Path("raw-data/all_gdpr_fines_raw_human_annotations.csv")
DEFAULT_FEATURE_MATRIX_PARQUET = Path("outputs/analysis/feature_matrix.parquet")
DEFAULT_FEATURE_MATRIX_METADATA = Path("outputs/analysis/feature_matrix_metadata.json")


def run_fine_reconciliation(
    *,
    ai_csv: Path,
    human_csv: Path,
    out_csv: Path,
    tolerance: float,
    comparison_csv: Optional[Path] = None,
    summary_json: Optional[Path] = None,
) -> Tuple[fine_reconciliation.ComparisonSummary, int]:
    rows = fine_reconciliation.load_ai_dataset(ai_csv)
    ai_fines = fine_reconciliation.load_ai_fines(rows)
    human_fines = fine_reconciliation.load_human_fines(human_csv)
    comparisons = fine_reconciliation.compare_fines(
        ai_fines, human_fines, tolerance=tolerance
    )
    summary = fine_reconciliation.summarise_comparisons(comparisons)

    updated_rows, overrides = fine_reconciliation.apply_human_overrides(
        rows, human_fines
    )
    fine_reconciliation.write_csv(updated_rows, out_csv)

    if comparison_csv:
        fine_reconciliation.write_comparison_csv(comparisons, comparison_csv)
    if summary_json:
        summary_json.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return summary, len(overrides)


def cmd_reconcile_fines(args: argparse.Namespace) -> int:
    summary, overrides = run_fine_reconciliation(
        ai_csv=Path(args.wide_csv),
        human_csv=Path(args.human_csv),
        out_csv=Path(args.out_csv),
        tolerance=args.tolerance,
        comparison_csv=Path(args.comparison_csv) if args.comparison_csv else None,
        summary_json=Path(args.summary_json) if args.summary_json else None,
    )

    print(
        json.dumps(
            {
                "updated_records": overrides,
                "summary": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_feature_matrix(effective_wide_csv: Path, args: argparse.Namespace) -> None:
    feature_out = Path(args.feature_matrix_parquet) if args.feature_matrix_parquet else DEFAULT_FEATURE_MATRIX_PARQUET
    metadata_out = Path(args.feature_matrix_metadata) if args.feature_matrix_metadata else DEFAULT_FEATURE_MATRIX_METADATA
    artifacts = feature_matrix_module.build_feature_matrix(effective_wide_csv)
    feature_matrix_module._save_outputs(artifacts, feature_out, metadata_out)
    print(f"Wrote feature matrix to {feature_out}\nWrote feature metadata to {metadata_out}")


def _run_evenness_pipeline(effective_wide_csv: Path, args: argparse.Namespace) -> None:
    from scripts.evenness.config import EvennessPaths
    from scripts.evenness.data import build_fact_matrix
    from scripts.evenness.omniscan import run_omniscan
    from scripts.evenness.foundation import run_phase_one
    from scripts.evenness.uniformity import run_phase_two
    from scripts.evenness.phase_three import run_phase_three

    default_paths = EvennessPaths()
    evenness_wide_path = Path(args.evenness_wide_csv) if args.evenness_wide_csv else default_paths.wide_csv
    evenness_wide_path.parent.mkdir(parents=True, exist_ok=True)
    if evenness_wide_path.resolve() != effective_wide_csv.resolve():
        shutil.copy2(effective_wide_csv, evenness_wide_path)
        print(f"Copied cleaned wide CSV to {evenness_wide_path} for evenness analyses")

    paths = EvennessPaths(wide_csv=evenness_wide_path)

    print("Starting evenness Phase 0 (omni-scan)...")
    run_omniscan(
        paths=paths,
        use_gpu=getattr(args, "evenness_use_gpu", False),
        light=getattr(args, "evenness_light", False),
    )

    print("Starting evenness Phase 1 (matching & twins)...")
    fact_df = build_fact_matrix(evenness_wide_path, discussed_only=getattr(args, "evenness_discussed_only", False))
    run_phase_one(fact_df, paths=paths)

    print("Starting evenness Phase 2 (uniformity tests)...")
    run_phase_two(paths=paths)

    print("Starting evenness Phase 3 (drivers & policy)...")
    run_phase_three(
        paths=paths,
        outcome=getattr(args, "evenness_phase_three_outcome", "fine_log1p"),
        randomization_permutations=getattr(args, "evenness_phase_three_permutations", 10),
    )

    print("Completed evenness Phases 0-3")


def cmd_build_enum_whitelist(args: argparse.Namespace) -> int:
    prompt_path = Path(args.prompt_path)
    out_path = Path(args.out)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    whitelist = build_enum_whitelist(prompt_text, source_path=str(prompt_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(whitelist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote enum whitelist to {out_path}", file=sys.stderr)
    return 0


def cmd_parse_stdin(args: argparse.Namespace) -> int:
    raw_text = sys.stdin.read()
    records = segment_records(raw_text)
    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for idx, rec_text in enumerate(records):
        parsed = parse_record(rec_text)
        out = {
            "record_index": idx,
            "ingestion_timestamp": now_iso,
            "answers": parsed["answers"],
            "metadata": parsed["metadata"],
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    return 0


def cmd_clean_wide(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv)
    validation_report = Path(args.validation_report)
    clean_csv_to_wide(input_csv, out_csv, validation_report)
    print(f"Wrote cleaned wide CSV to {out_csv}\nWrote validation report to {validation_report}")
    return 0


def cmd_emit_long(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    emitter = LongEmitter(out_dir)
    emitter.emit_from_csv(input_csv, input_format=args.input_format)
    print(f"Wrote long tables under {out_dir}")
    return 0


def cmd_consistency(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv)
    report_json = Path(args.report_json)
    run_consistency_checks(input_csv, report_json, input_format=args.input_format)
    print(f"Wrote consistency report to {report_json}")
    return 0


def cmd_qa_summary(args: argparse.Namespace) -> int:
    wide_csv = Path(args.wide_csv)
    out_csv = Path(args.out_csv)
    create_qa_summary(wide_csv, out_csv, top_k=args.top_k)
    print(f"Wrote QA summary to {out_csv}")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    prompt = Path(args.prompt_path) if args.prompt_path else DEFAULT_PROMPT
    enum_out = Path(args.enum_out) if args.enum_out else DEFAULT_ENUM_OUT
    whitelist = build_enum_whitelist(prompt.read_text(encoding="utf-8"), source_path=str(prompt))
    enum_out.parent.mkdir(parents=True, exist_ok=True)
    enum_out.write_text(json.dumps(whitelist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote enum whitelist to {enum_out}")

    input_csv = Path(args.input_csv) if args.input_csv else DEFAULT_INPUT_CSV
    out_csv = Path(args.out_csv) if args.out_csv else DEFAULT_WIDE_CSV
    validation_json = Path(args.validation_json) if args.validation_json else DEFAULT_VALIDATION_JSON
    clean_csv_to_wide(input_csv, out_csv, validation_json)
    print(f"Wrote cleaned wide CSV to {out_csv}\nWrote validation report to {validation_json}")

    effective_wide_csv = out_csv
    if not getattr(args, "skip_fine_reconciliation", False):
        human_csv = (
            Path(args.human_fines_csv)
            if args.human_fines_csv
            else DEFAULT_HUMAN_FINES_CSV
        )
        reconciled_out_csv = (
            Path(args.reconciled_out_csv)
            if args.reconciled_out_csv
            else out_csv
        )
        comparison_csv = (
            Path(args.fine_comparison_csv)
            if args.fine_comparison_csv
            else None
        )
        summary_json = (
            Path(args.fine_summary_json)
            if args.fine_summary_json
            else None
        )
        tolerance = getattr(args, "fine_tolerance", 1.0)
        summary, overrides = run_fine_reconciliation(
            ai_csv=out_csv,
            human_csv=human_csv,
            out_csv=reconciled_out_csv,
            tolerance=tolerance,
            comparison_csv=comparison_csv,
            summary_json=summary_json,
        )
        effective_wide_csv = reconciled_out_csv
        print(
            "Applied human fine overrides to "
            f"{effective_wide_csv} (updated {overrides} records)"
        )
        if summary_json is None and comparison_csv is None:
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))

    long_dir = Path(args.long_dir) if args.long_dir else DEFAULT_LONG_DIR
    if args.long_input_csv:
        long_input_csv = Path(args.long_input_csv)
    elif args.long_input_format == "raw":
        long_input_csv = input_csv
    else:
        long_input_csv = effective_wide_csv
    long_input_format = args.long_input_format or (
        "wide" if long_input_csv == effective_wide_csv else "auto"
    )
    emitter = LongEmitter(long_dir)
    emitter.emit_from_csv(long_input_csv, input_format=long_input_format)
    print(f"Wrote long tables under {long_dir}")

    consistency_json = Path(args.consistency_json) if args.consistency_json else DEFAULT_CONSISTENCY_JSON
    if args.consistency_input_csv:
        consistency_input_csv = Path(args.consistency_input_csv)
    elif args.consistency_input_format == "raw":
        consistency_input_csv = input_csv
    else:
        consistency_input_csv = effective_wide_csv
    consistency_input_format = args.consistency_input_format or (
        "wide" if consistency_input_csv == effective_wide_csv else "auto"
    )
    run_consistency_checks(consistency_input_csv, consistency_json, input_format=consistency_input_format)
    print(f"Wrote consistency report to {consistency_json}")

    qa_csv = Path(args.qa_summary_csv) if args.qa_summary_csv else DEFAULT_QA_SUMMARY_CSV
    create_qa_summary(effective_wide_csv, qa_csv, top_k=5)
    print(f"Wrote QA summary to {qa_csv}")

    if getattr(args, "build_feature_matrix", False):
        _run_feature_matrix(effective_wide_csv, args)

    if getattr(args, "run_evenness", False):
        _run_evenness_pipeline(effective_wide_csv, args)

    return 0


def cmd_export_parquet(args: argparse.Namespace) -> int:
    wide_csv = Path(args.wide_csv)
    long_tables_dir = Path(args.long_tables_dir) if args.long_tables_dir else None
    out_dir = Path(args.out_dir)

    partition_cols = args.partition_cols.split(',') if args.partition_cols else None

    exporter = ParquetExporter(wide_csv, long_tables_dir)
    exporter.export(out_dir, partition_cols=partition_cols)
    print(f"Exported Parquet datasets to {out_dir}")
    return 0


def cmd_export_arrow(args: argparse.Namespace) -> int:
    wide_csv = Path(args.wide_csv)
    long_tables_dir = Path(args.long_tables_dir) if args.long_tables_dir else None
    out_dir = Path(args.out_dir)

    exporter = ArrowExporter(wide_csv, long_tables_dir)
    exporter.export(out_dir, compression=args.compression)
    print(f"Exported Arrow/Feather datasets to {out_dir}")
    return 0


def cmd_export_graph(args: argparse.Namespace) -> int:
    wide_csv = Path(args.wide_csv)
    long_tables_dir = Path(args.long_tables_dir) if args.long_tables_dir else None
    out_dir = Path(args.out_dir)

    exporter = GraphExporter(wide_csv, long_tables_dir)
    exporter.export(out_dir)
    print(f"Exported graph/network datasets to {out_dir}")
    return 0


def cmd_export_stats(args: argparse.Namespace) -> int:
    wide_csv = Path(args.wide_csv)
    long_tables_dir = Path(args.long_tables_dir) if args.long_tables_dir else None
    out_dir = Path(args.out_dir)

    formats = args.formats.split(',') if args.formats else ['r', 'stata', 'spss']

    exporter = StatisticalPackageExporter(wide_csv, long_tables_dir)
    exporter.export(out_dir, formats=formats)
    print(f"Exported statistical package datasets ({', '.join(formats)}) to {out_dir}")
    return 0


def cmd_export_ml(args: argparse.Namespace) -> int:
    wide_csv = Path(args.wide_csv)
    long_tables_dir = Path(args.long_tables_dir) if args.long_tables_dir else None
    out_dir = Path(args.out_dir)

    exporter = MLFeatureExporter(wide_csv, long_tables_dir)
    exporter.export(
        out_dir,
        embeddings_model=args.embeddings_model,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"Exported ML-ready datasets to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dpa-scripts", description="DPA decisions data utilities")
    sub = p.add_subparsers(dest="command", required=True)

    s1 = sub.add_parser("build-enum-whitelist", help="Extract ENUM/MULTI_SELECT specs from the questionnaire prompt")
    s1.add_argument("--prompt-path", required=True, help="Path to data-extraction-prompt-sent-to-ai.md")
    s1.add_argument("--out", required=True, help="Output JSON path for whitelist")
    s1.set_defaults(func=cmd_build_enum_whitelist)

    s2 = sub.add_parser("parse-stdin", help="Parse concatenated Answer blocks from STDIN and emit JSONL")
    s2.set_defaults(func=cmd_parse_stdin)

    s3 = sub.add_parser("clean-wide", help="Produce initial cleaned wide CSV and validation report")
    s3.add_argument("--input-csv", required=True)
    s3.add_argument("--out-csv", required=True)
    s3.add_argument("--validation-report", required=True)
    s3.set_defaults(func=cmd_clean_wide)

    s4 = sub.add_parser("emit-long", help="Emit long-form tables for key multi-selects and enums")
    s4.add_argument("--input-csv", required=True)
    s4.add_argument("--out-dir", required=True)
    s4.add_argument("--input-format", choices=["auto", "raw", "wide"], default="auto", help="Schema of --input-csv")
    s4.set_defaults(func=cmd_emit_long)

    s5 = sub.add_parser("consistency", help="Run cross-field consistency checks and write report JSON")
    s5.add_argument("--input-csv", required=True)
    s5.add_argument("--report-json", required=True)
    s5.add_argument("--input-format", choices=["auto", "raw", "wide"], default="auto", help="Schema of --input-csv")
    s5.set_defaults(func=cmd_consistency)

    s6 = sub.add_parser("qa-summary", help="Create QA summary over wide CSV known/unknown/status triplets")
    s6.add_argument("--wide-csv", required=True)
    s6.add_argument("--out-csv", required=True)
    s6.add_argument("--top-k", type=int, default=5)
    s6.set_defaults(func=cmd_qa_summary)

    s7 = sub.add_parser("run-all", help="Run full pipeline with defaults or provided paths")
    s7.add_argument("--prompt-path")
    s7.add_argument("--enum-out")
    s7.add_argument("--input-csv")
    s7.add_argument("--out-csv")
    s7.add_argument("--validation-json")
    s7.add_argument("--long-dir")
    s7.add_argument("--long-input-csv")
    s7.add_argument("--long-input-format", choices=["auto", "raw", "wide"])
    s7.add_argument("--consistency-json")
    s7.add_argument("--consistency-input-csv")
    s7.add_argument("--consistency-input-format", choices=["auto", "raw", "wide"])
    s7.add_argument("--qa-summary-csv")
    s7.add_argument(
        "--skip-fine-reconciliation",
        action="store_true",
        help="Disable human fine overrides during run-all",
    )
    s7.add_argument(
        "--human-fines-csv",
        help="Override path to human-annotated fines CSV",
    )
    s7.add_argument(
        "--reconciled-out-csv",
        help="Output path for human-reconciled wide CSV (defaults to --out-csv)",
    )
    s7.add_argument(
        "--fine-comparison-csv",
        help="Optional path for detailed AI vs human fine comparison",
    )
    s7.add_argument(
        "--fine-summary-json",
        help="Optional path for reconciliation summary JSON",
    )
    s7.add_argument(
        "--fine-tolerance",
        type=float,
        default=1.0,
        help="Treat EUR differences below this threshold as matches when summarising",
    )
    s7.add_argument(
        "--build-feature-matrix",
        action="store_true",
        help="Materialise the analysis feature matrix after cleaning",
    )
    s7.add_argument(
        "--feature-matrix-parquet",
        help="Override path for the feature matrix parquet output",
    )
    s7.add_argument(
        "--feature-matrix-metadata",
        help="Override path for the feature matrix metadata JSON",
    )
    s7.add_argument(
        "--run-evenness",
        action="store_true",
        help="Run evenness Phases 0-3 after cleaning",
    )
    s7.add_argument(
        "--evenness-light",
        action="store_true",
        help="Use the lightweight Phase 0 configuration (skips SHAP/SAGE/knockoffs)",
    )
    s7.add_argument(
        "--evenness-wide-csv",
        help="Path to write/read the wide CSV for evenness runs (default: outputs/cleaned_wide_latest.csv)",
    )
    s7.add_argument(
        "--evenness-discussed-only",
        action="store_true",
        help="Limit fact matrix to discussed-only responses when running evenness phases",
    )
    s7.add_argument(
        "--evenness-use-gpu",
        action="store_true",
        help="Enable GPU acceleration for Phase 0 tree models (if available)",
    )
    s7.add_argument(
        "--evenness-phase-three-outcome",
        default="fine_log1p",
        help="Outcome column to model during Phase 3",
    )
    s7.add_argument(
        "--evenness-phase-three-permutations",
        type=int,
        default=10,
        help="Permutation count for Phase 3 randomization inference",
    )
    s7.set_defaults(func=cmd_run_all)

    s7b = sub.add_parser(
        "reconcile-fines",
        help="Override AI fine amounts with human annotations and emit comparison artefacts",
    )
    s7b.add_argument("--wide-csv", default=DEFAULT_WIDE_CSV, help="Path to AI wide CSV")
    s7b.add_argument(
        "--human-csv",
        default="raw-data/all_gdpr_fines_raw_human_annotations.csv",
        help="Path to human annotated fines CSV",
    )
    s7b.add_argument(
        "--out-csv",
        default="outputs/cleaned_wide_with_human_overrides.csv",
        help="Destination for reconciled wide CSV",
    )
    s7b.add_argument(
        "--comparison-csv",
        help="Optional path to write detailed comparison table",
    )
    s7b.add_argument(
        "--summary-json",
        help="Optional path to write comparison summary JSON",
    )
    s7b.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Treat differences below this EUR threshold as matches",
    )
    s7b.set_defaults(func=cmd_reconcile_fines)

    # Export commands
    s8 = sub.add_parser("export-parquet", help="Export data to Parquet format with partitioning")
    s8.add_argument("--wide-csv", required=True, help="Path to cleaned wide CSV")
    s8.add_argument("--long-tables-dir", help="Path to long tables directory (optional)")
    s8.add_argument("--out-dir", required=True, help="Output directory for Parquet files")
    s8.add_argument("--partition-cols", help="Comma-separated partition columns (default: country_group,decision_year)")
    s8.set_defaults(func=cmd_export_parquet)

    s9 = sub.add_parser("export-arrow", help="Export data to Arrow/Feather format")
    s9.add_argument("--wide-csv", required=True, help="Path to cleaned wide CSV")
    s9.add_argument("--long-tables-dir", help="Path to long tables directory (optional)")
    s9.add_argument("--out-dir", required=True, help="Output directory for Arrow files")
    s9.add_argument("--compression", default="zstd", choices=["zstd", "lz4", "uncompressed"], help="Compression method")
    s9.set_defaults(func=cmd_export_arrow)

    s10 = sub.add_parser("export-graph", help="Export data as graph/network formats")
    s10.add_argument("--wide-csv", required=True, help="Path to cleaned wide CSV")
    s10.add_argument("--long-tables-dir", help="Path to long tables directory (optional)")
    s10.add_argument("--out-dir", required=True, help="Output directory for graph files")
    s10.set_defaults(func=cmd_export_graph)

    s11 = sub.add_parser("export-stats", help="Export data for statistical packages (R, Stata, SPSS)")
    s11.add_argument("--wide-csv", required=True, help="Path to cleaned wide CSV")
    s11.add_argument("--long-tables-dir", help="Path to long tables directory (optional)")
    s11.add_argument("--out-dir", required=True, help="Output directory for statistical files")
    s11.add_argument("--formats", default="r,stata,spss", help="Comma-separated formats to export (r,stata,spss)")
    s11.set_defaults(func=cmd_export_stats)

    s12 = sub.add_parser("export-ml", help="Export ML-ready features with embeddings and splits")
    s12.add_argument("--wide-csv", required=True, help="Path to cleaned wide CSV")
    s12.add_argument("--long-tables-dir", help="Path to long tables directory (optional)")
    s12.add_argument("--out-dir", required=True, help="Output directory for ML datasets")
    s12.add_argument("--embeddings-model", default="all-MiniLM-L6-v2", help="Sentence transformer model for embeddings")
    s12.add_argument("--test-size", type=float, default=0.2, help="Test set size (0.0-1.0)")
    s12.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    s12.set_defaults(func=cmd_export_ml)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
