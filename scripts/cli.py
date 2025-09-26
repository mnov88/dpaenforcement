import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from scripts.parser.ingest import segment_records, parse_record
from scripts.parser.enums import build_enum_whitelist
from scripts.clean.wide_output import clean_csv_to_wide
from scripts.clean.long_tables import LongEmitter
from scripts.clean.consistency import run_consistency_checks
from scripts.clean.qa_summary import create_qa_summary


DEFAULT_PROMPT = Path("analyzed-decisions/data-extraction-prompt-sent-to-ai.md")
DEFAULT_ENUM_OUT = Path("resources/enum_whitelist.json")
DEFAULT_INPUT_CSV = Path("analyzed-decisions/master-analyzed-data-unclean.csv")
DEFAULT_WIDE_CSV = Path("outputs/cleaned_wide.csv")
DEFAULT_VALIDATION_JSON = Path("outputs/validation_report.json")
DEFAULT_LONG_DIR = Path("outputs/long_tables")
DEFAULT_CONSISTENCY_JSON = Path("outputs/consistency_report.json")
DEFAULT_QA_SUMMARY_CSV = Path("outputs/qa_summary.csv")


def cmd_build_enum_whitelist(args: argparse.Namespace) -> int:
    prompt_path = Path(args.prompt_path)
    out_path = Path(args.out)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    whitelist = build_enum_whitelist(prompt_text)
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
    emitter.emit_from_csv(input_csv)
    print(f"Wrote long tables under {out_dir}")
    return 0


def cmd_consistency(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv)
    report_json = Path(args.report_json)
    run_consistency_checks(input_csv, report_json)
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
    whitelist = build_enum_whitelist(prompt.read_text(encoding="utf-8"))
    enum_out.parent.mkdir(parents=True, exist_ok=True)
    enum_out.write_text(json.dumps(whitelist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote enum whitelist to {enum_out}")

    input_csv = Path(args.input_csv) if args.input_csv else DEFAULT_INPUT_CSV
    out_csv = Path(args.out_csv) if args.out_csv else DEFAULT_WIDE_CSV
    validation_json = Path(args.validation_json) if args.validation_json else DEFAULT_VALIDATION_JSON
    clean_csv_to_wide(input_csv, out_csv, validation_json)
    print(f"Wrote cleaned wide CSV to {out_csv}\nWrote validation report to {validation_json}")

    long_dir = Path(args.long_dir) if args.long_dir else DEFAULT_LONG_DIR
    emitter = LongEmitter(long_dir)
    emitter.emit_from_csv(input_csv)
    print(f"Wrote long tables under {long_dir}")

    consistency_json = Path(args.consistency_json) if args.consistency_json else DEFAULT_CONSISTENCY_JSON
    run_consistency_checks(input_csv, consistency_json)
    print(f"Wrote consistency report to {consistency_json}")

    qa_csv = Path(args.qa_summary_csv) if args.qa_summary_csv else DEFAULT_QA_SUMMARY_CSV
    create_qa_summary(out_csv, qa_csv, top_k=5)
    print(f"Wrote QA summary to {qa_csv}")

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
    s4.set_defaults(func=cmd_emit_long)

    s5 = sub.add_parser("consistency", help="Run cross-field consistency checks and write report JSON")
    s5.add_argument("--input-csv", required=True)
    s5.add_argument("--report-json", required=True)
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
    s7.add_argument("--consistency-json")
    s7.add_argument("--qa-summary-csv")
    s7.set_defaults(func=cmd_run_all)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
