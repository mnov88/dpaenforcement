import csv
import json
import shutil
import unittest
from pathlib import Path

from scripts.parser.ingest import segment_records, parse_record
from scripts.clean.typing_status import (
    parse_date_field,
    normalize_country,
    parse_number,
    derive_multiselect_status,
    detect_exclusivity_conflict,
)
from scripts.clean.isic_map import IsicIndex
from scripts.clean.consistency import run_consistency_checks
from scripts.clean.wide_output import clean_csv_to_wide
from scripts.clean.long_tables import LongEmitter


class TestParserAndSegmentation(unittest.TestCase):
    def test_segment_and_parse_record_basic(self):
        text = (
            "Answer 1: ISO_3166-1_ALPHA-2: DE\n"
            "Answer 2: BfDI\n"
            "Answer 3: 2022-01-31\n"
            "Answer 4: NO\n"
            "Answer 5: NOT_APPLICABLE\n"
            + "\n".join([f"Answer {i}: X" for i in range(6, 69)])
        )
        segs = segment_records(text)
        self.assertEqual(len(segs), 1)
        parsed = parse_record(segs[0])
        self.assertTrue(parsed["metadata"]["completeness"])
        self.assertTrue(parsed["answers"]["Q1"].startswith("ISO_3166-1_ALPHA-2"))
        self.assertIn("parser_version", parsed["metadata"])
        self.assertEqual(parsed["metadata"]["question_count"], 68)


class TestTypingAndCountry(unittest.TestCase):
    def test_date_and_country_parsing(self):
        dpr = parse_date_field("2023-10-26")
        self.assertIsNotNone(dpr.value)
        self.assertEqual(dpr.status, "DISCUSSED")
        self.assertEqual(dpr.raw, "2023-10-26")
        err = parse_date_field("2023-13-40")
        self.assertEqual(err.status, "PARSE_ERROR")
        self.assertIsNone(err.value)
        code, status = normalize_country("ISO_3166-1_ALPHA-2: FR")
        self.assertEqual(code, "FR")
        self.assertEqual(status, "DISCUSSED")
        num = parse_number("1,234.50")
        self.assertTrue(num.valid)
        self.assertAlmostEqual(num.value, 1234.50)
        euro = parse_number("EUR 200,000")
        self.assertTrue(euro.valid)
        self.assertEqual(euro.value, 200000)
        neg = parse_number("-5")
        self.assertFalse(neg.valid)
        self.assertEqual(neg.status, "NEGATIVE_VALUE")
        missing = parse_number("")
        self.assertEqual(missing.status, "NOT_MENTIONED")
        null = parse_number("null")
        self.assertEqual(null.status, "NOT_MENTIONED")
        no_token = parse_number("NO")
        self.assertEqual(no_token.status, "NOT_MENTIONED")
        schema = parse_number("TYPE:NUMBER 150000")
        self.assertTrue(schema.valid)
        self.assertEqual(schema.value, 150000)
        schema_only = parse_number("TYPE:NUMBER")
        self.assertEqual(schema_only.status, "NOT_MENTIONED")

    def test_multiselect_exclusivity_conflicts(self):
        no_conflict = ["NOT_APPLICABLE"]
        self.assertEqual(detect_exclusivity_conflict(no_conflict), 0)
        self.assertEqual(derive_multiselect_status("Q30", no_conflict), "NOT_APPLICABLE")

        with_substantive = ["NOT_APPLICABLE", "SECURITY"]
        self.assertEqual(detect_exclusivity_conflict(with_substantive), 1)
        self.assertEqual(
            derive_multiselect_status("Q30", with_substantive),
            "MIXED_CONTRADICTORY",
        )

        mixed_markers = ["NOT_APPLICABLE", "NONE_MENTIONED"]
        self.assertEqual(detect_exclusivity_conflict(mixed_markers), 1)
        self.assertEqual(
            derive_multiselect_status("Q30", mixed_markers),
            "MIXED_CONTRADICTORY",
        )

        none_violated_only = ["NONE_VIOLATED"]
        self.assertEqual(detect_exclusivity_conflict(none_violated_only), 0)
        self.assertEqual(
            derive_multiselect_status("Q57", none_violated_only),
            "NONE_VIOLATED",
        )

        none_violated_conflict = ["NONE_VIOLATED", "ACCESS_RIGHT"]
        self.assertEqual(detect_exclusivity_conflict(none_violated_conflict), 1)
        self.assertEqual(
            derive_multiselect_status("Q57", none_violated_conflict),
            "MIXED_CONTRADICTORY",
        )


class TestISICAndConsistency(unittest.TestCase):
    def test_isic_index_load(self):
        tmpdir = Path(".tmp_test_isic")
        tmpdir.mkdir(exist_ok=True)
        p = tmpdir / "isic.csv"
        p.write_text('"Code","Description"\n"J","Information and communication"\n"62","Computer programming, consultancy"\n"6209","Other information technology and computer service activities"\n', encoding="utf-8")
        idx = IsicIndex.load_from_file(p)
        entry, ok = idx.lookup("6209")
        self.assertTrue(ok)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.section, "J")

    def test_consistency_checks(self):
        tmpdir = Path(".tmp_test_consistency")
        tmpdir.mkdir(exist_ok=True)
        csv_p = tmpdir / "mini.csv"
        csv_p.write_text(
            "ID,response\n"
            "X1,\"Answer 37: 0\nAnswer 53: ADMINISTRATIVE_FINE\"\n",
            encoding="utf-8",
        )
        out_p = tmpdir / "report.json"
        run_consistency_checks(csv_p, out_p)
        data = json.loads(out_p.read_text(encoding="utf-8"))
        self.assertTrue(data)
        self.assertTrue(data[0]["flags"][0].startswith("admin_fine"))


class TestSchemaEchoNormalisation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(".tmp_schema_echo_tests")
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir)
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @staticmethod
    def _response(overrides: dict[str, str]) -> str:
        items = []
        base_overrides = {"Q1": "ISO_3166-1_ALPHA-2: FR"}
        base_overrides.update(overrides)
        for key, value in base_overrides.items():
            if not key.startswith("Q"):
                continue
            try:
                qnum = int(key[1:])
            except ValueError:
                continue
            items.append((qnum, value))
        items.sort()
        return "\n".join(f"Answer {qnum}: {value}" for qnum, value in items)

    def test_wide_output_strips_schema_prefixes(self):
        raw_csv = self.tmpdir / "raw.csv"
        response = self._response(
            {
                "Q1": "ISO_3166-1_ALPHA-2: FR",
                "Q2": "ENUM:CNIL",
                "Q3": "2024-01-01",
                "Q10": "ENUM:SME, ENUM:PUBLIC_SECTOR_BODY",
                "Q12": "FORMAT:6209",
                "Q15": "ENUM:COMPLAINT",
                "Q21": "ENUM:SECURITY_INCIDENT, ENUM:OTHER",
                "Q25": "ENUM:ARTICLE_9_SPECIAL_CATEGORY, ENUM:NEITHER",
                "Q28": "ENUM:STAFF_TRAINING, ENUM:LEGAL_ADVICE",
                "Q30": "ENUM:ACCOUNTABILITY, ENUM:SECURITY",
            }
        )
        raw_csv.write_text(
            "ID,response\n"
            + f"CASE-1,\"{response.replace('"', '""')}\"\n",
            encoding="utf-8",
        )
        out_csv = self.tmpdir / "wide.csv"
        report = self.tmpdir / "report.json"
        clean_csv_to_wide(raw_csv, out_csv, report)

        rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["raw_q15"], "COMPLAINT")
        self.assertEqual(row["raw_q21"], "SECURITY_INCIDENT, OTHER")
        self.assertEqual(row["raw_q10"], "SME, PUBLIC_SECTOR_BODY")
        self.assertEqual(row["raw_q25"], "ARTICLE_9_SPECIAL_CATEGORY, NEITHER")
        self.assertEqual(row["raw_q28"], "STAFF_TRAINING, LEGAL_ADVICE")
        self.assertEqual(row["raw_q30"], "ACCOUNTABILITY, SECURITY")
        flagged = set((row.get("schema_echo_fields") or "").split(";"))
        self.assertIn("Q2", flagged)
        self.assertIn("Q12", flagged)
        self.assertIn("Q15", flagged)
        self.assertIn("Q10", flagged)
        self.assertIn("Q21", flagged)
        self.assertIn("Q25", flagged)
        self.assertIn("Q28", flagged)
        self.assertIn("Q30", flagged)

    def test_long_tables_emit_clean_tokens(self):
        raw_csv = self.tmpdir / "raw.csv"
        response = self._response(
            {
                "Q1": "ISO_3166-1_ALPHA-2: FR",
                "Q10": "ENUM:SME, ENUM:PUBLIC_SECTOR_BODY",
                "Q21": "ENUM:SECURITY_INCIDENT, ENUM:OTHER",
                "Q25": "ENUM:ARTICLE_9_SPECIAL_CATEGORY",
                "Q28": "ENUM:STAFF_TRAINING, ENUM:LEGAL_ADVICE",
                "Q30": "ENUM:ACCOUNTABILITY, ENUM:SECURITY",
                "Q33": "ENUM:CONSENT",
                "Q35": "FORMAT:APPROVED",
            }
        )
        raw_csv.write_text(
            "ID,response\n"
            + f"CASE-1,\"{response.replace('"', '""')}\"\n",
            encoding="utf-8",
        )

        out_dir_raw = self.tmpdir / "long_raw"
        emitter_raw = LongEmitter(out_dir_raw)
        emitter_raw.emit_from_csv(raw_csv, input_format="raw")
        class_rows = list(csv.DictReader((out_dir_raw / "defendant_classifications.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "SME" for r in class_rows))
        self.assertTrue(any(r["option"] == "PUBLIC_SECTOR_BODY" for r in class_rows))
        breach_rows = list(csv.DictReader((out_dir_raw / "breach_types.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "SECURITY_INCIDENT" for r in breach_rows))
        self.assertTrue(any(r["option"] == "OTHER" for r in breach_rows))
        special_rows = list(csv.DictReader((out_dir_raw / "special_data_categories.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "ARTICLE_9_SPECIAL_CATEGORY" for r in special_rows))
        mitig_rows = list(csv.DictReader((out_dir_raw / "mitigating_actions.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "STAFF_TRAINING" for r in mitig_rows))
        self.assertTrue(any(r["option"] == "LEGAL_ADVICE" for r in mitig_rows))

        wide_csv = self.tmpdir / "wide.csv"
        report = self.tmpdir / "report.json"
        clean_csv_to_wide(raw_csv, wide_csv, report)
        out_dir_wide = self.tmpdir / "long_wide"
        emitter_wide = LongEmitter(out_dir_wide)
        emitter_wide.emit_from_csv(wide_csv, input_format="wide")
        class_rows_wide = list(csv.DictReader((out_dir_wide / "defendant_classifications.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "SME" for r in class_rows_wide))
        special_rows_wide = list(csv.DictReader((out_dir_wide / "special_data_categories.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "ARTICLE_9_SPECIAL_CATEGORY" for r in special_rows_wide))
        mitig_rows_wide = list(csv.DictReader((out_dir_wide / "mitigating_actions.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "STAFF_TRAINING" for r in mitig_rows_wide))
        rights_rows = list(csv.DictReader((out_dir_wide / "article_5_discussed.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "ACCOUNTABILITY" for r in rights_rows))
        li_rows = list(csv.DictReader((out_dir_wide / "li_test_outcome.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["option"] == "APPROVED" for r in li_rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
