import json
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
