import json
import unittest
from pathlib import Path

from scripts.parser.ingest import segment_records, parse_record
from scripts.clean.typing_status import parse_date_field, normalize_country
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


class TestTypingAndCountry(unittest.TestCase):
    def test_date_and_country_parsing(self):
        dpr = parse_date_field("2023-10-26")
        self.assertIsNotNone(dpr.value)
        self.assertEqual(dpr.status, "DISCUSSED")
        code, status = normalize_country("ISO_3166-1_ALPHA-2: FR")
        self.assertEqual(code, "FR")
        self.assertEqual(status, "DISCUSSED")


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
