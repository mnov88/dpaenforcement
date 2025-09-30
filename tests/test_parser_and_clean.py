import csv
import json
import math
import shutil
import unittest
from pathlib import Path

import pandas as pd

from scripts.parser.ingest import segment_records, parse_record
from scripts.clean.typing_status import (
    parse_date_field,
    normalize_country,
    parse_number,
    parse_enum_field,
    derive_multiselect_status,
    detect_exclusivity_conflict,
)
from scripts.clean.isic_map import IsicIndex
from scripts.clean.consistency import run_consistency_checks
from scripts.clean.wide_output import clean_csv_to_wide
from scripts.clean.long_tables import LongEmitter
from scripts.analysis.build_feature_matrix import build_feature_matrix


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
        schema_compact = parse_number("TYPE:150000")
        self.assertTrue(schema_compact.valid)
        self.assertEqual(schema_compact.value, 150000)
        schema_number_zero = parse_number("TYPE:NUMBER 0")
        self.assertTrue(schema_number_zero.valid)
        self.assertEqual(schema_number_zero.value, 0)
        schema_only = parse_number("TYPE:NUMBER")
        self.assertEqual(schema_only.status, "NOT_MENTIONED")

    def test_enum_field_parsing(self):
        allowed = [
            "YES_REQUIRED",
            "NO_NOT_REQUIRED",
            "DEFENDANT_DISPUTED_REQUIREMENT",
            "NOT_APPLICABLE",
            "NOT_MENTIONED",
            "UNCLEAR",
        ]
        res = parse_enum_field("YES_REQUIRED", allowed)
        self.assertEqual(res.value, "YES_REQUIRED")
        self.assertEqual(res.status, "DISCUSSED")
        menu = (
            "YES_REQUIRED, NO_NOT_REQUIRED, DEFENDANT_DISPUTED_REQUIREMENT, "
            "NOT_APPLICABLE, NOT_MENTIONED, UNCLEAR NOT_APPLICABLE"
        )
        res_menu = parse_enum_field(menu, allowed)
        self.assertEqual(res_menu.value, "NOT_APPLICABLE")
        self.assertEqual(res_menu.status, "NOT_APPLICABLE")
        self.assertEqual(res_menu.note, "menu_echo")
        trailing = parse_enum_field(
            "Was the defendant required to notify the DPA under Article 33?: No",
            allowed,
            {"NO": "NO_NOT_REQUIRED"},
        )
        self.assertEqual(trailing.value, "NO_NOT_REQUIRED")
        self.assertEqual(trailing.status, "DISCUSSED")
        conflict = parse_enum_field("NOT_MENTIONED, UNCLEAR", allowed)
        self.assertEqual(conflict.status, "MIXED_CONTRADICTORY")
        empty = parse_enum_field("", allowed)
        self.assertEqual(empty.status, "NOT_MENTIONED")
        notify_allowed = [
            "YES_NOTIFIED",
            "NO_NOT_NOTIFIED",
            "PARTIALLY_NOTIFIED",
            "NOT_APPLICABLE",
            "NOT_MENTIONED",
            "UNCLEAR",
        ]
        notify = parse_enum_field("NOTIFIED", notify_allowed, {"NOTIFIED": "YES_NOTIFIED"})
        self.assertEqual(notify.value, "YES_NOTIFIED")
        self.assertEqual(notify.status, "DISCUSSED")

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

        not_discussed_only = ["NOT_DISCUSSED"]
        self.assertEqual(detect_exclusivity_conflict(not_discussed_only), 0)
        self.assertEqual(
            derive_multiselect_status("Q43", not_discussed_only),
            "NOT_DISCUSSED",
        )

        not_discussed_conflict = ["NOT_DISCUSSED", "YES_MATERIAL_HARM"]
        self.assertEqual(detect_exclusivity_conflict(not_discussed_conflict), 1)
        self.assertEqual(
            derive_multiselect_status("Q43", not_discussed_conflict),
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
        self.assertEqual(entry.section_description, "Information and communication")
        self.assertEqual(entry.division, "62")
        self.assertEqual(entry.division_description, "Computer programming, consultancy")
        self.assertEqual(entry.group, None)
        self.assertEqual(entry.group_description, None)
        division_entry, ok_div = idx.lookup("62")
        self.assertTrue(ok_div)
        self.assertEqual(division_entry.code, "62")
        self.assertEqual(division_entry.division, "62")
        self.assertEqual(division_entry.group, None)
        parsed, invalid = idx.parse_codes("J;62;6209;9999")
        self.assertEqual([p.code for p in parsed], ["J", "62", "6209"])
        self.assertEqual(invalid, ["9999"])
        combo_parsed, combo_invalid = idx.parse_codes("J62")
        self.assertEqual([p.code for p in combo_parsed], ["62"])
        self.assertFalse(combo_invalid)

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
        self.assertEqual(row["isic_code"], "6209")
        self.assertEqual(row["isic_section"], "J")
        self.assertEqual(row["isic_division_code"], "62")
        self.assertEqual(row["isic_group_code"], "620")
        self.assertEqual(row["isic_codes_all"], "6209")
        self.assertEqual(row["isic_multi_sector"], "0")
        self.assertEqual(row["isic_unparsed_tokens"], "")
        self.assertTrue(row["isic_reference_version"])

    def test_long_tables_emit_clean_tokens(self):
        raw_csv = self.tmpdir / "raw.csv"
        response = self._response(
            {
                "Q1": "ISO_3166-1_ALPHA-2: FR",
                "Q10": "ENUM:SME, ENUM:PUBLIC_SECTOR_BODY",
                "Q12": "J, 62, 63",
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
        isic_rows_raw = list(csv.DictReader((out_dir_raw / "isic_assignments.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["isic_code"] == "62" for r in isic_rows_raw))
        self.assertTrue(any(r["isic_code"] == "63" for r in isic_rows_raw))
        self.assertTrue(any(r["parse_status"] == "MATCHED" for r in isic_rows_raw))

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
        isic_rows_wide = list(csv.DictReader((out_dir_wide / "isic_assignments.csv").open(encoding="utf-8")))
        self.assertTrue(any(r["isic_code"] == "62" for r in isic_rows_wide))
        self.assertTrue(any(r["isic_code"] == "63" for r in isic_rows_wide))
        self.assertTrue(any(r["is_primary"] == "1" for r in isic_rows_wide))

    def test_feature_matrix_derives_isic_indicators(self):
        records: list[dict[str, object]] = []
        for idx in range(5):
            records.append(
                {
                    "decision_id": f"J-{idx}",
                    "country_code": "FR",
                    "country_group": "EU",
                    "dpa_name_canonical": "CNIL",
                    "decision_year": 2023,
                    "decision_quarter": "Q1",
                    "breach_case": 1,
                    "severity_measures_present": 1,
                    "remedy_only_case": 0,
                    "fine_eur": 1000.0,
                    "fine_log1p": math.log1p(1000.0),
                    "fine_status": "DISCUSSED",
                    "fine_to_turnover_ratio": 0.1,
                    "turnover_eur": 10000.0,
                    "turnover_log1p": math.log1p(10000.0),
                    "turnover_status": "DISCUSSED",
                    "isic_section": "J",
                    "isic_section_desc": "Information and communication",
                    "isic_code": "6209",
                    "isic_desc": "Other information technology and computer service activities",
                    "isic_division_code": "62",
                    "isic_division_desc": "Computer programming, consultancy and related activities",
                    "isic_group_code": "620",
                    "isic_group_desc": "Computer programming, consultancy and related activities",
                    "isic_multi_sector": 0,
                    "isic_codes_all": "6209",
                    "n_principles_discussed": 1,
                    "n_principles_violated": 1,
                    "n_corrective_measures": 1,
                }
            )
        for idx in range(5):
            records.append(
                {
                    "decision_id": f"G-{idx}",
                    "country_code": "DE",
                    "country_group": "EU",
                    "dpa_name_canonical": "BfDI",
                    "decision_year": 2022,
                    "decision_quarter": "Q2",
                    "breach_case": 1,
                    "severity_measures_present": 0,
                    "remedy_only_case": 0,
                    "fine_eur": 500.0,
                    "fine_log1p": math.log1p(500.0),
                    "fine_status": "DISCUSSED",
                    "fine_to_turnover_ratio": 0.05,
                    "turnover_eur": 8000.0,
                    "turnover_log1p": math.log1p(8000.0),
                    "turnover_status": "DISCUSSED",
                    "isic_section": "G",
                    "isic_section_desc": "Wholesale and retail trade; repair of motor vehicles and motorcycles",
                    "isic_code": "4711",
                    "isic_desc": "Retail sale in non-specialized stores with food, beverages or tobacco predominating",
                    "isic_division_code": "47",
                    "isic_division_desc": "Retail trade, except of motor vehicles and motorcycles",
                    "isic_group_code": "471",
                    "isic_group_desc": "Retail sale in non-specialized stores",
                    "isic_multi_sector": 0,
                    "isic_codes_all": "4711",
                    "n_principles_discussed": 0,
                    "n_principles_violated": 0,
                    "n_corrective_measures": 0,
                }
            )

        df = pd.DataFrame(records)
        wide_path = self.tmpdir / "wide_isic.csv"
        df.to_csv(wide_path, index=False)
        artifacts = build_feature_matrix(wide_path)
        matrix = artifacts.dataframe
        self.assertIn("ISIC_SECTION_J", matrix.columns)
        self.assertIn("ISIC_SECTION_G", matrix.columns)
        self.assertIn("ISIC_SECTION_MISSING", matrix.columns)
        self.assertIn("ISIC_DIVISION_62", matrix.columns)
        self.assertIn("ISIC_DIVISION_47", matrix.columns)
        j_mask = matrix["decision_id"].str.startswith("J-")
        g_mask = matrix["decision_id"].str.startswith("G-")
        self.assertTrue(bool((matrix.loc[j_mask, "ISIC_SECTION_J"] == 1).all()))
        self.assertTrue(bool((matrix.loc[g_mask, "ISIC_SECTION_G"] == 1).all()))
        self.assertTrue(bool((matrix.loc[:, "isic_multi_sector"] == 0).all()))
        sections_group = artifacts.column_groups.get("isic_sections", [])
        self.assertIn("ISIC_SECTION_J", sections_group)
        divisions_group = artifacts.column_groups.get("isic_divisions", [])
        self.assertIn("ISIC_DIVISION_62", divisions_group)


if __name__ == "__main__":
    unittest.main(verbosity=2)
