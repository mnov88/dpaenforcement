# Fine Amount Comparison Snapshot

This note summarises the discrepancy analysis between the AI-generated wide
dataset (`outputs/cleaned_wide.csv`) and the human annotated fines from
`raw-data/all_gdpr_fines_raw_human_annotations.csv`.  Figures were produced with
`python -m scripts.analysis.fine_reconciliation compare --tolerance 1.0`.

## Aggregate status counts

| Status          | Records |
|-----------------|---------|
| Matches         | 697     |
| Conflicts       | 417     |
| AI missing      | 313     |
| Human missing   | 637     |
| Missing in both | 629     |
| **Total**       | 2,693   |

*Tolerance*: entries are treated as matches when the absolute difference is at
most 1 EUR.

## Top five conflicts by absolute Euro difference

| Decision ID  | AI fine (EUR) | Human fine (EUR) | Δ (EUR) |
|--------------|---------------|------------------|---------|
| Ireland_40   | 0             | 210,000,000      | 210,000,000 |
| Ireland_41   | 0             | 180,000,000      | 180,000,000 |
| Norway_34    | 100,000,000   | 8,900,000        | 91,100,000 |
| Hungary_56   | 80,000,000    | 200,000          | 79,800,000 |
| Hungary_18   | 80,000,000    | 208,000          | 79,792,000 |

## Currency coverage in human annotations

Detected currencies (after normalisation): BGN, CZK, DKK, EUR, GBP, HRK, HUF,
ISK, NOK, PLN, RON, SEK.

These figures highlight the large backlog of fines that require updates from
the human annotations—both for missing AI values and clear conflicts.
