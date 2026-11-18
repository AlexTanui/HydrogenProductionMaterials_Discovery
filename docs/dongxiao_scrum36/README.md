# SCRUM-36 review package

Start with the [16 November review](alex_review_2026_11_16/Review.md), which updates the historical [15 November analysis](SCRUM36_Report.md). The write-up is also integrated into
[the technical report](../technical_report.md) on the dongxiao branch.

Status: descriptive comparison and initial write-up ready for review.
Inferential statistics are pending aligned predictions and independent training replications.
This package does not claim the whole Jira ticket is complete.

- results/comparison.csv: five model variants, common metrics and provenance.
- results/phase3_differences.csv: absolute and relative differences.
- results/accuracy.png and results/coverage.png: actual saved results, not illustrations.
- results/audit.json: completed checks and remaining limitations.
- sources/: immutable copies of the 15 upstream run, subset and test records.
- source_manifest.json: pinned GitHub URLs and SHA256 checksums.
- analyze.py: reproducible audit and numerical/figure generation.

The 16 November review supersedes the earlier interpretation: use the matched NLL pair as the headline, and keep BLIP MSE as excluded historical context. The review also updates rescore status and distinguishes calibration-subset results. Source paths inside
JSON records are historical machine paths; this analysis does not require them to exist.

## Suggested Jira update
Completed a five-variant Phase 1/2/3 comparison on matching ethanol DFT splits and metric
definitions. Added reproducible tables, figures, source provenance and a technical-report
draft, explicitly documenting where Phase 3 underperforms BLIP. Statistical significance
is not claimed: paired prediction exports and multi-seed replications are still needed.
Please review the descriptive findings and confirm the remaining statistical-analysis scope.
