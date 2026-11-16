# Dataset

Place the original UCI `ai4i2020.csv` at
`ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv`, or pass `--data`.
The ZIP supplied with this project is already extracted at that path.
The experiment never generates substitute observations.

The committed CSV uses UTF-8 and LF line endings, enforced by `.gitattributes`.
Its exact SHA-256 is `59db4f1d9c34c58136d89e5a006ec190dcea19e9dbea74f6b3b0c6f22a44d183`.
Changing each LF to CRLF recovers the historical run's hash
`dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.
Only line endings changed; all cells and row ordering are retained.
See [migration evidence](eol_migration.json). The verifier checks exact bytes.

Source: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset), DOI 10.24432/C5HS5C, CC BY 4.0.
UCI describes this as synthetic data resembling industrial maintenance data.
