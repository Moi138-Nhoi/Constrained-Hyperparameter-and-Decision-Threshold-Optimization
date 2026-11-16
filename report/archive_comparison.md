# Audit of the ZIP supplied for this completion

Source ZIP SHA-256: `e30d970d83251aa3098c36f32b9520d9cd89ef4d068e513198d981c49c6d1400`.
Compared with remote commit `239b4db` before edits. This audit refers to the ZIP supplied in this task; it replaces an older comparison of a different ZIP.

| Item | Supplied ZIP | Final submission |
|---|---|---|
| Experiment | 5 baselines; 40 Random Search + 40 TPE | Preserved, not rerun |
| Code/log comparison with remote | Shared text equal after newline normalization | Modeling/selection logic retained; writers explicitly use LF |
| Model and figures | Byte-identical to remote | Preserved |
| Dataset | LF; historical manifest still hashes CRLF | LF enforced by `.gitattributes`; actual byte hashes recorded |
| Report | Root `Bao_Caopdf.pdf`, 8 pages; missing linked `report/` artifacts | One 12-page editable DOCX/PDF pair under `report/` |
| Defense aid | Missing from ZIP | One-page DOCX/PDF |
| Identity | Cover identifies Hu?nh Tr? Minh, 24748020102035, Nguy?n Th? Di?m Anh | Copied with provenance; no alternative concrete identity found |
| Git | No `.git` in supplied ZIP or old nested working directory | New clone; original remote history retained |

The dataset's LF bytes have SHA-256 `59db4f1d9c34c58136d89e5a006ec190dcea19e9dbea74f6b3b0c6f22a44d183`.
Replacing each LF with CRLF exactly recovers the original experiment hash `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.
This proves the mismatch is packaging, not changed cells. Both full and pilot manifests retain the old hash as provenance and reference actual LF bytes for verification. Original timestamps, predictions and timing observations are unchanged.

The previous 10-page GitHub report and root 8-page report are not competing final versions. The final report uses the original cover motif/logo and four extracted real-run images; editable text fixes the cover positioning and supplies all seven required sections. The old PDF is retained in Git history, not as a second current report.

Pilot test exposure remains disclosed. Six initial commits on 9 November 2026 do not demonstrate five weeks of development; no history or dates are rewritten. The final ZIP is created with `git archive` from the submitted commit, with every entry checked against that commit's blob and an external SHA-256 inventory.
