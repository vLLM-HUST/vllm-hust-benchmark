# Single dataset matrix correction

Website PR443 restores the existing matrix format. Controller is a column alongside the other MODs.
Its FrontierScience values occupy the original Olympiad and Research rows, and SWE-smith is one
additional dataset row explicitly labeled output-throughput gain. No separate profile or replay
section remains. Configurations, matching baseline receipts and metric limitations stay in the
standard dataset details. Historical overview Native values and all original measurements remain
unchanged; the dated new matching Native is still available in the FrontierScience detail. The
previous deep link continues to work.

All4exact-head CI checks passed, including desktop/mobile Chinese/English one-table/six-row
verification. The actual published page was inspected in the in-app browser and DOM check
confirms1table,6rows, Controller column present,0separate sections. Pages built the exact merge.
Served JS/CSS/snapshot/source report bytes match the validated checkout. HTML matches after removing
exactly one injected Cloudflare analytics script line; receipt records this normalization. The
included screenshot is from the live site. No experiment or statistics were changed to fit the
layout. Other datasets continue running independently.
