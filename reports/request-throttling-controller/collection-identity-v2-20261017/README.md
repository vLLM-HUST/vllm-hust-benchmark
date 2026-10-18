# Collection identity correction after r17

Measurement sources remain frozen. The first r17 collection incorrectly inherited
metadata.remote_root=r16, producing a mislabeled private archive. It was detected before any public
backup, marked INVALID, and retained locally. Correct r17 data was recollected with explicit
--remote-root after checking run/job/root/report output/seed identity. Correct private archive
SHA96760bd309f7f7f294ef7382b2073d5bba369f9c04519fb4884859735c298ad4.

This collection/export tool revision rejects stale root metadata and cross-run report/job
association. Three focused regression tests pass. The frozen offline-burst-protocol snapshot retains
the original collection script as historical source; use this revised version for subsequent
collection. No model, workload, scheduling algorithm or measurement code was changed. Planned r18
collection metadata was corrected before submission.
