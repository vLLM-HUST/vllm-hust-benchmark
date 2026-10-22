# Preserved rejected variants

1. Vector int8-to-FP16-to-FP32 dequantization compiled and ran, but reconstructed signed int8 data incorrectly on this toolchain path: relative RMS 1.66922 and maximum dense error 2.92924. It was replaced by exact scalar int8 extraction plus vector arithmetic.
2. `WholeReduceSum` smoothing prototype compiled and ran, but the chosen repeat/stride formulation produced maximum smoothing error 18.265289 and five quantized-code mismatches. It was reverted rather than weakening the exactness gate.
3. The accepted M2 component prototype remains a negative performance result: exact codes and bounded reconstruction, but 18.3705 ms before rank fitting versus at most 40.789 us of copy savings.
