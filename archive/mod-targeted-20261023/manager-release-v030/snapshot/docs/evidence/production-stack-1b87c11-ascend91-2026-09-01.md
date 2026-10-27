# Production Stack control-plane acceptance — 2026-09-01

This acceptance ran only in the isolated kind cluster
`vllmhust-ps-e2e-20260901` on `ascend91-host`. Its kubeconfig was isolated from
all host contexts. No model, GPU, or NPU was used.

## Fixed inputs

- official Production Stack source commit:
  `1b87c11a24c144f6b63a64dbae4fc8c875059731`;
- source archive SHA-256:
  `43127a52d883bccf2affea982680b181a2f6a427fe523e0731b2323f8edfc1ac`;
- kind `0.33.0`, Kubernetes node `1.34.11`, Helm `4.2.4`;
- official metrics-server `0.9.0` manifest SHA-256:
  `1cec29a5267809306a2c6ec74a3e449abbb705b4a8beed0c8a1963910f72c79b`;
- Router image built from the official Dockerfile and exact source commit:
  `vllmhust/production-stack-router:1b87c11-arm64`, image ID
  `sha256:012e8bcde16016665d056d62a36def9b71792b7606072b75d6a9b75812742a97`;
- controller binary built from the official Go source and exact commit, carried
  in a test-only `scratch` image because the upstream distroless registry was
  unreachable: image ID
  `sha256:3a6a61a13c16c0e7653423668e480e2e8db738f0911b5f9c1e9cdd482b2b474a`.

The test-only controller carrier is not represented as an official release
image. Its binary is the upstream controller; only the final image carrier and
Go proxy differed from the upstream Dockerfile.

## Results

1. The official controller reconciled `VLLMRouter/vllmhust-e2e-router` into an
   owned ServiceAccount, Role, RoleBinding, Service, and Deployment. Deployment
   and Service owner references pointed to the CR with `controller=true`.
2. Updating `spec.replicas` from one to two produced two ready Router pods;
   restoring it to one reconciled back to one. CR status was `Ready`.
3. The official Router forwarded `POST /v1/completions` to the external test
   backend and returned HTTP 200 with marker `forwarded-by-vllmhust-e2e`; the
   backend log independently recorded the POST.
4. With the backend fully absent, a UUID-unique request returned HTTP 500.
   Restoring the independently owned backend restored HTTP 200 without
   reinstalling the Router.
5. The real Metrics API reported baseline and load CPU. Under load, an HPA on a
   separately owned Router Deployment observed `87m`, `87%`,
   `ScalingActive=True`, and `AbleToScale=True`, then scaled from one to the
   configured maximum of three ready replicas.
6. A negative ownership test put an HPA on the `VLLMRouter`-owned Deployment.
   HPA desired two replicas, while the official controller immediately
   reconciled the Deployment back to the CR's one replica. This proves a
   two-writer conflict, not a supported autoscaling arrangement.
7. Deleting the `VLLMRouter` garbage-collected every owned resource. The direct
   HPA and its Deployment were deleted by the external test operator.

The controller's initial status briefly reported Ready before the Router's
fixed-delay liveness probe exposed an invalid single-label static backend URL.
Using the Kubernetes service FQDN fixed the upstream Router URL-validation
constraint. This demonstrates why Provider health requires distinct controller,
traffic, and autoscaler evidence rather than a single rollout boolean.

The Manager never applied, patched, scaled, or deleted these resources. Those
mutations were performed by the explicitly external acceptance operator.
