# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-05

### Added

- Initial public pre-release of ovsensors 0.2.0, with a C API and Python bindings
  for simulating sensors over application-owned ovstage scenes using USD RenderProducts.
- ovrtx provider and support for custom providers, with automatic selection by
  applied schema or explicit provider selection.
- Application-controlled stepping with sensor cadences and timestamped results.
- CPU and CUDA output access with NumPy and DLPack interoperability, subject to
  provider support.
- SDKs and Python wheels for Linux x86-64/ARM64 and Windows x86-64.
- Source examples for camera images, lidar point clouds, CPU/CUDA tensor access,
  and a custom CPU provider.

[0.2.0]: #020---2026-11-05
