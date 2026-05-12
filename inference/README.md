# Inference Backend

This directory will contain the C# Web API using ONNX Runtime.

The workflow will be:
1. Python trains the model in `../training/`.
2. Python exports `model.onnx` and `quantizer_bins.json` to this directory.
3. The C# backend loads the ONNX model and serves it via a REST API to OpenBackTest.
