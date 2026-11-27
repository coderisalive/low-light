# Tests

This directory contains the automated validation checks for the project.

## Included tests

- `test_multi_dataset.py`: verifies multi-dataset discovery and loader behavior
- `test_paired_rgb_restoration.py`: validates paired RGB restoration functionality

## Purpose

The tests confirm that:

- dataset path resolution works for common dataset layouts
- paired restoration code integrates correctly with the training pipeline
- the project remains stable across targeted training and evaluation scenarios

## Running tests

Use pytest from the project root:

```bash
pytest tests
```
