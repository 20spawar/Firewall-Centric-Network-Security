# Anomaly Detection Module

## Purpose

The Anomaly Detection Module identifies unusual network behaviour using an unsupervised Isolation Forest model.

## Pipeline
pip 
Raw network data
→ preprocessing
→ feature extraction
→ Isolation Forest
→ anomaly score
→ anomaly classification

## Input

The module accepts network-flow data in CSV format.

## Output

The module produces:

- anomaly score
- anomaly classification
- relevant flow identifiers

## Model

Isolation Forest is used for unsupervised anomaly detection.

## Integration

The module is designed to operate independently from the Payload Analysis and Insider Threat modules.

Its output can later be consumed by the Fusion module.