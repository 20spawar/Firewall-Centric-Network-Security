# Fusion Module

## Purpose

The Fusion Module combines the outputs of the Insider Module
and Anomaly Module into a unified risk score.

## Inputs

### Insider Module

- user
- insider_score
- is_insider

### Anomaly Module

- user
- anomaly_score
- is_anomaly

## Fusion Method

The initial implementation uses equal-weight score fusion:

final_score =
    0.50 × insider_score +
    0.50 × anomaly_score

Scores are normalized to the range 0-100.

## Decision Levels

- NORMAL: 0-29.99
- LOW: 30-59.99
- MEDIUM: 60-79.99
- HIGH: 80-100

## Output

fusion/outputs/fusion_results.csv

Output columns:

- user
- insider_score
- anomaly_score
- is_insider
- is_anomaly
- final_score
- decision

## Future Integration

The Payload Module can be added later as a third input:

final_score =
    w1 × insider_score +
    w2 × anomaly_score +
    w3 × payload_score