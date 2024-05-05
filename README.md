# SMoTeF: Smurf-based Money Laundering Detection Framework

## Overview

SMoTeF (Smurf Money Laundering Detection using Temporal Order and Flow Analysis) is a novel framework designed to enhance the detection of smurfing money laundering activities in financial networks. Smurfing involves dispersing large amounts of illegal money through small, often rapid transactions to avoid detection. Traditional methods fall short by producing high false-positive rates. SMoTeF addresses this by distinguishing fraudulent from non-fraudulent transactions with a focus on temporal order and maximum flow analysis.

## Key Features

- **Temporal Order Constraint:** Ensures that money laundering detection considers the sequence of transactions, improving accuracy by verifying the source of funds over time.
- **Maximum Temporal Flow Algorithm:** Calculates the maximum flow of money possible within the constraints of temporal order, minimizing false positives significantly.
- **Efficient Pattern Detection:** Introduces methods to detect, validate, and report suspicious transaction patterns with minimal runtime overhead.

## Algorithms

1. **Temporal Order Checking:** Validates the temporal sequence of transactions to ensure logical progression from source to destination.
2. **Maximum Temporal Flow Computation:** Computes the potential maximum flow in a transaction sequence, helping identify and prune unlikely transaction paths.
3. **Pattern Detection and Pruning:** Utilizes the above methodologies to detect and prune smurfing patterns, refining the detection process.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy and Pandas for numerical operations
