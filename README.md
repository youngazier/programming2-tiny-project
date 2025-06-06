# Programming 2 - Tiny Project
## Lecturer: Huynh Trung Hieu
## Student: Tran Trong Nhan - 10423160
This project consists of two main parts:
## [Part A (click here for more details)](./Part%20A/README.md)
We implemented custom `Vector`, `Matrix`, and `LinearSystem` class in C++.

These provide the core building blocks for performing numerical computations without relying on external libraries.

- **Vector Class**: supports basic vector operations and dynamic memory.

- **Matrix Class**: supports addition, multiplication, transpose, inverse, and pseudoinverse.

- **LinearSystem Class**: solves linear systems using Gaussian elimination, least squares, or regularized approaches.

These classes are reusable for any future numerical modeling work.

## [Part B](./Part%20B/README.md)
We used the [Computer Hardware dataset](https://archive.ics.uci.edu/dataset/29/computer+hardware) from the UCI Machine Learning Repository to build a linear regression model for predicting published relative performance (PRP).

- Data Parsing: read and normalize 6 features

- Model: $\text{PRP} = x_1 \cdot \text{MYCT} + x_2 \cdot \text{MMIN} + x_3 \cdot \text{MMAX} + x_4 \cdot \text{CACH} + x_5 \cdot \text{CHMIN} + x_6 \cdot \text{CHMAX}$

- Training: gradient descent over 5000 epochs

- Logging: RMSE is logged to CSV and visualized

### Dependences
- Standard C++ (no external math libraries)

- Python (for optional visualization)

## Folder Structure
```bash
├── tiny project
│   ├── Part A
│   │   ├── Vector.cpp
│   │   ├── Matrix.cpp
│   │   ├── LinearSystem.cpp
│   │   ├── PosSymLinSystem.cpp
│   │   ├── main.cpp
│   │
│   ├── Part B
│   │   ├── main.cpp
│   │   ├── plot.metrics.py
│   │   ├── Index (ignored)
│   │   ├── machine.data (ignored)
│   │   ├── machine.names (ignored)
│   │   ├── metrics.csv (ignored)
```

