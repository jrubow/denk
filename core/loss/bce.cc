#include "bce.hh"
#include "matrix.hh"
#include <cmath>

Matrix BCE::compute(const Matrix *predicted, const Matrix *actual) const {
    const double eps = 1e-12;
    double sum = 0.0;
    int rows = predicted->getRows();
    int cols = predicted->getCols();
    int N = rows * cols;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double p = predicted->get(r, c);
            double y = actual->get(r, c);
            // clamp p to avoid log(0)
            p = std::min(1.0 - eps, std::max(eps, p));
            sum += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        }
    }
    double loss = sum / static_cast<double>(N);
    return Matrix(1, 1, std::vector<double>{loss});
}

Matrix BCE::derivate(const Matrix *predicted, const Matrix *actual) const {
    const double eps = 1e-12;
    int rows = predicted->getRows();
    int cols = predicted->getCols();
    Matrix out(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double p = predicted->get(r, c);
            double y = actual->get(r, c);
            p = std::min(1.0 - eps, std::max(eps, p));
            double val = (p - y) / (p * (1.0 - p));
            out.set(r, c, val / static_cast<double>(rows * cols));
        }
    }
    return out;
}
