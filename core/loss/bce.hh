#ifndef BCE_H
#define BCE_H

#include "loss.hh"

class BCE : public Loss {
public:
    BCE() = default;
    Matrix compute(const Matrix *predicted, const Matrix *actual) const override;
    Matrix derivate(const Matrix *predicted, const Matrix *actual) const override;
};

#endif // BCE_H
