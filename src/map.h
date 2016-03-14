#ifndef MAP_H
#define MAP_H

#include "algebra.h"
#include "algebra/solvers.h"
#include <iostream>

/** file map.h
 * \brief Maximum A Posteriori Estimators
 *
 * This file provides class templates for the inversion of a given forward model
 * using maximum a posteriori estimators. That means that a solution of the
 * inverse problem is obtained by minimizing the cost function
 *
 * \f[
 *   J(\mathbf{x}) &= \frac{1}{2}
 *   \left((\mathbf{F}(\mathbf{x}) - \mathbf{y}) \mathbf{S}_e
 *         (\mathbf{F}(\mathbf{x}) - \mathbf{y})
 *        +(\mathbf{x} - \mathbf{x}_a)\mathbf{S}_a
 *         (\mathbf{x} - \mathbf{x}_a) \right)
 * \f]
 *
 * To this end, this file provides the class template MAP, and three partial
 * specializations, one for each of the three different possible formulations of
 * the problem, here called the standard, n-form and m-form as given in formulas
 * (5.8), (5.9), (5.10) in \cite rodgers, respectively.
 *
 * Methods common to all three methods are aggregated in the base class MAPBase.
 *
 */

namespace invlib
{

/** \brief Formulation enum specifiying which formulation of the MAP estimator
 * to use.
 *
 * For details on the form see template specializations.
 */
enum class Formulation {STANDARD, NFORM, MFORM};

/**
 * \brief MAP base class
 *
 * Implements methods common to all MAP estimators independent of formulation.
 * Holds references to the forward model, the a priori state vector as well
 * as the a priori and measurement space covariance matrices. Provides a member
 * to hold a pointer to the measurement vector, when computing the estimator.
 * This is necessary for the class to be able to provide the
 * cost_function(const Vector &x) function taking only the current state space
 * vector as an argument, which is used by the optimizer to minimize the cost
 * function for a given measurement vector.
 *
 * To allow for maximum flexibility the type of the ForwardModel used is a
 * class template parameter. To be used with the MAP class, the functions
 * ForwardModel type must provide the following member functions:
 *
 * - evaluate(const Vector& x): Evaluate the forward model at the given
 * state space vector.
 * - Jacobian(const Vector& x): Compute the Jacobian of the forward model
 * at the given state space vector.
 *
 * \tparam ForwardModel The forward model type to be used.
 * \tparam Real The matrix type to be used for the linear model.
 * \tparam Vector The matrix type to be used for the linear model.
 * \tparam Matrix The matrix type to be used.
 * \tparam SaMatrix The type of the a priori matrix used for the computations.
 * \tparam SeMatrix The type of the measurement space covariance matrix to be
 * used.
 */
template
<
typename ForwardModel,
typename Real,
typename Vector,
typename Matrix,
typename SaMatrix,
typename SeMatrix
>
class MAPBase
{

public:

    Real cost_function(const Vector &x,
                       const Vector &y,
                       const Vector &yi)
    {
        Vector dy = y - yi;
        Vector dx = xa - x;
        return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
    }

    Real cost_function(const Vector &x) const
    {
        Vector dy = F.evaluate(x) - *y_ptr;
        Vector dx = xa - x;
        return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
    }

    Matrix gain_matrix(const Vector &x) const
    {
        Matrix K = F.Jacobian(x);
        Matrix SeInvK = inv(Se) * K;
        Matrix G = inv(transp(K) * SeInvK + inv(Sa)) * SeInvK;
        return G;
    }

    MAPBase( ForwardModel &F_,
            const Vector   &xa_,
            const SaMatrix &Sa_,
            const SeMatrix &Se_ )
        : F(F_), xa(xa_), Sa(Sa_), Se(Se_), K(), y_ptr(nullptr)
    {
        n = F_.n;
        m = F_.m;
    }



protected:

    unsigned int n, m;

    ForwardModel &F;
    const Vector   &xa;
    const Vector   *y_ptr;
    const Matrix     K;
    const SaMatrix &Sa;
    const SeMatrix &Se;
};

template
<
typename ForwardModel,
typename Real,
typename Vector,
typename Matrix,
typename SaMatrix,
typename SeMatrix,
Formulation Form = Formulation::STANDARD
>
class MAP;

template
<
typename ForwardModel,
typename Real,
typename Vector,
typename Matrix,
typename SaMatrix,
typename SeMatrix
>
class MAP<ForwardModel, Real, Vector, Matrix,
          SaMatrix, SeMatrix, Formulation::STANDARD>
    : public MAPBase<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix>
{

public:

    using Base = MAPBase<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix>;
    using Base::m; using Base::n;
    using Base::y_ptr; using Base::xa;
    using Base::F; using Base::K;
    using Base::Sa; using Base:: Se;
    using Base::cost_function;

    MAP( ForwardModel &F_,
         const Vector   &xa_,
         const SaMatrix &Sa_,
         const SeMatrix &Se_ )
        : Base(F_, xa_, Sa_, Se_) {}

    template<typename Minimizer>
    int compute( Vector       &x,
                 const Vector &y,
                 Minimizer M )
    {
        y_ptr = &y;
        x = xa;
        Vector yi = F.evaluate(x);
        Vector dx;

        bool converged     = false;
        unsigned int iter = 0;
        Real cost          = cost_function(x, y, yi);

        iter = 0;
        while (iter < M.maximum_iterations())
        {
            auto K   = F.Jacobian(x);
            auto tmp = transp(K) * inv(Se);
            auto H   = tmp * K + inv(Sa);
            Vector g = tmp * (yi - y) + inv(Sa) * (x - xa);

            if ((g.norm() / n) < M.tolerance())
            {
                converged = true;
                break;
            }

            M.step(dx, x, g, H, (*this));

            x += dx;
            yi = F.evaluate(x);
            iter++;
        }
    }
};

template
<
typename ForwardModel,
typename Real,
typename Vector,
typename Matrix,
typename SaMatrix,
typename SeMatrix
>
class MAP<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix, Formulation::NFORM>
    : public MAPBase<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix>
{

public:

    using Base = MAPBase<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix>;
    using Base::m; using Base::n;
    using Base::y_ptr; using Base::xa;
    using Base::F; using Base::K;
    using Base::Sa; using Base:: Se;
    using Base::cost_function;

    MAP( ForwardModel &F_,
         const Vector   &xa_,
         const SaMatrix &Sa_,
         const SeMatrix &Se_ )
        : Base(F_, xa_, Sa_, Se_) {}

    template<typename Minimizer>
    int compute( Vector       &x,
                 const Vector &y,
                 Minimizer M )
    {

        y_ptr = &y;
        x = xa;
        Vector yi = F.evaluate(x);
        Vector dx;

        bool converged = false;
        unsigned int iter = 0;
        Real cost = cost_function(x, y, yi);

        while (iter < M.maximum_iterations())
        {
            auto K   = F.Jacobian(x);
            auto tmp = transp(K) * inv(Se);
            Matrix H = tmp * K + inv(Sa);

            Vector g = tmp * (yi - y) + inv(Sa) * (x - xa);
            if ((g.norm() / n) < M.tolerance())
            {
                converged = true;
                break;
            }

            g = tmp * (y - yi + (K * (x - xa)));
            M.step(dx, xa, g, H, (*this));

            x = xa - dx;
            yi = F.evaluate(x);
            iter++;
        }
    }
};

template
<
typename ForwardModel,
typename Real,
typename Vector,
typename Matrix,
typename SaMatrix,
typename SeMatrix
>
class MAP<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix, Formulation::MFORM>
    : public MAPBase<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix>
{

public:

    using Base = MAPBase<ForwardModel, Real, Vector, Matrix, SaMatrix, SeMatrix>;
    using Base::m; using Base::n;
    using Base::y_ptr; using Base::xa;
    using Base::F; using Base::K;
    using Base::Sa; using Base:: Se;

    MAP( ForwardModel &F_,
         const Vector   &xa_,
         const SaMatrix &Sa_,
         const SeMatrix &Se_ )
        : Base(F_, xa_, Sa_, Se_) {}

    Matrix gain_matrix(const Vector &x) const
    {
        Matrix K = F.Jacobian(x);
        Matrix SaKT = Sa * transp(K);
        Matrix G = SaKT * inv(K * SaKT + Se);
        return G;
    }

    Real cost_function(const Vector &x,
                       const Vector &y,
                       const Vector &yi)
    {
        Vector dy(y - yi);
        Vector dx(xa - x);
        return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
    }

    Real cost_function(const Vector &x) const
    {
        Vector dy(F.evaluate(x) - *y_ptr);
        Vector dx = xa + (-1.0) * Sa * transp(K) * x;
        return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
    }

    template<typename Minimizer>
    int compute( Vector       &x,
                 const Vector &y,
                 Minimizer M )
    {

        y_ptr = &y;
        x = xa;
        Vector yi = F.evaluate(x), yold;
        Vector dx;

        bool converged = false;
        unsigned int iter = 0;

        while (iter < M.maximum_iterations())
        {
            auto K   = F.Jacobian(x);
            auto tmp = Sa * transp(K);
            Matrix H   = Se + K * tmp;
            Vector g = y - yi + K * (x - xa);

            M.step(dx, xa, g, H, (*this));
            x = xa - tmp * dx;

            yold = yi;
            yi = F.evaluate(x);
            Vector dy = yi - yold;
            Vector r = Se * H * Se * dy;

            if ((dot(dy, r) / m) < M.tolerance())
            {
                converged = true;
                break;
            }
            iter++;
        }
    }
};

}
#endif // MAP_H
