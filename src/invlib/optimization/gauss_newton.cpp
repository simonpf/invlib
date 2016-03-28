// ------------------------------- //
//  Constructors and Destructors   //
// ------------------------------- //

template
<
typename RealType,
typename Solver = Standard
>
GaussNewton<RealType, Solve>::GaussNewton(Solver solver_ = Solver())
    : tolerance(tolerance_), max_iter(1000), solver(solver_)
{
    // Nothing to do here.
}

template
<
typename RealType,
typename Solver = Standard
>
GaussNewton<RealType, Solve>::GaussNewton(RealType tolerance,
                                          unsigned int maximum_iterations_,
                                          Solver solver_ = Solver())
    : tolerance(tolerance_), max_iter(maximum_iterations_), solver(solver_)
{
    // Nothing to do here.
}

// -------------------------- //
//    Getters and Setters     //
// -------------------------- //

template
<
typename RealType,
typename Solver = Standard
>
auto GaussNewton<RealType, Solve>::get_maximum_iterations()
    -> unsigned int
{
    return maximum_iterations;
}

template
<
typename RealType,
typename Solver = Standard
>
void GaussNewton<RealType, Solve>::set_maximum_iterations(unsigned int n)
{
    maximum_iterations = n;
}

template
<
typename RealType,
typename Solver = Standard
>
auto GaussNewton<RealType, Solve>::get_tolerance()
    -> unsigned int
{
    return tolerance;
}

template
<
typename RealType,
typename Solver = Standard
>
void GaussNewton<RealType, Solve>::set_tolerance(RealType tolerance_)
{
    tolerance = tolerance_;
}

template
<
typename RealType,
typename Solver = Standard
>
GaussNewton<RealType, Solve>::GaussNewton(RealType tolerance_,
                                          RealType maximum_iterations_,
                                          Solver   solve_ = Solver())
    : tolerance(1e-5), maximum_iterations(maximum_iterations_), solver(solver_)
{
    // Nothing to do here.
}

// --------------------------- //
//  Perform Minimization Step  //
// --------------------------- //

template
<
typename RealType,
typename Solver = Standard
>
template
<
typename VectorType,
typename MatrixType,
typename CostFunction
>
auto GaussNewton<RealType, Solve>
::step<VectorType, MatrixType, CostFunction>(const Vector &
                                             const Vector &g,
                                             const Matrix &B,
                                             const CostFunction &)
    -> VectorType
{
    dx = -1.0 * solver.solve(B, g);
    return 0;
}
