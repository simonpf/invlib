template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
MAPBase<ForwardModel, MatrixType, SaType, SeType>
::MAPBase(ForwardModel     &F_,
          const VectorType &xa_,
          const SaType     &Sa_,
          const SeType     &Se_)
    : F(F_), xa(xa_), Sa(Sa_), Se(Se_), K(), y_ptr(nullptr), n(F_.n), m(F_.m)
{
    // Nothing to do here.
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAPBase<ForwardModel, MatrixType, SaType, SeType>
::cost_function(const VectorType &x,
                const VectorType &y,
                const VectorType &yi)
    -> RealType
{
    VectorType dy = y - yi;
    VectorType dx = xa - x;
    return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAPBase<ForwardModel, MatrixType, SaType, SeType>
::cost_function(const VectorType &x)
    -> RealType
{
    VectorType  y = F.evaluate(x);
    VectorType dy = y - *y_ptr;
    VectorType dx = xa - x;
    return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAPBase<ForwardModel, MatrixType, SaType, SeType>
::cost_x(const VectorType &x)
    -> RealType
{
    VectorType dx = (xa - x);
    return dot(dx, inv(Sa) * dx);
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAPBase<ForwardModel, MatrixType, SaType, SeType>
::cost_y(const VectorType &y,
         const VectorType &yi)
    -> RealType
{
    VectorType dy = y - yi;
    return dot(dy, inv(Se) * dy);
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAPBase<ForwardModel, MatrixType, SaType, SeType>
::gain_matrix(const VectorType &x)
    -> MatrixType
{
    K = F.Jacobian(x);
    MatrixType tmp = transp(K) * inv(Se);
    MatrixType G = inv(tmp * K + inv(Sa)) * tmp;
    return G;
}

// ------------- //
//   Standard    //
// ------------- //

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::STANDARD>
::MAP( ForwardModel &F_,
       const VectorType   &xa_,
       const SaType &Sa_,
       const SeType &Se_ )
    : Base(F_, xa_, Sa_, Se_)
{
    // Nothing to do here.
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
template<typename Minimizer, template <LogType> typename Log>
auto MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::STANDARD>
::compute(VectorType       &x,
          const VectorType &y,
          Minimizer M,
          int verbosity)
    -> int
{

    Log<LogType::MAP> log(verbosity);

    y_ptr = &y;
    x = xa;
    VectorType yi = F.evaluate(x);
    VectorType dx;

    bool converged     = false;
    unsigned int iter = 0;

    RealType cost, cost_x, cost_y;
    cost_x = this->cost_x(x);
    cost_y = this->cost_y(y, yi);
    cost   = cost_x + cost_y;

    iter = 0;
    while (iter < M.maximum_iterations())
    {
        K        = F.Jacobian(x);
        auto tmp = transp(K) * inv(Se);

        // Compute Hessian and transform.
        auto H  = tmp * K + inv(Sa);

        // Compute gradient and transform.
        VectorType g  = tmp * (yi - y) + inv(Sa) * (x - xa);

        if ((g.norm() / n) < M.tolerance())
        {
            converged = true;
            break;
        }

        M.step(dx, x, g, H, (*this));
        x += dx;
        yi = F.evaluate(x);
        iter++;

        cost_x = this->cost_x(x);
        cost_y = this->cost_y(y, yi);
        cost   = cost_x + cost_y;

        log.step(iter, cost, cost_x, cost_y);
    }

    log.finalize(converged, iter, cost, cost_x, cost_y);

    return 0;
}

// --------------- //
//     N-form      //
// --------------- //

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::NFORM>
::MAP( ForwardModel &F_,
       const VectorType   &xa_,
       const SaType &Sa_,
       const SeType &Se_ )
    : Base(F_, xa_, Sa_, Se_)
{
    // Nothing to do here.
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
template<typename Minimizer, template <LogType> typename Log>
auto MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::NFORM>
::compute(VectorType       &x,
          const VectorType &y,
          Minimizer M,
          int verbosity)
    -> int
{

    Log<LogType::MAP> log(verbosity);

    y_ptr = &y;
    x = xa;
    VectorType yi = F.evaluate(x);
    VectorType dx;

    bool converged = false;
    unsigned int iter = 0;

    RealType cost, cost_x, cost_y;
    cost_x = this->cost_x(x);
    cost_y = this->cost_y(y, yi);
    cost   = cost_x + cost_y;

    while (iter < M.maximum_iterations())
    {
        K        = F.Jacobian(x);
        auto tmp = transp(K) * inv(Se);

        // Compute true gradient for convergence test.
        VectorType g  = tmp * (yi - y) + inv(Sa) * (x - xa);

        if ((g.norm() / n) < M.tolerance())
        {
            converged = true;
            break;
        }

        // Compute Hessian and transform.
        auto H  = tmp * K + inv(Sa);

        // Compute gradient and transform.
        g = tmp * (y - yi + (K * (x - xa)));

        M.step(dx, xa, g, H, (*this));

        x = xa - dx;
        yi = F.evaluate(x);
        iter++;

        cost_x = this->cost_x(x);
        cost_y = this->cost_y(y, yi);
        cost   = cost_x + cost_y;

        log.step(iter, cost, cost_x, cost_y);
    }

    log.finalize(converged, iter, cost, cost_x, cost_y);

    return 0;
}

// --------------- //
//     M-form      //
// --------------- //

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::MFORM>
::MAP( ForwardModel &F_,
       const VectorType   &xa_,
       const SaType &Sa_,
       const SeType &Se_ )
    : Base(F_, xa_, Sa_, Se_)
{
    // Nothing to do here.
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::MFORM>
::gain_matrix(const VectorType &x)
    -> MatrixType
{
    K = F.Jacobian(x);
    MatrixType SaKT = Sa * transp(K);
    MatrixType G = SaKT * inv(K * SaKT + Se);
    return G;
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::MFORM>
::cost_function(const VectorType &x,
                const VectorType &y,
                const VectorType &yi)
    -> RealType
{
    VectorType dy(y - yi);
    VectorType dx(xa - x);
    return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
auto MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::MFORM>
::cost_function(const VectorType &x)
    -> RealType
{
    VectorType dy(F.evaluate(x) - *y_ptr);
    VectorType dx = xa + (-1.0) * Sa * transp(K) * x;
    return dot(dy, inv(Se) * dy) + dot(dx, inv(Sa) * dx);
}

template
<
typename ForwardModel,
typename MatrixType,
typename SaType,
typename SeType
>
template<typename Minimizer, template <LogType> typename Log>
auto MAP<ForwardModel, MatrixType, SaType, SeType, Formulation::MFORM>
::compute(VectorType       &x,
          const VectorType &y,
          Minimizer M,
          int verbosity)
    -> int
{

    Log<LogType::MAP> log(verbosity);

    y_ptr = &y;
    x = xa;
    VectorType yi = F.evaluate(x), yold;
    VectorType dx;

    bool converged = false;
    unsigned int iter = 0;

    while (iter < M.maximum_iterations())
    {
        K   = F.Jacobian(x);
        auto tmp = Sa * transp(K);

        // Compute Hessian.
        auto H   = Se + K * tmp;

        // Compute gradient.
        VectorType g  = y - yi + K * (x - xa);

        M.step(dx, xa, g, H, (*this));
        x = xa - tmp * dx;

        yold = yi;
        yi = F.evaluate(x);
        VectorType dy = yi - yold;
        VectorType r = Se * H * Se * dy;

        if ((dot(dy, r) / m) < M.tolerance())
        {
            converged = true;
            break;
        }
        iter++;
    }
    return 0;
}
