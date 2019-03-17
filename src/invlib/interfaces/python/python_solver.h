template<typename VectorType>
class CGPythonSettings {
public:

    using RealType = VectorType::RealType;

    CGPythonSettings(double tol,
                     double step_lim)
        : tolerance(tol), step_limin(step_lim)
    {
        // Nothing to do here.
    }

    CGPythonSettings(const CGPytonSettings &)  = default;
    CGPythonSettings(      CGPytonSettings &&) = default;
    CGPythonSettings & operator=(const CGPythonSettings &)  = default;
    CGPythonSettings & operator=(      CGPythonSettings &&) = default;
    ~CGPythonSettings() = default;

    const VectorType & start_vector(const VectorType &w) {
        auto a_ptr = reinterpret_cast<void *>(&w);

        VectorType & b_ptr;
        if (start_vector_ptr) {
            b_ptr = start_vector_ptr(a_ptr);
        } else {
            b_ptr = new VectorType();
            b_ptr->resize(w.nrows());
        }
        return & reinterpret_cast<VectorType *>(b_ptr);
    }

    bool converged(const VectorType &r,
                   const VectorType &v) {

        RealType t;
        if (relative) {
            t = r.norm() / v.norm();
        } else {
            t = r.norm();
        }
        if (t < tolerance) {
            return true;
        }

        if (steps >= step_limit) {
            return true;
        }

        return false;
    }

private:

    void (start_vector_ptr*)(void *, void *) = nullptr;
    bool     relative   = true;
    RealType tolerance  = 1e-6;
    size_t   step_limit = 1e3;

}
