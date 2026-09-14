// Generic deterministic integration; F supplies generated RHS/readout and VJPs.
// Two RK4 half-steps per cell expose a midpoint readout, matching the direct
// likelihood's coefficient sampling convention without any model-specific math.
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>

namespace pnl_continuous {

inline double sigmoid(double x) {
    if (x >= 0.) { const double e = std::exp(-x); return 1. / (1. + e); }
    const double e = std::exp(x); return e / (1. + e);
}

template<class F> using State = std::array<double, F::S>;

template<class F>
struct RK4Tape {
    State<F> a, b, c, k1, k2, k3, k4;
};

template<class F>
State<F> rk4(const State<F>& x, const double* u, const double* p,
             double t, double h, RK4Tape<F>& q) {
    F::rhs(x.data(), u, p, t, q.k1.data());
    for (int j = 0; j < F::S; ++j) q.a[j] = x[j] + .5 * h * q.k1[j];
    F::rhs(q.a.data(), u, p, t + .5 * h, q.k2.data());
    for (int j = 0; j < F::S; ++j) q.b[j] = x[j] + .5 * h * q.k2[j];
    F::rhs(q.b.data(), u, p, t + .5 * h, q.k3.data());
    for (int j = 0; j < F::S; ++j) q.c[j] = x[j] + h * q.k3[j];
    F::rhs(q.c.data(), u, p, t + h, q.k4.data());
    State<F> y;
    for (int j = 0; j < F::S; ++j)
        y[j] = x[j] + h / 6. * (q.k1[j] + 2. * q.k2[j] + 2. * q.k3[j] + q.k4[j]);
    return y;
}

template<class F>
State<F> rk4_vjp(const State<F>& x, const double* u, const double* p,
                 double t, double h, const RK4Tape<F>& q, const State<F>& gy,
                 double* gu, double* gp, double& gt, double& gh) {
    State<F> gx = gy, g1, g2, g3, g4, z{};
    for (int j = 0; j < F::S; ++j) {
        g1[j] = h / 6. * gy[j]; g2[j] = h / 3. * gy[j];
        g3[j] = h / 3. * gy[j]; g4[j] = h / 6. * gy[j];
        gh += gy[j] / 6. * (q.k1[j] + 2. * q.k2[j] + 2. * q.k3[j] + q.k4[j]);
    }
    double time_grad = 0.;
    F::rhs_vjp(q.c.data(), u, p, t + h, g4.data(), z.data(), gu, gp, &time_grad);
    gt += time_grad; gh += time_grad;
    for (int j = 0; j < F::S; ++j) {
        gx[j] += z[j]; gh += z[j] * q.k3[j]; g3[j] += h * z[j];
    }
    z.fill(0.); time_grad = 0.;
    F::rhs_vjp(q.b.data(), u, p, t + .5 * h, g3.data(), z.data(), gu, gp, &time_grad);
    gt += time_grad; gh += .5 * time_grad;
    for (int j = 0; j < F::S; ++j) {
        gx[j] += z[j]; gh += .5 * z[j] * q.k2[j]; g2[j] += .5 * h * z[j];
    }
    z.fill(0.); time_grad = 0.;
    F::rhs_vjp(q.a.data(), u, p, t + .5 * h, g2.data(), z.data(), gu, gp, &time_grad);
    gt += time_grad; gh += .5 * time_grad;
    for (int j = 0; j < F::S; ++j) {
        gx[j] += z[j]; gh += .5 * z[j] * q.k1[j]; g1[j] += .5 * h * z[j];
    }
    F::rhs_vjp(x.data(), u, p, t, g1.data(), gx.data(), gu, gp, &gt);
    return gx;
}

template<class F>
void check(const at::Tensor& state, const at::Tensor& inputs, const at::Tensor& params,
           const at::Tensor& duration, const at::Tensor& start, const at::Tensor& steps) {
    TORCH_CHECK(state.dim() == 2 && state.size(1) == F::S, "Incorrect state shape.");
    const auto batch = state.size(0);
    for (const auto& v : {state, inputs, params, duration, start})
        TORCH_CHECK(v.device().is_cpu() && v.scalar_type() == at::kDouble && v.is_contiguous(),
                    "Dynamics inputs must be contiguous CPU float64.");
    TORCH_CHECK(inputs.sizes() == at::IntArrayRef({batch, F::I}), "Incorrect input shape.");
    TORCH_CHECK(params.sizes() == at::IntArrayRef({batch, F::P}), "Incorrect parameter shape.");
    TORCH_CHECK(duration.sizes() == at::IntArrayRef({batch}) && start.sizes() == duration.sizes(), "Incorrect clock shape.");
    TORCH_CHECK(steps.device().is_cpu() && steps.is_contiguous() && steps.scalar_type() == at::kLong
                && steps.sizes() == duration.sizes(), "Incorrect integration counts.");
}

template<class F>
std::vector<at::Tensor> forward(const at::Tensor& state, const at::Tensor& inputs, const at::Tensor& params,
    const at::Tensor& duration, const at::Tensor& start, const at::Tensor& steps, bool store_history) {
    check<F>(state, inputs, params, duration, start, steps);
    const auto batch = state.size(0);
    const auto counts = steps.data_ptr<int64_t>();
    int64_t maximum = 0;
    for (int64_t lane = 0; lane < batch; ++lane) {
        TORCH_CHECK(counts[lane] >= 0, "Integration counts must be nonnegative.");
        maximum = std::max(maximum, counts[lane]);
    }
    at::Tensor final = at::empty_like(state);
    at::Tensor readouts = at::zeros({batch, maximum, F::R}, state.options());
    at::Tensor history = store_history ? at::zeros({maximum + 1, batch, F::S}, state.options()) : at::empty({0}, state.options());
    const auto x0 = state.data_ptr<double>(), uv = inputs.data_ptr<double>(), pv = params.data_ptr<double>();
    const auto dv = duration.data_ptr<double>(), tv = start.data_ptr<double>();
    auto yf = final.data_ptr<double>(), rv = readouts.data_ptr<double>(), tape = history.data_ptr<double>();
    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
        State<F> x; std::copy_n(x0 + lane * F::S, F::S, x.begin());
        const auto u = F::I ? uv + lane * F::I : uv;
        const auto p = F::P ? pv + lane * F::P : pv;
        const double h = counts[lane] ? dv[lane] / counts[lane] : 0.;
        if (store_history) std::copy_n(x.data(), F::S, tape + lane * F::S);
        for (int64_t k = 0; k < counts[lane]; ++k) {
            const double t = tv[lane] + k * h;
            RK4Tape<F> q;
            const auto midpoint = rk4<F>(x, u, p, t, .5 * h, q);
            if (F::R) F::readout(midpoint.data(), u, p, t + .5 * h, rv + (lane * maximum + k) * F::R);
            x = rk4<F>(midpoint, u, p, t + .5 * h, .5 * h, q);
            if (store_history) std::copy_n(x.data(), F::S, tape + ((k + 1) * batch + lane) * F::S);
        }
        std::copy_n(x.data(), F::S, yf + lane * F::S);
    }
    return {readouts, final, history};
}

template<class F>
std::vector<at::Tensor> backward(const at::Tensor& history, const at::Tensor& state,
    const at::Tensor& inputs, const at::Tensor& params, const at::Tensor& duration,
    const at::Tensor& start, const at::Tensor& steps, const at::Tensor& grad_readouts, const at::Tensor& grad_final) {
    check<F>(state, inputs, params, duration, start, steps);
    const int64_t batch = state.size(0);
    const auto counts = steps.data_ptr<int64_t>();
    int64_t maximum = 0;
    for (int64_t lane = 0; lane < batch; ++lane) {
        TORCH_CHECK(counts[lane] >= 0, "Integration counts must be nonnegative.");
        maximum = std::max(maximum, counts[lane]);
    }
    for (const auto& v : {history, grad_readouts, grad_final})
        TORCH_CHECK(v.device().is_cpu() && v.scalar_type() == at::kDouble && v.is_contiguous(), "Invalid adjoint buffer type.");
    TORCH_CHECK(history.sizes() == at::IntArrayRef({maximum + 1, batch, F::S}), "Invalid history shape.");
    TORCH_CHECK(grad_readouts.sizes() == at::IntArrayRef({batch, maximum, F::R}), "Invalid readout gradient shape.");
    TORCH_CHECK(grad_final.sizes() == state.sizes(), "Invalid final-state gradient shape.");
    auto gx = at::zeros_like(state), gu = at::zeros_like(inputs), gp = at::zeros_like(params);
    auto gd = at::zeros_like(duration), gt = at::zeros_like(start);
    const auto uv = inputs.data_ptr<double>(), pv = params.data_ptr<double>(), dv = duration.data_ptr<double>(), tv = start.data_ptr<double>();
    const auto tape = history.data_ptr<double>(), gr = grad_readouts.data_ptr<double>(), gf = grad_final.data_ptr<double>();
    auto dx = gx.data_ptr<double>(), du = gu.data_ptr<double>(), dp = gp.data_ptr<double>(), dd = gd.data_ptr<double>(), dt = gt.data_ptr<double>();
    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
        State<F> gradient; std::copy_n(gf + lane * F::S, F::S, gradient.begin());
        const auto u = F::I ? uv + lane * F::I : uv;
        const auto p = F::P ? pv + lane * F::P : pv;
        auto input_gradient = F::I ? du + lane * F::I : du;
        auto param_gradient = F::P ? dp + lane * F::P : dp;
        const double h = counts[lane] ? dv[lane] / counts[lane] : 0.;
        for (int64_t k = counts[lane]; k-- > 0;) {
            State<F> x; std::copy_n(tape + (k * batch + lane) * F::S, F::S, x.begin());
            const double t = tv[lane] + k * h;
            RK4Tape<F> q1, q2;
            const auto midpoint = rk4<F>(x, u, p, t, .5 * h, q1);
            rk4<F>(midpoint, u, p, t + .5 * h, .5 * h, q2);
            double gtime1 = 0., gtime2 = 0., gh1 = 0., gh2 = 0., gtime_read = 0.;
            auto gm = rk4_vjp<F>(midpoint, u, p, t + .5 * h, .5 * h, q2, gradient,
                                input_gradient, param_gradient, gtime2, gh2);
            if (F::R) F::readout_vjp(midpoint.data(), u, p, t + .5 * h, gr + (lane * maximum + k) * F::R,
                                    gm.data(), input_gradient, param_gradient, &gtime_read);
            gradient = rk4_vjp<F>(x, u, p, t, .5 * h, q1, gm, input_gradient, param_gradient, gtime1, gh1);
            const double clock_gradient = gtime1 + gtime2 + gtime_read;
            dt[lane] += clock_gradient;
            dd[lane] += (.5 * (gh1 + gh2 + gtime2 + gtime_read) + k * clock_gradient) / counts[lane];
        }
        std::copy_n(gradient.data(), F::S, dx + lane * F::S);
    }
    return {gx, gu, gp, gd, gt};
}

} // namespace pnl_continuous
