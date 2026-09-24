// Fused CPU kernels for the complete deterministic-LCA CSI direct likelihood.
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

template <typename scalar_t>
inline scalar_t logistic(scalar_t value) {
    if (value >= scalar_t(0)) {
        const scalar_t exponent = std::exp(-value);
        return scalar_t(1) / (scalar_t(1) + exponent);
    }
    const scalar_t exponent = std::exp(value);
    return exponent / (scalar_t(1) + exponent);
}

template <typename scalar_t>
struct State {
    scalar_t first;
    scalar_t second;
};

template <typename scalar_t>
struct StepGradient {
    State<scalar_t> state;
    State<scalar_t> task;
    scalar_t gain;
    scalar_t step;
};

template <typename scalar_t>
inline State<scalar_t> add(State<scalar_t> lhs, State<scalar_t> rhs) {
    return {lhs.first + rhs.first, lhs.second + rhs.second};
}

template <typename scalar_t>
inline State<scalar_t> scale(scalar_t amount, State<scalar_t> value) {
    return {amount * value.first, amount * value.second};
}

template <typename scalar_t>
inline scalar_t dot(State<scalar_t> lhs, State<scalar_t> rhs) {
    return lhs.first * rhs.first + lhs.second * rhs.second;
}

template <typename scalar_t>
inline State<scalar_t> rhs(
    State<scalar_t> state,
    State<scalar_t> task,
    scalar_t gain,
    scalar_t leak,
    scalar_t competition
) {
    const scalar_t first_activity = logistic(gain * state.first);
    const scalar_t second_activity = logistic(gain * state.second);
    return {
        -leak * state.first + task.first - competition * second_activity,
        -leak * state.second + task.second - competition * first_activity,
    };
}

template <typename scalar_t>
inline State<scalar_t> euler_step(
    State<scalar_t> state,
    State<scalar_t> task,
    scalar_t gain,
    scalar_t step,
    scalar_t leak,
    scalar_t competition
) {
    return add(
        state,
        scale(step, rhs(state, task, gain, leak, competition))
    );
}

template <typename scalar_t>
inline StepGradient<scalar_t> rhs_vjp(
    State<scalar_t> state,
    scalar_t gain,
    scalar_t leak,
    scalar_t competition,
    State<scalar_t> gradient
) {
    const scalar_t first_activity = logistic(gain * state.first);
    const scalar_t second_activity = logistic(gain * state.second);
    const scalar_t first_slope = first_activity * (scalar_t(1) - first_activity);
    const scalar_t second_slope = second_activity * (scalar_t(1) - second_activity);
    return {
        {
            -leak * gradient.first
                - competition * gain * first_slope * gradient.second,
            -competition * gain * second_slope * gradient.first
                - leak * gradient.second,
        },
        gradient,
        -competition
            * (second_slope * state.second * gradient.first
               + first_slope * state.first * gradient.second),
        scalar_t(0),
    };
}

template <typename scalar_t>
inline State<scalar_t> rk4_step(
    State<scalar_t> state,
    State<scalar_t> task,
    scalar_t gain,
    scalar_t step,
    scalar_t leak,
    scalar_t competition
) {
    const State<scalar_t> k1 = rhs(state, task, gain, leak, competition);
    const State<scalar_t> k2 = rhs(
        add(state, scale(scalar_t(0.5) * step, k1)),
        task,
        gain,
        leak,
        competition
    );
    const State<scalar_t> k3 = rhs(
        add(state, scale(scalar_t(0.5) * step, k2)),
        task,
        gain,
        leak,
        competition
    );
    const State<scalar_t> k4 = rhs(
        add(state, scale(step, k3)), task, gain, leak, competition
    );
    return {
        state.first
            + (step / scalar_t(6))
                * (k1.first + scalar_t(2) * k2.first
                   + scalar_t(2) * k3.first + k4.first),
        state.second
            + (step / scalar_t(6))
                * (k1.second + scalar_t(2) * k2.second
                   + scalar_t(2) * k3.second + k4.second),
    };
}

template <typename scalar_t>
struct DriftGradient {
    State<scalar_t> midpoint;
    scalar_t gain;
    scalar_t correct_response;
    scalar_t stimulus[4];
};

template <typename scalar_t>
inline scalar_t drift_value(
    const scalar_t* stimulus,
    State<scalar_t> midpoint,
    scalar_t gain,
    scalar_t correct_response
) {
    const scalar_t control_first = logistic(gain * midpoint.first);
    const scalar_t control_second = logistic(gain * midpoint.second);
    const scalar_t a = logistic(
        stimulus[0] - stimulus[1] + scalar_t(4) * control_first
        - scalar_t(4)
    );
    const scalar_t b = logistic(
        stimulus[1] - stimulus[0] + scalar_t(4) * control_first
        - scalar_t(4)
    );
    const scalar_t c = logistic(
        stimulus[2] - stimulus[3] + scalar_t(4) * control_second
        - scalar_t(4)
    );
    const scalar_t d = logistic(
        stimulus[3] - stimulus[2] + scalar_t(4) * control_second
        - scalar_t(4)
    );
    const scalar_t contrast = a - b + c - d;
    return (logistic(contrast) - logistic(-contrast)) * correct_response;
}

template <typename scalar_t>
inline DriftGradient<scalar_t> drift_vjp(
    const scalar_t* stimulus,
    State<scalar_t> midpoint,
    scalar_t gain,
    scalar_t correct_response,
    scalar_t gradient_drift
) {
    const scalar_t control_first = logistic(gain * midpoint.first);
    const scalar_t control_second = logistic(gain * midpoint.second);
    const scalar_t a = logistic(
        stimulus[0] - stimulus[1] + scalar_t(4) * control_first
        - scalar_t(4)
    );
    const scalar_t b = logistic(
        stimulus[1] - stimulus[0] + scalar_t(4) * control_first
        - scalar_t(4)
    );
    const scalar_t c = logistic(
        stimulus[2] - stimulus[3] + scalar_t(4) * control_second
        - scalar_t(4)
    );
    const scalar_t d = logistic(
        stimulus[3] - stimulus[2] + scalar_t(4) * control_second
        - scalar_t(4)
    );
    const scalar_t contrast = a - b + c - d;
    const scalar_t positive = logistic(contrast);
    const scalar_t negative = logistic(-contrast);
    const scalar_t gradient_contrast = gradient_drift * correct_response
        * (positive * (scalar_t(1) - positive)
           + negative * (scalar_t(1) - negative));

    const scalar_t gradient_a = gradient_contrast
        * a * (scalar_t(1) - a);
    const scalar_t gradient_b = -gradient_contrast
        * b * (scalar_t(1) - b);
    const scalar_t gradient_c = gradient_contrast
        * c * (scalar_t(1) - c);
    const scalar_t gradient_d = -gradient_contrast
        * d * (scalar_t(1) - d);
    const scalar_t gradient_control_first = scalar_t(4)
        * (gradient_a + gradient_b);
    const scalar_t gradient_control_second = scalar_t(4)
        * (gradient_c + gradient_d);
    const scalar_t first_activity_gradient = gradient_control_first
        * control_first * (scalar_t(1) - control_first);
    const scalar_t second_activity_gradient = gradient_control_second
        * control_second * (scalar_t(1) - control_second);

    DriftGradient<scalar_t> result;
    result.midpoint = {
        first_activity_gradient * gain,
        second_activity_gradient * gain,
    };
    result.gain = first_activity_gradient * midpoint.first
        + second_activity_gradient * midpoint.second;
    result.correct_response = gradient_drift * (positive - negative);
    result.stimulus[0] = gradient_a - gradient_b;
    result.stimulus[1] = -gradient_a + gradient_b;
    result.stimulus[2] = gradient_c - gradient_d;
    result.stimulus[3] = -gradient_c + gradient_d;
    return result;
}

template <typename scalar_t>
inline StepGradient<scalar_t> rk4_step_vjp(
    State<scalar_t> state,
    State<scalar_t> task,
    scalar_t gain,
    scalar_t step,
    scalar_t leak,
    scalar_t competition,
    State<scalar_t> gradient_output
) {
    const State<scalar_t> k1 = rhs(state, task, gain, leak, competition);
    const State<scalar_t> second_state = add(
        state, scale(scalar_t(0.5) * step, k1)
    );
    const State<scalar_t> k2 = rhs(
        second_state, task, gain, leak, competition
    );
    const State<scalar_t> third_state = add(
        state, scale(scalar_t(0.5) * step, k2)
    );
    const State<scalar_t> k3 = rhs(
        third_state, task, gain, leak, competition
    );
    const State<scalar_t> fourth_state = add(state, scale(step, k3));
    const State<scalar_t> k4 = rhs(
        fourth_state, task, gain, leak, competition
    );

    State<scalar_t> gradient_state = gradient_output;
    State<scalar_t> gradient_task{scalar_t(0), scalar_t(0)};
    scalar_t gradient_gain = scalar_t(0);
    const State<scalar_t> weighted_sum{
        (k1.first + scalar_t(2) * k2.first + scalar_t(2) * k3.first
         + k4.first) / scalar_t(6),
        (k1.second + scalar_t(2) * k2.second + scalar_t(2) * k3.second
         + k4.second) / scalar_t(6),
    };
    scalar_t gradient_step = dot(gradient_output, weighted_sum);

    State<scalar_t> gradient_k1 = scale(
        step / scalar_t(6), gradient_output
    );
    State<scalar_t> gradient_k2 = scale(
        step / scalar_t(3), gradient_output
    );
    State<scalar_t> gradient_k3 = scale(
        step / scalar_t(3), gradient_output
    );
    const State<scalar_t> gradient_k4 = scale(
        step / scalar_t(6), gradient_output
    );

    const StepGradient<scalar_t> fourth = rhs_vjp(
        fourth_state, gain, leak, competition, gradient_k4
    );
    gradient_task = add(gradient_task, fourth.task);
    gradient_gain += fourth.gain;
    gradient_state = add(gradient_state, fourth.state);
    gradient_step += dot(fourth.state, k3);
    gradient_k3 = add(gradient_k3, scale(step, fourth.state));

    const StepGradient<scalar_t> third = rhs_vjp(
        third_state, gain, leak, competition, gradient_k3
    );
    gradient_task = add(gradient_task, third.task);
    gradient_gain += third.gain;
    gradient_state = add(gradient_state, third.state);
    gradient_step += scalar_t(0.5) * dot(third.state, k2);
    gradient_k2 = add(
        gradient_k2, scale(scalar_t(0.5) * step, third.state)
    );

    const StepGradient<scalar_t> second = rhs_vjp(
        second_state, gain, leak, competition, gradient_k2
    );
    gradient_task = add(gradient_task, second.task);
    gradient_gain += second.gain;
    gradient_state = add(gradient_state, second.state);
    gradient_step += scalar_t(0.5) * dot(second.state, k1);
    gradient_k1 = add(
        gradient_k1, scale(scalar_t(0.5) * step, second.state)
    );

    const StepGradient<scalar_t> first = rhs_vjp(
        state, gain, leak, competition, gradient_k1
    );
    gradient_task = add(gradient_task, first.task);
    gradient_gain += first.gain;
    gradient_state = add(gradient_state, first.state);

    return {
        gradient_state, gradient_task, gradient_gain, gradient_step
    };
}

void check_inputs(
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& csi_duration,
    const at::Tensor& state_duration,
    const at::Tensor& csi_steps,
    const at::Tensor& state_steps
) {
    TORCH_CHECK(task.device().is_cpu(), "The native LCA scan is CPU-only.");
    TORCH_CHECK(task.dim() == 2 && task.size(1) == 2, "task must have shape [trials, 2].");
    const int64_t trials = task.size(0);
    TORCH_CHECK(gain.sizes() == at::IntArrayRef({trials}), "gain must have shape [trials].");
    TORCH_CHECK(csi_duration.sizes() == at::IntArrayRef({trials}), "csi_duration must have shape [trials].");
    TORCH_CHECK(state_duration.sizes() == at::IntArrayRef({trials}), "state_duration must have shape [trials].");
    TORCH_CHECK(csi_steps.sizes() == at::IntArrayRef({trials}), "csi_steps must have shape [trials].");
    TORCH_CHECK(state_steps.sizes() == at::IntArrayRef({trials}), "state_steps must have shape [trials].");
    TORCH_CHECK(task.is_contiguous(), "task must be contiguous.");
    TORCH_CHECK(gain.is_contiguous(), "gain must be contiguous.");
    TORCH_CHECK(csi_duration.is_contiguous(), "csi_duration must be contiguous.");
    TORCH_CHECK(state_duration.is_contiguous(), "state_duration must be contiguous.");
    TORCH_CHECK(csi_steps.is_contiguous(), "csi_steps must be contiguous.");
    TORCH_CHECK(state_steps.is_contiguous(), "state_steps must be contiguous.");
    TORCH_CHECK(task.scalar_type() == gain.scalar_type(), "floating inputs must share a dtype.");
    TORCH_CHECK(task.scalar_type() == csi_duration.scalar_type(), "floating inputs must share a dtype.");
    TORCH_CHECK(task.scalar_type() == state_duration.scalar_type(), "floating inputs must share a dtype.");
    TORCH_CHECK(csi_steps.scalar_type() == at::kLong, "csi_steps must use int64.");
    TORCH_CHECK(state_steps.scalar_type() == at::kLong, "state_steps must use int64.");
}

template <typename scalar_t>
std::vector<at::Tensor> forward_impl(
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& csi_duration,
    const at::Tensor& state_duration,
    const at::Tensor& csi_steps,
    const at::Tensor& state_steps,
    double iti_duration,
    int64_t iti_steps,
    double leak,
    double competition
) {
    const int64_t trials = task.size(0);
    const auto* csi_count = csi_steps.data_ptr<int64_t>();
    const auto* state_count = state_steps.data_ptr<int64_t>();
    int64_t total_steps = trials * iti_steps;
    for (int64_t trial = 0; trial < trials; ++trial) {
        TORCH_CHECK(csi_count[trial] >= 0, "csi_steps cannot be negative.");
        TORCH_CHECK(state_count[trial] >= 0, "state_steps cannot be negative.");
        total_steps += csi_count[trial] + state_count[trial];
    }

    at::Tensor history = at::empty({total_steps + 1, 2}, task.options());
    at::Tensor onset = at::empty({trials, 2}, task.options());
    at::Tensor after = at::empty({trials, 2}, task.options());
    const auto task_values = task.accessor<scalar_t, 2>();
    const auto gain_values = gain.accessor<scalar_t, 1>();
    const auto csi_values = csi_duration.accessor<scalar_t, 1>();
    const auto state_values = state_duration.accessor<scalar_t, 1>();
    auto history_values = history.accessor<scalar_t, 2>();
    auto onset_values = onset.accessor<scalar_t, 2>();
    auto after_values = after.accessor<scalar_t, 2>();

    State<scalar_t> state{scalar_t(0), scalar_t(0)};
    const State<scalar_t> zero_task{scalar_t(0), scalar_t(0)};
    history_values[0][0] = state.first;
    history_values[0][1] = state.second;
    int64_t history_index = 0;

    const auto integrate = [&] (
        State<scalar_t> local_task,
        scalar_t local_gain,
        scalar_t duration,
        int64_t steps
    ) {
        if (steps == 0) {
            return;
        }
        const scalar_t step = duration / static_cast<scalar_t>(steps);
        for (int64_t step_index = 0; step_index < steps; ++step_index) {
            state = rk4_step(
                state,
                local_task,
                local_gain,
                step,
                static_cast<scalar_t>(leak),
                static_cast<scalar_t>(competition)
            );
            ++history_index;
            history_values[history_index][0] = state.first;
            history_values[history_index][1] = state.second;
        }
    };

    for (int64_t trial = 0; trial < trials; ++trial) {
        const State<scalar_t> active_task{
            task_values[trial][0], task_values[trial][1]
        };
        const scalar_t local_gain = gain_values[trial];
        integrate(
            zero_task,
            local_gain,
            static_cast<scalar_t>(iti_duration),
            iti_steps
        );
        integrate(
            active_task,
            local_gain,
            csi_values[trial],
            csi_count[trial]
        );
        onset_values[trial][0] = state.first;
        onset_values[trial][1] = state.second;
        integrate(
            active_task,
            local_gain,
            state_values[trial],
            state_count[trial]
        );
        after_values[trial][0] = state.first;
        after_values[trial][1] = state.second;
    }
    TORCH_CHECK(history_index == total_steps, "Internal LCA history size mismatch.");
    return {onset, after, history};
}

template <typename scalar_t>
std::vector<at::Tensor> backward_impl(
    const at::Tensor& history,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& csi_duration,
    const at::Tensor& state_duration,
    const at::Tensor& csi_steps,
    const at::Tensor& state_steps,
    const at::Tensor& gradient_onset,
    const at::Tensor& gradient_after,
    double iti_duration,
    int64_t iti_steps,
    double leak,
    double competition
) {
    const int64_t trials = task.size(0);
    at::Tensor gradient_task = at::zeros_like(task);
    at::Tensor gradient_gain = at::zeros_like(gain);
    at::Tensor gradient_csi = at::zeros_like(csi_duration);
    at::Tensor gradient_duration = at::zeros_like(state_duration);

    const auto history_values = history.accessor<scalar_t, 2>();
    const auto task_values = task.accessor<scalar_t, 2>();
    const auto gain_values = gain.accessor<scalar_t, 1>();
    const auto csi_values = csi_duration.accessor<scalar_t, 1>();
    const auto state_values = state_duration.accessor<scalar_t, 1>();
    const auto csi_count = csi_steps.accessor<int64_t, 1>();
    const auto state_count = state_steps.accessor<int64_t, 1>();
    const auto onset_values = gradient_onset.accessor<scalar_t, 2>();
    const auto after_values = gradient_after.accessor<scalar_t, 2>();
    auto task_gradients = gradient_task.accessor<scalar_t, 2>();
    auto gain_gradients = gradient_gain.accessor<scalar_t, 1>();
    auto csi_gradients = gradient_csi.accessor<scalar_t, 1>();
    auto duration_gradients = gradient_duration.accessor<scalar_t, 1>();

    int64_t history_index = history.size(0) - 1;
    State<scalar_t> gradient_state{scalar_t(0), scalar_t(0)};

    for (int64_t trial = trials - 1; trial >= 0; --trial) {
        const State<scalar_t> active_task{
            task_values[trial][0], task_values[trial][1]
        };
        const scalar_t local_gain = gain_values[trial];
        gradient_state.first += after_values[trial][0];
        gradient_state.second += after_values[trial][1];

        const auto reverse_interval = [&] (
            State<scalar_t> local_task,
            scalar_t duration,
            int64_t steps,
            bool accumulate_task,
            scalar_t* gradient_interval
        ) {
            if (steps == 0) {
                return;
            }
            const scalar_t step = duration / static_cast<scalar_t>(steps);
            scalar_t step_gradient = scalar_t(0);
            for (int64_t step_index = steps - 1; step_index >= 0; --step_index) {
                const State<scalar_t> input_state{
                    history_values[history_index - 1][0],
                    history_values[history_index - 1][1],
                };
                const StepGradient<scalar_t> gradients = rk4_step_vjp(
                    input_state,
                    local_task,
                    local_gain,
                    step,
                    static_cast<scalar_t>(leak),
                    static_cast<scalar_t>(competition),
                    gradient_state
                );
                gradient_state = gradients.state;
                if (accumulate_task) {
                    task_gradients[trial][0] += gradients.task.first;
                    task_gradients[trial][1] += gradients.task.second;
                }
                gain_gradients[trial] += gradients.gain;
                step_gradient += gradients.step;
                --history_index;
            }
            if (gradient_interval != nullptr) {
                *gradient_interval += step_gradient / static_cast<scalar_t>(steps);
            }
        };

        reverse_interval(
            active_task,
            state_values[trial],
            state_count[trial],
            true,
            &duration_gradients[trial]
        );
        gradient_state.first += onset_values[trial][0];
        gradient_state.second += onset_values[trial][1];
        reverse_interval(
            active_task,
            csi_values[trial],
            csi_count[trial],
            true,
            &csi_gradients[trial]
        );
        reverse_interval(
            {scalar_t(0), scalar_t(0)},
            static_cast<scalar_t>(iti_duration),
            iti_steps,
            false,
            nullptr
        );
    }
    TORCH_CHECK(history_index == 0, "Internal LCA reverse-history mismatch.");
    return {
        gradient_task, gradient_gain, gradient_csi, gradient_duration
    };
}

template <typename scalar_t>
std::vector<at::Tensor> drift_forward_impl(
    const at::Tensor& initial_state,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    const int64_t batch = initial_state.size(0);
    at::Tensor drift = at::empty({batch, steps}, initial_state.options());
    at::Tensor final_state = at::empty_like(initial_state);
    at::Tensor history = at::empty(
        {steps + 1, batch, 2}, initial_state.options()
    );
    const auto initial_values = initial_state.accessor<scalar_t, 2>();
    const auto task_values = task.accessor<scalar_t, 2>();
    const auto gain_values = gain.accessor<scalar_t, 1>();
    const auto stimulus_values = stimulus.accessor<scalar_t, 2>();
    const auto response_values = correct_response.accessor<scalar_t, 1>();
    auto drift_values = drift.accessor<scalar_t, 2>();
    auto final_values = final_state.accessor<scalar_t, 2>();
    auto history_values = history.accessor<scalar_t, 3>();
    const scalar_t half_step = static_cast<scalar_t>(step_size / 2.0);

    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
            State<scalar_t> state{
                initial_values[lane][0], initial_values[lane][1]
            };
            const State<scalar_t> local_task{
                task_values[lane][0], task_values[lane][1]
            };
            const scalar_t local_gain = gain_values[lane];
            scalar_t local_stimulus[4];
            for (int64_t index = 0; index < 4; ++index) {
                local_stimulus[index] = stimulus_values[lane][index];
            }
            history_values[0][lane][0] = state.first;
            history_values[0][lane][1] = state.second;
            for (int64_t step_index = 0; step_index < steps; ++step_index) {
                const State<scalar_t> midpoint = rk4_step(
                    state,
                    local_task,
                    local_gain,
                    half_step,
                    static_cast<scalar_t>(leak),
                    static_cast<scalar_t>(competition)
                );
                drift_values[lane][step_index] = drift_value(
                    local_stimulus,
                    midpoint,
                    local_gain,
                    response_values[lane]
                );
                state = rk4_step(
                    midpoint,
                    local_task,
                    local_gain,
                    half_step,
                    static_cast<scalar_t>(leak),
                    static_cast<scalar_t>(competition)
                );
                history_values[step_index + 1][lane][0] = state.first;
                history_values[step_index + 1][lane][1] = state.second;
            }
            final_values[lane][0] = state.first;
            final_values[lane][1] = state.second;
    }
    return {drift, final_state, history};
}

template <typename scalar_t>
at::Tensor integrate_euler_impl(
    const at::Tensor& initial_state,
    const at::Tensor& task,
    const at::Tensor& gain,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    const int64_t batch = initial_state.size(0);
    at::Tensor final_state = at::empty_like(initial_state);
    const auto initial_values = initial_state.accessor<scalar_t, 2>();
    const auto task_values = task.accessor<scalar_t, 2>();
    const auto gain_values = gain.accessor<scalar_t, 1>();
    auto final_values = final_state.accessor<scalar_t, 2>();

    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
        State<scalar_t> state{
            initial_values[lane][0], initial_values[lane][1]
        };
        const State<scalar_t> local_task{
            task_values[lane][0], task_values[lane][1]
        };
        const scalar_t local_gain = gain_values[lane];
        for (int64_t step_index = 0; step_index < steps; ++step_index) {
            state = euler_step(
                state,
                local_task,
                local_gain,
                static_cast<scalar_t>(step_size),
                static_cast<scalar_t>(leak),
                static_cast<scalar_t>(competition)
            );
        }
        final_values[lane][0] = state.first;
        final_values[lane][1] = state.second;
    }
    return final_state;
}

template <typename scalar_t>
std::vector<at::Tensor> drift_forward_euler_impl(
    const at::Tensor& initial_state,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    const int64_t batch = initial_state.size(0);
    at::Tensor drift = at::empty({batch, steps}, initial_state.options());
    at::Tensor final_state = at::empty_like(initial_state);
    const auto initial_values = initial_state.accessor<scalar_t, 2>();
    const auto task_values = task.accessor<scalar_t, 2>();
    const auto gain_values = gain.accessor<scalar_t, 1>();
    const auto stimulus_values = stimulus.accessor<scalar_t, 2>();
    const auto response_values = correct_response.accessor<scalar_t, 1>();
    auto drift_values = drift.accessor<scalar_t, 2>();
    auto final_values = final_state.accessor<scalar_t, 2>();

    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
        State<scalar_t> state{
            initial_values[lane][0], initial_values[lane][1]
        };
        const State<scalar_t> local_task{
            task_values[lane][0], task_values[lane][1]
        };
        const scalar_t local_gain = gain_values[lane];
        scalar_t local_stimulus[4];
        for (int64_t index = 0; index < 4; ++index) {
            local_stimulus[index] = stimulus_values[lane][index];
        }
        for (int64_t step_index = 0; step_index < steps; ++step_index) {
            // PNL advances the LCA before evaluating the logistic output that
            // supplies this scheduler pass's DDM drift.
            state = euler_step(
                state,
                local_task,
                local_gain,
                static_cast<scalar_t>(step_size),
                static_cast<scalar_t>(leak),
                static_cast<scalar_t>(competition)
            );
            drift_values[lane][step_index] = drift_value(
                local_stimulus,
                state,
                local_gain,
                response_values[lane]
            );
        }
        final_values[lane][0] = state.first;
        final_values[lane][1] = state.second;
    }
    return {drift, final_state};
}

template <typename scalar_t>
std::vector<at::Tensor> drift_backward_impl(
    const at::Tensor& history,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response,
    const at::Tensor& gradient_drift,
    const at::Tensor& gradient_final_state,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    const int64_t batch = task.size(0);
    at::Tensor gradient_initial = at::empty({batch, 2}, task.options());
    at::Tensor gradient_task = at::zeros_like(task);
    at::Tensor gradient_gain = at::zeros_like(gain);
    at::Tensor gradient_stimulus = at::zeros_like(stimulus);
    at::Tensor gradient_response = at::zeros_like(correct_response);
    const auto history_values = history.accessor<scalar_t, 3>();
    const auto task_values = task.accessor<scalar_t, 2>();
    const auto gain_values = gain.accessor<scalar_t, 1>();
    const auto stimulus_values = stimulus.accessor<scalar_t, 2>();
    const auto response_values = correct_response.accessor<scalar_t, 1>();
    const auto drift_gradients = gradient_drift.accessor<scalar_t, 2>();
    const auto final_gradients = gradient_final_state.accessor<scalar_t, 2>();
    auto initial_gradients = gradient_initial.accessor<scalar_t, 2>();
    auto task_gradients = gradient_task.accessor<scalar_t, 2>();
    auto gain_gradients = gradient_gain.accessor<scalar_t, 1>();
    auto stimulus_gradients = gradient_stimulus.accessor<scalar_t, 2>();
    auto response_gradients = gradient_response.accessor<scalar_t, 1>();
    const scalar_t half_step = static_cast<scalar_t>(step_size / 2.0);

    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
            const State<scalar_t> local_task{
                task_values[lane][0], task_values[lane][1]
            };
            const scalar_t local_gain = gain_values[lane];
            scalar_t local_stimulus[4];
            for (int64_t index = 0; index < 4; ++index) {
                local_stimulus[index] = stimulus_values[lane][index];
            }
            State<scalar_t> gradient_state{
                final_gradients[lane][0], final_gradients[lane][1]
            };
            State<scalar_t> local_task_gradient{scalar_t(0), scalar_t(0)};
            scalar_t local_gain_gradient = scalar_t(0);
            scalar_t local_stimulus_gradient[4]{
                scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)
            };
            scalar_t local_response_gradient = scalar_t(0);

            for (int64_t step_index = steps - 1; step_index >= 0; --step_index) {
                const State<scalar_t> input_state{
                    history_values[step_index][lane][0],
                    history_values[step_index][lane][1]
                };
                const State<scalar_t> midpoint = rk4_step(
                    input_state,
                    local_task,
                    local_gain,
                    half_step,
                    static_cast<scalar_t>(leak),
                    static_cast<scalar_t>(competition)
                );
                const StepGradient<scalar_t> second = rk4_step_vjp(
                    midpoint,
                    local_task,
                    local_gain,
                    half_step,
                    static_cast<scalar_t>(leak),
                    static_cast<scalar_t>(competition),
                    gradient_state
                );
                const DriftGradient<scalar_t> drift = drift_vjp(
                    local_stimulus,
                    midpoint,
                    local_gain,
                    response_values[lane],
                    drift_gradients[lane][step_index]
                );
                const State<scalar_t> gradient_midpoint = add(
                    second.state, drift.midpoint
                );
                const StepGradient<scalar_t> first = rk4_step_vjp(
                    input_state,
                    local_task,
                    local_gain,
                    half_step,
                    static_cast<scalar_t>(leak),
                    static_cast<scalar_t>(competition),
                    gradient_midpoint
                );
                gradient_state = first.state;
                local_task_gradient = add(
                    local_task_gradient, add(first.task, second.task)
                );
                local_gain_gradient += first.gain + second.gain + drift.gain;
                for (int64_t index = 0; index < 4; ++index) {
                    local_stimulus_gradient[index] += drift.stimulus[index];
                }
                local_response_gradient += drift.correct_response;
            }
            initial_gradients[lane][0] = gradient_state.first;
            initial_gradients[lane][1] = gradient_state.second;
            task_gradients[lane][0] = local_task_gradient.first;
            task_gradients[lane][1] = local_task_gradient.second;
            gain_gradients[lane] = local_gain_gradient;
            for (int64_t index = 0; index < 4; ++index) {
                stimulus_gradients[lane][index]
                    = local_stimulus_gradient[index];
            }
            response_gradients[lane] = local_response_gradient;
    }
    return {
        gradient_initial,
        gradient_task,
        gradient_gain,
        gradient_stimulus,
        gradient_response,
    };
}

}  // namespace

std::vector<at::Tensor> lca_subject_forward(
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& csi_duration,
    const at::Tensor& state_duration,
    const at::Tensor& csi_steps,
    const at::Tensor& state_steps,
    double iti_duration,
    int64_t iti_steps,
    double leak,
    double competition
) {
    check_inputs(
        task, gain, csi_duration, state_duration, csi_steps, state_steps
    );
    TORCH_CHECK(iti_steps >= 0, "iti_steps cannot be negative.");
    std::vector<at::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(task.scalar_type(), "lca_subject_forward", [&] {
        result = forward_impl<scalar_t>(
            task,
            gain,
            csi_duration,
            state_duration,
            csi_steps,
            state_steps,
            iti_duration,
            iti_steps,
            leak,
            competition
        );
    });
    return result;
}

std::vector<at::Tensor> lca_subject_backward(
    const at::Tensor& history,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& csi_duration,
    const at::Tensor& state_duration,
    const at::Tensor& csi_steps,
    const at::Tensor& state_steps,
    const at::Tensor& gradient_onset,
    const at::Tensor& gradient_after,
    double iti_duration,
    int64_t iti_steps,
    double leak,
    double competition
) {
    check_inputs(
        task, gain, csi_duration, state_duration, csi_steps, state_steps
    );
    TORCH_CHECK(history.device().is_cpu(), "history must be on the CPU.");
    TORCH_CHECK(gradient_onset.is_contiguous(), "gradient_onset must be contiguous.");
    TORCH_CHECK(gradient_after.is_contiguous(), "gradient_after must be contiguous.");
    std::vector<at::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(task.scalar_type(), "lca_subject_backward", [&] {
        result = backward_impl<scalar_t>(
            history,
            task,
            gain,
            csi_duration,
            state_duration,
            csi_steps,
            state_steps,
            gradient_onset,
            gradient_after,
            iti_duration,
            iti_steps,
            leak,
            competition
        );
    });
    return result;
}

void check_lca_inputs(
    const at::Tensor& state,
    const at::Tensor& task,
    const at::Tensor& gain
) {
    TORCH_CHECK(state.device().is_cpu(), "The native LCA drift scan is CPU-only.");
    TORCH_CHECK(state.dim() == 2 && state.size(1) == 2, "state must have shape [batch, 2].");
    const int64_t batch = state.size(0);
    TORCH_CHECK(task.sizes() == at::IntArrayRef({batch, 2}), "task must have shape [batch, 2].");
    TORCH_CHECK(gain.sizes() == at::IntArrayRef({batch}), "gain must have shape [batch].");
    for (const at::Tensor& value : {state, task, gain}) {
        TORCH_CHECK(value.device().is_cpu(), "All native LCA inputs must be on the CPU.");
        TORCH_CHECK(value.is_contiguous(), "All native LCA inputs must be contiguous.");
        TORCH_CHECK(value.scalar_type() == state.scalar_type(), "All native LCA inputs must share a dtype.");
    }
}

void check_drift_inputs(
    const at::Tensor& state,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response
) {
    check_lca_inputs(state, task, gain);
    const int64_t batch = state.size(0);
    TORCH_CHECK(stimulus.sizes() == at::IntArrayRef({batch, 4}), "stimulus must have shape [batch, 4].");
    TORCH_CHECK(correct_response.sizes() == at::IntArrayRef({batch}), "correct_response must have shape [batch].");
    for (const at::Tensor& value : {stimulus, correct_response}) {
        TORCH_CHECK(value.device().is_cpu(), "All native LCA drift inputs must be on the CPU.");
        TORCH_CHECK(value.is_contiguous(), "All native LCA drift inputs must be contiguous.");
        TORCH_CHECK(value.scalar_type() == state.scalar_type(), "All native LCA drift inputs must share a dtype.");
    }
}

at::Tensor lca_integrate_euler(
    const at::Tensor& state,
    const at::Tensor& task,
    const at::Tensor& gain,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    check_lca_inputs(state, task, gain);
    TORCH_CHECK(steps >= 0, "Euler integration steps cannot be negative.");
    at::Tensor result;
    AT_DISPATCH_FLOATING_TYPES(state.scalar_type(), "lca_integrate_euler", [&] {
        result = integrate_euler_impl<scalar_t>(
            state,
            task,
            gain,
            steps,
            step_size,
            leak,
            competition
        );
    });
    return result;
}

std::vector<at::Tensor> lca_drift_forward(
    const at::Tensor& state,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    check_drift_inputs(state, task, gain, stimulus, correct_response);
    TORCH_CHECK(steps > 0, "The LCA drift path must contain at least one step.");
    std::vector<at::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(state.scalar_type(), "lca_drift_forward", [&] {
        result = drift_forward_impl<scalar_t>(
            state,
            task,
            gain,
            stimulus,
            correct_response,
            steps,
            step_size,
            leak,
            competition
        );
    });
    return result;
}

std::vector<at::Tensor> lca_drift_forward_euler(
    const at::Tensor& state,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    check_drift_inputs(state, task, gain, stimulus, correct_response);
    TORCH_CHECK(steps > 0, "The Euler LCA drift path must contain at least one step.");
    std::vector<at::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(state.scalar_type(), "lca_drift_forward_euler", [&] {
        result = drift_forward_euler_impl<scalar_t>(
            state,
            task,
            gain,
            stimulus,
            correct_response,
            steps,
            step_size,
            leak,
            competition
        );
    });
    return result;
}

std::vector<at::Tensor> lca_drift_backward(
    const at::Tensor& history,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response,
    const at::Tensor& gradient_drift,
    const at::Tensor& gradient_final_state,
    int64_t steps,
    double step_size,
    double leak,
    double competition
) {
    check_drift_inputs(
        history[0], task, gain, stimulus, correct_response
    );
    TORCH_CHECK(history.is_contiguous(), "history must be contiguous.");
    TORCH_CHECK(gradient_drift.is_contiguous(), "gradient_drift must be contiguous.");
    TORCH_CHECK(gradient_final_state.is_contiguous(), "gradient_final_state must be contiguous.");
    std::vector<at::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(task.scalar_type(), "lca_drift_backward", [&] {
        result = drift_backward_impl<scalar_t>(
            history,
            task,
            gain,
            stimulus,
            correct_response,
            gradient_drift,
            gradient_final_state,
            steps,
            step_size,
            leak,
            competition
        );
    });
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &lca_subject_forward, "Fused CSI LCA subject scan");
    module.def("backward", &lca_subject_backward, "Fused CSI LCA subject adjoint");
    module.def("drift_forward", &lca_drift_forward, "Fused CSI LCA drift paths");
    module.def("drift_backward", &lca_drift_backward, "Fused CSI LCA drift adjoint");
    module.def("euler_integrate", &lca_integrate_euler, "Forward-only Euler LCA integration");
    module.def("euler_drift_forward", &lca_drift_forward_euler, "Forward-only Euler CSI LCA drift paths");
}
