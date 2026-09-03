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

template <typename scalar_t>
inline scalar_t chang_cooper_delta(scalar_t peclet) {
    if (std::abs(peclet) < scalar_t(1.0e-4)) {
        return scalar_t(0.5) - peclet / scalar_t(12)
            + peclet * peclet * peclet / scalar_t(720);
    }
    return scalar_t(1) / peclet - scalar_t(1) / std::expm1(peclet);
}

template <typename scalar_t>
struct Dual3 {
    scalar_t value;
    scalar_t drift;
    scalar_t threshold;
    scalar_t collapse;
};

template <typename scalar_t>
inline Dual3<scalar_t> operator+(
    Dual3<scalar_t> lhs, Dual3<scalar_t> rhs
) {
    return {
        lhs.value + rhs.value,
        lhs.drift + rhs.drift,
        lhs.threshold + rhs.threshold,
        lhs.collapse + rhs.collapse,
    };
}

template <typename scalar_t>
inline Dual3<scalar_t> operator-(
    Dual3<scalar_t> lhs, Dual3<scalar_t> rhs
) {
    return {
        lhs.value - rhs.value,
        lhs.drift - rhs.drift,
        lhs.threshold - rhs.threshold,
        lhs.collapse - rhs.collapse,
    };
}

template <typename scalar_t>
inline Dual3<scalar_t> operator-(Dual3<scalar_t> value) {
    return {
        -value.value, -value.drift, -value.threshold, -value.collapse
    };
}

template <typename scalar_t>
inline Dual3<scalar_t> operator*(
    Dual3<scalar_t> lhs, Dual3<scalar_t> rhs
) {
    return {
        lhs.value * rhs.value,
        lhs.drift * rhs.value + lhs.value * rhs.drift,
        lhs.threshold * rhs.value + lhs.value * rhs.threshold,
        lhs.collapse * rhs.value + lhs.value * rhs.collapse,
    };
}

template <typename scalar_t>
inline Dual3<scalar_t> operator/(
    Dual3<scalar_t> lhs, Dual3<scalar_t> rhs
) {
    const scalar_t inverse = scalar_t(1) / rhs.value;
    const scalar_t inverse_squared = inverse * inverse;
    return {
        lhs.value * inverse,
        (lhs.drift * rhs.value - lhs.value * rhs.drift) * inverse_squared,
        (lhs.threshold * rhs.value - lhs.value * rhs.threshold)
            * inverse_squared,
        (lhs.collapse * rhs.value - lhs.value * rhs.collapse)
            * inverse_squared,
    };
}

template <typename scalar_t>
inline Dual3<scalar_t> dual_constant(scalar_t value) {
    return {value, scalar_t(0), scalar_t(0), scalar_t(0)};
}

template <typename scalar_t>
inline Dual3<scalar_t> dual_expm1(Dual3<scalar_t> input) {
    const scalar_t value = std::expm1(input.value);
    const scalar_t slope = value + scalar_t(1);
    return {
        value,
        slope * input.drift,
        slope * input.threshold,
        slope * input.collapse,
    };
}

template <typename scalar_t>
inline Dual3<scalar_t> chang_cooper_delta(Dual3<scalar_t> peclet) {
    if (std::abs(peclet.value) < scalar_t(1.0e-4)) {
        const Dual3<scalar_t> twelve = dual_constant<scalar_t>(scalar_t(12));
        const Dual3<scalar_t> seven_twenty = dual_constant<scalar_t>(
            scalar_t(720)
        );
        return dual_constant<scalar_t>(scalar_t(0.5)) - peclet / twelve
            + peclet * peclet * peclet / seven_twenty;
    }
    const Dual3<scalar_t> one = dual_constant<scalar_t>(scalar_t(1));
    return one / peclet - one / dual_expm1(peclet);
}

template <typename scalar_t>
std::vector<at::Tensor> ddm_forward_impl(
    const at::Tensor& drift,
    const at::Tensor& threshold,
    const at::Tensor& collapse_rate,
    const at::Tensor& interval_low,
    const at::Tensor& interval_high,
    const at::Tensor& choice,
    double time_step,
    int64_t spatial_points,
    double noise,
    double boundary_floor,
    int64_t rannacher_steps,
    bool store_history
) {
    const int64_t batch = drift.size(0);
    const int64_t time_steps = drift.size(1);
    const int64_t interior = spatial_points - 2;
    const scalar_t dt = static_cast<scalar_t>(time_step);
    const scalar_t dy = scalar_t(2) / static_cast<scalar_t>(spatial_points - 1);
    const scalar_t half_noise_squared = static_cast<scalar_t>(
        noise * noise / 2.0
    );

    at::Tensor probability = at::zeros({batch}, drift.options());
    at::Tensor upper_probability = at::zeros_like(probability);
    at::Tensor lower_probability = at::zeros_like(probability);
    at::Tensor survival = at::zeros_like(probability);
    at::Tensor mass_error = at::zeros_like(probability);
    at::Tensor minimum_density = at::zeros_like(probability);
    at::Tensor invalid = at::zeros(
        {batch}, drift.options().dtype(at::kBool)
    );
    at::Tensor history = store_history
        ? at::empty({time_steps + 1, batch, interior}, drift.options())
        : at::empty({0}, drift.options());

    const auto drift_values = drift.accessor<scalar_t, 2>();
    const auto threshold_values = threshold.accessor<scalar_t, 1>();
    const auto collapse_values = collapse_rate.accessor<scalar_t, 1>();
    const auto low_values = interval_low.accessor<scalar_t, 1>();
    const auto high_values = interval_high.accessor<scalar_t, 1>();
    const auto choice_values = choice.accessor<scalar_t, 1>();
    auto probability_values = probability.accessor<scalar_t, 1>();
    auto upper_probability_values = upper_probability.accessor<scalar_t, 1>();
    auto lower_probability_values = lower_probability.accessor<scalar_t, 1>();
    auto survival_values = survival.accessor<scalar_t, 1>();
    auto error_values = mass_error.accessor<scalar_t, 1>();
    auto minimum_values = minimum_density.accessor<scalar_t, 1>();
    auto invalid_values = invalid.accessor<bool, 1>();
    scalar_t* history_values = store_history
        ? history.data_ptr<scalar_t>()
        : nullptr;

    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
            const scalar_t local_threshold = threshold_values[lane];
            const scalar_t local_collapse = collapse_values[lane];
            const scalar_t local_low = low_values[lane];
            const scalar_t local_high = high_values[lane];
            bool local_invalid = local_high <= scalar_t(0)
                || local_high <= local_low;
            for (int64_t step_index = 0;
                 step_index < time_steps && !local_invalid;
                 ++step_index) {
                const scalar_t t0 = static_cast<scalar_t>(step_index) * dt;
                if (t0 >= local_high) {
                    break;
                }
                const scalar_t midpoint_time = t0 + scalar_t(0.5) * dt;
                if (local_threshold + local_collapse * midpoint_time
                    <= static_cast<scalar_t>(boundary_floor)) {
                    local_invalid = true;
                }
            }
            invalid_values[lane] = local_invalid;

            std::vector<scalar_t> density(interior, scalar_t(0));
            std::vector<scalar_t> next_density(interior, scalar_t(0));
            std::vector<scalar_t> lower(interior);
            std::vector<scalar_t> diagonal(interior);
            std::vector<scalar_t> upper(interior);
            std::vector<scalar_t> right_hand_side(interior);
            std::vector<scalar_t> c_prime(interior);
            std::vector<scalar_t> d_prime(interior);
            std::vector<scalar_t> face_velocity(interior + 1);
            std::vector<scalar_t> delta(interior + 1);
            density[interior / 2] = scalar_t(1) / dy;
            if (store_history) {
                for (int64_t index = 0; index < interior; ++index) {
                    history_values[(lane * interior) + index] = density[index];
                }
            }

            scalar_t local_observed = scalar_t(0);
            scalar_t local_upper_probability = scalar_t(0);
            scalar_t local_lower_probability = scalar_t(0);
            scalar_t local_minimum_density = scalar_t(0);

            for (int64_t step_index = 0; step_index < time_steps; ++step_index) {
                const scalar_t t0 = static_cast<scalar_t>(step_index) * dt;
                const scalar_t t1 = t0 + dt;
                const bool active = t0 < local_high && !local_invalid;
                if (active) {
                    const scalar_t midpoint_time = t0 + scalar_t(0.5) * dt;
                    const scalar_t boundary = local_threshold
                        + local_collapse * midpoint_time;
                    const scalar_t diffusion = half_noise_squared
                        / (boundary * boundary);
                    const scalar_t diffusion_over_dy = diffusion / dy;
                    for (int64_t face = 0; face <= interior; ++face) {
                        const scalar_t face_location = -scalar_t(1)
                            + (static_cast<scalar_t>(face) + scalar_t(0.5)) * dy;
                        face_velocity[face] = (
                            drift_values[lane][step_index]
                            - local_collapse * face_location
                        ) / boundary;
                        delta[face] = chang_cooper_delta(
                            face_velocity[face] * dy / diffusion
                        );
                    }
                    for (int64_t index = 0; index < interior; ++index) {
                        const scalar_t raw_lower = (
                            face_velocity[index]
                                * (scalar_t(1) - delta[index])
                            + diffusion_over_dy
                        ) / dy;
                        const scalar_t raw_diagonal = (
                            -(
                                face_velocity[index + 1]
                                    * (scalar_t(1) - delta[index + 1])
                                + diffusion_over_dy
                            )
                            + face_velocity[index] * delta[index]
                            - diffusion_over_dy
                        ) / dy;
                        const scalar_t raw_upper = (
                            -face_velocity[index + 1] * delta[index + 1]
                            + diffusion_over_dy
                        ) / dy;
                        lower[index] = index == 0 ? scalar_t(0) : raw_lower;
                        diagonal[index] = raw_diagonal;
                        upper[index] = index == interior - 1
                            ? scalar_t(0)
                            : raw_upper;
                    }

                    const scalar_t theta = step_index < rannacher_steps
                        ? scalar_t(1)
                        : scalar_t(0.5);
                    for (int64_t index = 0; index < interior; ++index) {
                        const scalar_t left = index == 0
                            ? scalar_t(0)
                            : density[index - 1];
                        const scalar_t right = index == interior - 1
                            ? scalar_t(0)
                            : density[index + 1];
                        right_hand_side[index] = density[index]
                            + (scalar_t(1) - theta) * dt
                                * (
                                    lower[index] * left
                                    + diagonal[index] * density[index]
                                    + upper[index] * right
                                );
                        lower[index] = -theta * dt * lower[index];
                        diagonal[index] = scalar_t(1)
                            - theta * dt * diagonal[index];
                        upper[index] = -theta * dt * upper[index];
                    }

                    c_prime[0] = upper[0] / diagonal[0];
                    d_prime[0] = right_hand_side[0] / diagonal[0];
                    for (int64_t index = 1; index < interior; ++index) {
                        const scalar_t denominator = diagonal[index]
                            - lower[index] * c_prime[index - 1];
                        c_prime[index] = index == interior - 1
                            ? scalar_t(0)
                            : upper[index] / denominator;
                        d_prime[index] = (
                            right_hand_side[index]
                            - lower[index] * d_prime[index - 1]
                        ) / denominator;
                    }
                    next_density[interior - 1] = d_prime[interior - 1];
                    for (int64_t index = interior - 2; index >= 0; --index) {
                        next_density[index] = d_prime[index]
                            - c_prime[index] * next_density[index + 1];
                    }

                    scalar_t step_minimum = next_density[0];
                    for (int64_t index = 1; index < interior; ++index) {
                        step_minimum = std::min(
                            step_minimum, next_density[index]
                        );
                    }
                    local_minimum_density = std::min(
                        local_minimum_density, step_minimum
                    );
                    const scalar_t midpoint_left = scalar_t(0.5)
                        * (density[0] + next_density[0]);
                    const scalar_t midpoint_right = scalar_t(0.5)
                        * (density[interior - 1]
                           + next_density[interior - 1]);
                    const scalar_t raw_lower_flux = (
                        -face_velocity[0] * delta[0]
                        + diffusion_over_dy
                    ) * midpoint_left;
                    const scalar_t raw_upper_flux = (
                        face_velocity[interior]
                            * (scalar_t(1) - delta[interior])
                        + diffusion_over_dy
                    ) * midpoint_right;
                    const scalar_t lower_flux = std::max(
                        raw_lower_flux, scalar_t(0)
                    );
                    const scalar_t upper_flux = std::max(
                        raw_upper_flux, scalar_t(0)
                    );
                    local_lower_probability += dt * lower_flux;
                    local_upper_probability += dt * upper_flux;
                    const scalar_t overlap_left = local_low > t0
                        ? local_low
                        : t0;
                    const scalar_t overlap_right = local_high < t1
                        ? local_high
                        : t1;
                    if (overlap_right > overlap_left) {
                        local_observed += (overlap_right - overlap_left)
                            * (choice_values[lane] > scalar_t(0.5)
                                ? upper_flux
                                : lower_flux);
                    }
                    density.swap(next_density);
                }
                if (store_history) {
                    const int64_t history_offset = (
                        (step_index + 1) * batch + lane
                    ) * interior;
                    for (int64_t index = 0; index < interior; ++index) {
                        history_values[history_offset + index] = density[index];
                    }
                }
            }

            scalar_t local_survival = scalar_t(0);
            for (scalar_t value : density) {
                local_survival += value * dy;
            }
            probability_values[lane] = local_invalid
                ? scalar_t(0)
                : local_observed;
            upper_probability_values[lane] = local_upper_probability;
            lower_probability_values[lane] = local_lower_probability;
            survival_values[lane] = local_survival;
            error_values[lane] = std::abs(
                local_survival + local_upper_probability
                + local_lower_probability - scalar_t(1)
            );
            minimum_values[lane] = local_minimum_density;
    }
    return {
        probability,
        upper_probability,
        lower_probability,
        survival,
        mass_error,
        minimum_density,
        invalid,
        history,
    };
}

template <typename scalar_t>
std::vector<at::Tensor> ddm_backward_impl(
    const at::Tensor& history,
    const at::Tensor& drift,
    const at::Tensor& threshold,
    const at::Tensor& collapse_rate,
    const at::Tensor& interval_low,
    const at::Tensor& interval_high,
    const at::Tensor& choice,
    const at::Tensor& invalid,
    const at::Tensor& gradient_probability,
    double time_step,
    int64_t spatial_points,
    double noise,
    int64_t rannacher_steps
) {
    const int64_t batch = drift.size(0);
    const int64_t time_steps = drift.size(1);
    const int64_t interior = spatial_points - 2;
    const scalar_t dt = static_cast<scalar_t>(time_step);
    const scalar_t dy = scalar_t(2) / static_cast<scalar_t>(spatial_points - 1);
    const scalar_t half_noise_squared = static_cast<scalar_t>(
        noise * noise / 2.0
    );
    at::Tensor gradient_drift = at::zeros_like(drift);
    at::Tensor gradient_threshold = at::zeros_like(threshold);
    at::Tensor gradient_collapse = at::zeros_like(collapse_rate);
    at::Tensor gradient_low = at::zeros_like(interval_low);
    at::Tensor gradient_high = at::zeros_like(interval_high);

    const auto history_values = history.accessor<scalar_t, 3>();
    const auto drift_values = drift.accessor<scalar_t, 2>();
    const auto threshold_values = threshold.accessor<scalar_t, 1>();
    const auto collapse_values = collapse_rate.accessor<scalar_t, 1>();
    const auto low_values = interval_low.accessor<scalar_t, 1>();
    const auto high_values = interval_high.accessor<scalar_t, 1>();
    const auto choice_values = choice.accessor<scalar_t, 1>();
    const auto invalid_values = invalid.accessor<bool, 1>();
    const auto probability_gradients = gradient_probability.accessor<scalar_t, 1>();
    auto drift_gradients = gradient_drift.accessor<scalar_t, 2>();
    auto threshold_gradients = gradient_threshold.accessor<scalar_t, 1>();
    auto collapse_gradients = gradient_collapse.accessor<scalar_t, 1>();
    auto low_gradients = gradient_low.accessor<scalar_t, 1>();
    auto high_gradients = gradient_high.accessor<scalar_t, 1>();

    #pragma omp parallel for schedule(static)
    for (int64_t lane = 0; lane < batch; ++lane) {
            if (invalid_values[lane]) {
                continue;
            }
            const scalar_t local_threshold = threshold_values[lane];
            const scalar_t local_collapse = collapse_values[lane];
            const scalar_t local_low = low_values[lane];
            const scalar_t local_high = high_values[lane];
            const scalar_t gradient_observed = probability_gradients[lane];
            std::vector<scalar_t> gradient_state(interior, scalar_t(0));
            std::vector<scalar_t> gradient_input(interior);
            std::vector<scalar_t> gradient_solution(interior);
            std::vector<scalar_t> lambda(interior);
            std::vector<scalar_t> lower_transpose(interior);
            std::vector<scalar_t> diagonal_transpose(interior);
            std::vector<scalar_t> upper_transpose(interior);
            std::vector<scalar_t> c_prime(interior);
            std::vector<scalar_t> d_prime(interior);
            std::vector<Dual3<scalar_t>> lower(interior);
            std::vector<Dual3<scalar_t>> diagonal(interior);
            std::vector<Dual3<scalar_t>> upper(interior);
            std::vector<Dual3<scalar_t>> face_velocity(interior + 1);
            std::vector<Dual3<scalar_t>> delta(interior + 1);
            scalar_t local_threshold_gradient = scalar_t(0);
            scalar_t local_collapse_gradient = scalar_t(0);
            scalar_t local_low_gradient = scalar_t(0);
            scalar_t local_high_gradient = scalar_t(0);

            const auto accumulate_parameter_gradient = [&] (
                scalar_t multiplier, Dual3<scalar_t> value,
                scalar_t* local_drift_gradient
            ) {
                *local_drift_gradient += multiplier * value.drift;
                local_threshold_gradient += multiplier * value.threshold;
                local_collapse_gradient += multiplier * value.collapse;
            };

            for (int64_t step_index = time_steps - 1;
                 step_index >= 0;
                 --step_index) {
                const scalar_t t0 = static_cast<scalar_t>(step_index) * dt;
                if (t0 >= local_high) {
                    continue;
                }
                const scalar_t t1 = t0 + dt;
                const scalar_t midpoint_time = t0 + scalar_t(0.5) * dt;
                const Dual3<scalar_t> local_drift{
                    drift_values[lane][step_index],
                    scalar_t(1),
                    scalar_t(0),
                    scalar_t(0),
                };
                const Dual3<scalar_t> local_rate{
                    local_collapse,
                    scalar_t(0),
                    scalar_t(0),
                    scalar_t(1),
                };
                const Dual3<scalar_t> boundary{
                    local_threshold + local_collapse * midpoint_time,
                    scalar_t(0),
                    scalar_t(1),
                    midpoint_time,
                };
                const Dual3<scalar_t> diffusion
                    = dual_constant<scalar_t>(half_noise_squared)
                    / (boundary * boundary);
                const Dual3<scalar_t> diffusion_over_dy
                    = diffusion / dual_constant<scalar_t>(dy);
                for (int64_t face = 0; face <= interior; ++face) {
                    const scalar_t face_location = -scalar_t(1)
                        + (static_cast<scalar_t>(face) + scalar_t(0.5)) * dy;
                    face_velocity[face] = (
                        local_drift
                        - local_rate
                            * dual_constant<scalar_t>(face_location)
                    ) / boundary;
                    const Dual3<scalar_t> peclet = face_velocity[face]
                        * dual_constant<scalar_t>(dy) / diffusion;
                    delta[face] = chang_cooper_delta(peclet);
                }
                for (int64_t index = 0; index < interior; ++index) {
                    const Dual3<scalar_t> raw_lower = (
                        face_velocity[index]
                            * (dual_constant<scalar_t>(scalar_t(1))
                               - delta[index])
                        + diffusion_over_dy
                    ) / dual_constant<scalar_t>(dy);
                    const Dual3<scalar_t> raw_diagonal = (
                        -(
                            face_velocity[index + 1]
                                * (dual_constant<scalar_t>(scalar_t(1))
                                   - delta[index + 1])
                            + diffusion_over_dy
                        )
                        + face_velocity[index] * delta[index]
                        - diffusion_over_dy
                    ) / dual_constant<scalar_t>(dy);
                    const Dual3<scalar_t> raw_upper = (
                        -face_velocity[index + 1] * delta[index + 1]
                        + diffusion_over_dy
                    ) / dual_constant<scalar_t>(dy);
                    lower[index] = index == 0
                        ? dual_constant<scalar_t>(scalar_t(0))
                        : raw_lower;
                    diagonal[index] = raw_diagonal;
                    upper[index] = index == interior - 1
                        ? dual_constant<scalar_t>(scalar_t(0))
                        : raw_upper;
                }

                const Dual3<scalar_t> lower_flux_coefficient
                    = -face_velocity[0] * delta[0] + diffusion_over_dy;
                const Dual3<scalar_t> upper_flux_coefficient
                    = face_velocity[interior]
                        * (dual_constant<scalar_t>(scalar_t(1))
                           - delta[interior])
                        + diffusion_over_dy;
                const scalar_t input_left = history_values[step_index][lane][0];
                const scalar_t output_left
                    = history_values[step_index + 1][lane][0];
                const scalar_t input_right
                    = history_values[step_index][lane][interior - 1];
                const scalar_t output_right
                    = history_values[step_index + 1][lane][interior - 1];
                const scalar_t midpoint_left = scalar_t(0.5)
                    * (input_left + output_left);
                const scalar_t midpoint_right = scalar_t(0.5)
                    * (input_right + output_right);
                const scalar_t raw_lower_flux
                    = lower_flux_coefficient.value * midpoint_left;
                const scalar_t raw_upper_flux
                    = upper_flux_coefficient.value * midpoint_right;
                const scalar_t lower_flux = std::max(
                    raw_lower_flux, scalar_t(0)
                );
                const scalar_t upper_flux = std::max(
                    raw_upper_flux, scalar_t(0)
                );
                const scalar_t overlap_left = local_low > t0
                    ? local_low
                    : t0;
                const scalar_t overlap_right = local_high < t1
                    ? local_high
                    : t1;
                const scalar_t overlap = overlap_right > overlap_left
                    ? overlap_right - overlap_left
                    : scalar_t(0);
                const bool upper_choice = choice_values[lane] > scalar_t(0.5);
                const scalar_t selected_flux = upper_choice
                    ? upper_flux
                    : lower_flux;
                const bool positive_selected_flux = upper_choice
                    ? raw_upper_flux > scalar_t(0)
                    : raw_lower_flux > scalar_t(0);

                std::fill(
                    gradient_input.begin(), gradient_input.end(), scalar_t(0)
                );
                gradient_solution = gradient_state;
                scalar_t local_drift_gradient = scalar_t(0);
                if (overlap > scalar_t(0) && positive_selected_flux) {
                    const scalar_t gradient_flux = gradient_observed * overlap;
                    const Dual3<scalar_t> selected_coefficient = upper_choice
                        ? upper_flux_coefficient
                        : lower_flux_coefficient;
                    const int64_t boundary_index = upper_choice
                        ? interior - 1
                        : 0;
                    gradient_input[boundary_index] += scalar_t(0.5)
                        * gradient_flux * selected_coefficient.value;
                    gradient_solution[boundary_index] += scalar_t(0.5)
                        * gradient_flux * selected_coefficient.value;
                    accumulate_parameter_gradient(
                        gradient_flux
                            * (upper_choice ? midpoint_right : midpoint_left),
                        selected_coefficient,
                        &local_drift_gradient
                    );
                }
                if (overlap > scalar_t(0)) {
                    const scalar_t gradient_overlap
                        = gradient_observed * selected_flux;
                    if (local_low > t0) {
                        local_low_gradient -= gradient_overlap;
                    }
                    if (local_high < t1) {
                        local_high_gradient += gradient_overlap;
                    }
                }

                const scalar_t theta = step_index < rannacher_steps
                    ? scalar_t(1)
                    : scalar_t(0.5);
                for (int64_t index = 0; index < interior; ++index) {
                    const scalar_t original_lower
                        = -theta * dt * lower[index].value;
                    const scalar_t original_diagonal
                        = scalar_t(1) - theta * dt * diagonal[index].value;
                    const scalar_t original_upper
                        = -theta * dt * upper[index].value;
                    lower_transpose[index] = index == 0
                        ? scalar_t(0)
                        : -theta * dt * upper[index - 1].value;
                    diagonal_transpose[index] = original_diagonal;
                    upper_transpose[index] = index == interior - 1
                        ? scalar_t(0)
                        : -theta * dt * lower[index + 1].value;
                    (void) original_lower;
                    (void) original_upper;
                }
                c_prime[0] = upper_transpose[0] / diagonal_transpose[0];
                d_prime[0] = gradient_solution[0] / diagonal_transpose[0];
                for (int64_t index = 1; index < interior; ++index) {
                    const scalar_t denominator = diagonal_transpose[index]
                        - lower_transpose[index] * c_prime[index - 1];
                    c_prime[index] = index == interior - 1
                        ? scalar_t(0)
                        : upper_transpose[index] / denominator;
                    d_prime[index] = (
                        gradient_solution[index]
                        - lower_transpose[index] * d_prime[index - 1]
                    ) / denominator;
                }
                lambda[interior - 1] = d_prime[interior - 1];
                for (int64_t index = interior - 2; index >= 0; --index) {
                    lambda[index] = d_prime[index]
                        - c_prime[index] * lambda[index + 1];
                }

                const scalar_t explicit_scale = (scalar_t(1) - theta) * dt;
                for (int64_t index = 0; index < interior; ++index) {
                    gradient_input[index] += lambda[index];
                    gradient_input[index] += explicit_scale
                        * diagonal[index].value * lambda[index];
                    if (index > 0) {
                        gradient_input[index - 1] += explicit_scale
                            * lower[index].value * lambda[index];
                    }
                    if (index + 1 < interior) {
                        gradient_input[index + 1] += explicit_scale
                            * upper[index].value * lambda[index];
                    }

                    const scalar_t input_center
                        = history_values[step_index][lane][index];
                    const scalar_t output_center
                        = history_values[step_index + 1][lane][index];
                    const scalar_t center_mix = theta * output_center
                        + (scalar_t(1) - theta) * input_center;
                    accumulate_parameter_gradient(
                        dt * lambda[index] * center_mix,
                        diagonal[index],
                        &local_drift_gradient
                    );
                    if (index > 0) {
                        const scalar_t input_left_value
                            = history_values[step_index][lane][index - 1];
                        const scalar_t output_left_value
                            = history_values[step_index + 1][lane][index - 1];
                        accumulate_parameter_gradient(
                            dt * lambda[index]
                                * (
                                    theta * output_left_value
                                    + (scalar_t(1) - theta)
                                        * input_left_value
                                ),
                            lower[index],
                            &local_drift_gradient
                        );
                    }
                    if (index + 1 < interior) {
                        const scalar_t input_right_value
                            = history_values[step_index][lane][index + 1];
                        const scalar_t output_right_value
                            = history_values[step_index + 1][lane][index + 1];
                        accumulate_parameter_gradient(
                            dt * lambda[index]
                                * (
                                    theta * output_right_value
                                    + (scalar_t(1) - theta)
                                        * input_right_value
                                ),
                            upper[index],
                            &local_drift_gradient
                        );
                    }
                }
                gradient_state.swap(gradient_input);
                drift_gradients[lane][step_index] = local_drift_gradient;
            }
            threshold_gradients[lane] = local_threshold_gradient;
            collapse_gradients[lane] = local_collapse_gradient;
            low_gradients[lane] = local_low_gradient;
            high_gradients[lane] = local_high_gradient;
    }
    return {
        gradient_drift,
        gradient_threshold,
        gradient_collapse,
        gradient_low,
        gradient_high,
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

void check_drift_inputs(
    const at::Tensor& state,
    const at::Tensor& task,
    const at::Tensor& gain,
    const at::Tensor& stimulus,
    const at::Tensor& correct_response
) {
    TORCH_CHECK(state.device().is_cpu(), "The native LCA drift scan is CPU-only.");
    TORCH_CHECK(state.dim() == 2 && state.size(1) == 2, "state must have shape [batch, 2].");
    const int64_t batch = state.size(0);
    TORCH_CHECK(task.sizes() == at::IntArrayRef({batch, 2}), "task must have shape [batch, 2].");
    TORCH_CHECK(gain.sizes() == at::IntArrayRef({batch}), "gain must have shape [batch].");
    TORCH_CHECK(stimulus.sizes() == at::IntArrayRef({batch, 4}), "stimulus must have shape [batch, 4].");
    TORCH_CHECK(correct_response.sizes() == at::IntArrayRef({batch}), "correct_response must have shape [batch].");
    for (const at::Tensor& value : {state, task, gain, stimulus, correct_response}) {
        TORCH_CHECK(value.device().is_cpu(), "All native LCA drift inputs must be on the CPU.");
        TORCH_CHECK(value.is_contiguous(), "All native LCA drift inputs must be contiguous.");
        TORCH_CHECK(value.scalar_type() == state.scalar_type(), "All native LCA drift inputs must share a dtype.");
    }
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

std::vector<at::Tensor> ddm_forward(
    const at::Tensor& drift,
    const at::Tensor& threshold,
    const at::Tensor& collapse_rate,
    const at::Tensor& interval_low,
    const at::Tensor& interval_high,
    const at::Tensor& choice,
    double time_step,
    int64_t spatial_points,
    double noise,
    double boundary_floor,
    int64_t rannacher_steps,
    bool store_history
) {
    TORCH_CHECK(drift.device().is_cpu(), "The native DDM forward solve is CPU-only.");
    TORCH_CHECK(drift.dim() == 2, "drift must have shape [batch, time].");
    const int64_t batch = drift.size(0);
    for (const at::Tensor& value : {
        threshold, collapse_rate, interval_low, interval_high, choice
    }) {
        TORCH_CHECK(value.sizes() == at::IntArrayRef({batch}), "DDM lane inputs must have shape [batch].");
        TORCH_CHECK(value.device().is_cpu(), "All native DDM inputs must be on the CPU.");
        TORCH_CHECK(value.is_contiguous(), "All native DDM inputs must be contiguous.");
        TORCH_CHECK(value.scalar_type() == drift.scalar_type(), "All native DDM inputs must share a dtype.");
    }
    TORCH_CHECK(drift.is_contiguous(), "drift must be contiguous.");
    TORCH_CHECK(spatial_points >= 5 && spatial_points % 2 == 1, "spatial_points must be odd and at least five.");
    std::vector<at::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(drift.scalar_type(), "ddm_forward", [&] {
        result = ddm_forward_impl<scalar_t>(
            drift,
            threshold,
            collapse_rate,
            interval_low,
            interval_high,
            choice,
            time_step,
            spatial_points,
            noise,
            boundary_floor,
            rannacher_steps,
            store_history
        );
    });
    return result;
}

std::vector<at::Tensor> ddm_backward(
    const at::Tensor& history,
    const at::Tensor& drift,
    const at::Tensor& threshold,
    const at::Tensor& collapse_rate,
    const at::Tensor& interval_low,
    const at::Tensor& interval_high,
    const at::Tensor& choice,
    const at::Tensor& invalid,
    const at::Tensor& gradient_probability,
    double time_step,
    int64_t spatial_points,
    double noise,
    int64_t rannacher_steps
) {
    TORCH_CHECK(history.device().is_cpu(), "The native DDM adjoint is CPU-only.");
    TORCH_CHECK(history.is_contiguous(), "density history must be contiguous.");
    TORCH_CHECK(invalid.scalar_type() == at::kBool, "invalid must be boolean.");
    TORCH_CHECK(invalid.is_contiguous(), "invalid must be contiguous.");
    TORCH_CHECK(gradient_probability.is_contiguous(), "probability gradient must be contiguous.");
    std::vector<at::Tensor> result;
    AT_DISPATCH_FLOATING_TYPES(drift.scalar_type(), "ddm_backward", [&] {
        result = ddm_backward_impl<scalar_t>(
            history,
            drift,
            threshold,
            collapse_rate,
            interval_low,
            interval_high,
            choice,
            invalid,
            gradient_probability,
            time_step,
            spatial_points,
            noise,
            rannacher_steps
        );
    });
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &lca_subject_forward, "Fused CSI LCA subject scan");
    module.def("backward", &lca_subject_backward, "Fused CSI LCA subject adjoint");
    module.def("drift_forward", &lca_drift_forward, "Fused CSI LCA drift paths");
    module.def("drift_backward", &lca_drift_backward, "Fused CSI LCA drift adjoint");
    module.def("ddm_forward", &ddm_forward, "Fused moving-boundary DDM forward solve");
    module.def("ddm_backward", &ddm_backward, "Fused moving-boundary DDM adjoint");
}
