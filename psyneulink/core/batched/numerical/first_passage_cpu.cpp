// Batched scalar first-passage PDE and discrete adjoint. No model-specific dynamics.
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

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

void check_ddm_inputs(
    const at::Tensor& drift,
    const at::Tensor& threshold,
    const at::Tensor& collapse_rate,
    const at::Tensor& interval_low,
    const at::Tensor& interval_high,
    const at::Tensor& choice,
    double time_step,
    int64_t spatial_points,
    double noise,
    int64_t rannacher_steps
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
    TORCH_CHECK(std::isfinite(time_step) && time_step > 0, "time_step must be finite and positive.");
    TORCH_CHECK(std::isfinite(noise) && noise > 0, "noise must be finite and positive.");
    TORCH_CHECK(rannacher_steps >= 0, "rannacher_steps cannot be negative.");
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
    check_ddm_inputs(
        drift, threshold, collapse_rate, interval_low, interval_high, choice,
        time_step, spatial_points, noise, rannacher_steps
    );
    TORCH_CHECK(std::isfinite(boundary_floor) && boundary_floor > 0, "boundary_floor must be finite and positive.");
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
    check_ddm_inputs(
        drift, threshold, collapse_rate, interval_low, interval_high, choice,
        time_step, spatial_points, noise, rannacher_steps
    );
    const int64_t batch = drift.size(0);
    TORCH_CHECK(history.device().is_cpu() && history.is_contiguous(), "density history must be contiguous CPU data.");
    TORCH_CHECK(history.scalar_type() == drift.scalar_type(), "density history must share the drift dtype.");
    TORCH_CHECK(history.sizes() == at::IntArrayRef({drift.size(1) + 1, batch, spatial_points - 2}),
        "density history must have shape [time+1, batch, spatial_points-2].");
    TORCH_CHECK(invalid.device().is_cpu() && invalid.is_contiguous() && invalid.scalar_type() == at::kBool,
        "invalid must be contiguous CPU boolean data.");
    TORCH_CHECK(invalid.sizes() == at::IntArrayRef({batch}), "invalid must have shape [batch].");
    TORCH_CHECK(gradient_probability.device().is_cpu() && gradient_probability.is_contiguous(),
        "probability gradient must be contiguous CPU data.");
    TORCH_CHECK(gradient_probability.scalar_type() == drift.scalar_type(), "probability gradient must share the drift dtype.");
    TORCH_CHECK(gradient_probability.sizes() == at::IntArrayRef({batch}), "probability gradient must have shape [batch].");
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
    module.def("ddm_forward", &ddm_forward, "Batched scalar first-passage forward solve");
    module.def("ddm_backward", &ddm_backward, "Batched scalar first-passage adjoint");
}
