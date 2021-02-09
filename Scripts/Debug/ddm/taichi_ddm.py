import taichi as ti
import taichi_utils as tu

num_simulations = 1000000
rt = ti.field(ti.f32, num_simulations)
decision = ti.field(ti.i32, num_simulations)

@ti.func
def ddm_time_step(prev_value, drift_rate, time_step_size):
    return prev_value + (tu.rand_normal() + drift_rate * time_step_size) * ti.sqrt(time_step_size)


@ti.func
def simulate_ddm(starting_value, non_decision_time, drift_rate, threshold, time_step_size):
    particle = starting_value
    t = 0
    while abs(particle) < threshold:
        particle = ddm_time_step(particle, drift_rate, time_step_size)
        t = t + 1

    rt = (non_decision_time + t * time_step_size)
    decision = 1
    if particle < -threshold:
        decision = 0

    return rt, decision


@ti.kernel
def simulate_many_ddms(starting_value: ti.f32,
                       non_decision_time: ti.f32,
                       drift_rate: ti.f32,
                       threshold: ti.f32,
                       time_step_size: ti.f32):
    for i in rt:
        rt[i], decision[i] = simulate_ddm(starting_value, non_decision_time,
                                          drift_rate, threshold, time_step_size)
