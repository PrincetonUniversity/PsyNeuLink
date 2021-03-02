import taichi as ti
import taichi_utils as tu

ti.init(arch=ti.gpu)

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


if __name__ == "__main__":

    ddm_params = dict(starting_value=0.0, non_decision_time=0.15, drift_rate=0.3, threshold=0.6, time_step_size=0.001)

    simulate_many_ddms(*list(ddm_params.values()))
    rts = rt.to_numpy()
    choices = decision.to_numpy()

    ti.sync()
    rts = rt.to_numpy()
    choices = decision.to_numpy()
    valid = rts > 0.0
    rts = rts[valid]
    choices = choices[valid]

    import time
    t0 = time.time()

    NUM_TIMES = 50
    for i in range(NUM_TIMES):
        simulate_many_ddms(*list(ddm_params.values()))
        ti.sync()
        rts = rt.to_numpy()
        choices = decision.to_numpy()
        valid = rts > 0.0
        rts = rts[valid]
        choices = choices[valid]

    print(f"Elapsed: { 1000*((time.time() - t0) / NUM_TIMES)} milliseconds")