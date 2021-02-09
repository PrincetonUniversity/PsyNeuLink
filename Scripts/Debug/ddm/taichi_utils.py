import math
import taichi as ti
import taichi_glsl as ts


@ti.func
def rand_normal():
    """
    Generate a normally distributed random number with mean=0 and variance=1

    Returns
    -------
        A scalar randome number
    """
    u0 = ti.random(ti.f32)
    u1 = ti.random(ti.f32)
    r = ti.sqrt(-2*ti.log(u0))
    theta = 2 * math.pi * u1
    r0 = r * ti.sin(theta)
    #r1 = r * ti.cos(theta)
    return r0

@ti.func
def rand_normal2():
    """
    Generate a normally distributed random number with mean=0 and variance=1

    Returns
    -------
        A scalar randome number
    """
    u0 = ti.random(ti.f32)
    u1 = ti.random(ti.f32)
    r = ti.sqrt(-2*ti.log(u0))
    theta = 2 * math.pi * u1
    r0 = r * ti.sin(theta)
    r1 = r * ti.cos(theta)
    return ti.Vector([r0, r1])



@ti.func
def rand_normal_vec():
    v = ti.Vector([0.0, 0.0])
    # for i in ti.static(range(0, len(v), 2)):
    #     u0 = ti.random(ti.f32)
    #     u1 = ti.random(ti.f32)
    #     r = ti.sqrt(-2*ti.log(u0))
    #     theta = 2 * math.pi * u1
    #     r0 = r * ti.sin(theta)
    #     v[i] = r0
    #     v[i+1] = r * ti.cos(theta)

    return v


@ti.pyfunc
def relu(v: ti.template()):
    return (ti.abs(v) + v) / 2.0

@ti.pyfunc
def logistic(v: ti.template(), gain):
    return 1.0 / (1.0 + ti.exp(-gain * v))


@ti.func
def argmax(v: ti.template()):
    maxval = v.max()
    for i in ti.static(range(2)):
        if v[i] == maxval:
            return i

