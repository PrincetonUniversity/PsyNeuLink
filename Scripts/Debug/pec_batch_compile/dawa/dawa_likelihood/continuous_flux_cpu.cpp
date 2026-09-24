// CPU SSP-RK2 for a four-neighbor absorbing Markov generator, plus its adjoint.
// The absorbing time loop accepts arbitrary transition rates. A separate fused
// sigmoid-LCA coefficient calculation below matches the Torch discretization.
#include <torch/extension.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using at::Tensor;

static void check(Tensor mass, Tensor rates, int64_t substeps, double dt) {
    TORCH_CHECK(mass.device().is_cpu() && rates.device().is_cpu(), "CPU tensors required");
    TORCH_CHECK(mass.scalar_type() == at::kDouble && rates.scalar_type() == at::kDouble, "float64 required");
    TORCH_CHECK(mass.is_contiguous() && rates.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(mass.dim() == 2 && mass.size(0) == mass.size(1), "square mass grid required");
    TORCH_CHECK(rates.dim() == 4 && rates.size(0) >= 2 && rates.size(1) == 4 &&
                rates.size(2) == mass.size(0) && rates.size(3) == mass.size(1), "invalid rate shape");
    TORCH_CHECK(substeps > 0 && dt > 0 && std::isfinite(dt), "positive substeps and dt required");
}

static void euler(const double* mass, const double* r0, const double* r1, double fraction,
                  int64_t cells, double h,
                  double* output, double* flux) {
    const int64_t n = int64_t(std::sqrt(double(cells)));
    const double a=h*(1.-fraction), b=h*fraction;
    std::fill(flux, flux+3, 0.);
    for (int64_t i=0; i<cells; ++i)
        output[i]=mass[i]*(1.-a*(r0[i]+r0[cells+i]+r0[2*cells+i]+r0[3*cells+i])
                           -b*(r1[i]+r1[cells+i]+r1[2*cells+i]+r1[3*cells+i]));
    // Interior gathers have fixed offsets, allowing SIMD without scatter races.
    for (int64_t i=0; i<cells-n; ++i) {
        output[i]+=(a*r0[i+n]+b*r1[i+n])*mass[i+n];
        output[i+n]+=(a*r0[cells+i]+b*r1[cells+i])*mass[i];
    }
    for (int64_t row=0; row<cells; row+=n) for (int64_t j=0; j<n-1; ++j) {
        const int64_t i=row+j;
        output[i]+=(a*r0[2*cells+i+1]+b*r1[2*cells+i+1])*mass[i+1];
        output[i+1]+=(a*r0[3*cells+i]+b*r1[3*cells+i])*mass[i];
    }
    for (int64_t j=0; j<n; ++j) {
        const int64_t last=cells-n+j, left=j*n, right=left+n-1;
        flux[0]+=(a*r0[cells+last]+b*r1[cells+last])*mass[last];
        flux[1]+=(a*r0[3*cells+right]+b*r1[3*cells+right])*mass[right];
        flux[2]+=(a*r0[j]+b*r1[j])*mass[j]+(a*r0[2*cells+left]+b*r1[2*cells+left])*mass[left];
    }
}

static void euler_vjp(const double* mass, const double* r0, const double* r1, double fraction,
                      int64_t cells, double h,
                      const double* adjoint, const double* adj_flux, double* grad_mass,
                      double* grad_r0, double* grad_r1) {
    const int64_t n = int64_t(std::sqrt(double(cells)));
    std::copy(adjoint, adjoint+cells, grad_mass);
    for (int d=0; d<4; ++d) {
        const int64_t offset=d==0 ? -n : d==1 ? n : d==2 ? -1 : 1;
        const double exit=d==1 ? adj_flux[0] : d==3 ? adj_flux[1] : adj_flux[2];
        for (int64_t row=0; row<cells; row+=n) {
            const bool absorbing_row=(d==0 && row==0) || (d==1 && row==cells-n);
            const int64_t first=d==2 ? 1 : 0, stop=d==3 ? n-1 : n;
            for (int64_t j=first; j<stop; ++j) {
                const int64_t i=row+j, k=d*cells+i;
                const double destination=absorbing_row ? exit : adjoint[i+offset];
                const double difference=h*(destination-adjoint[i]);
                grad_mass[i]+=((1.-fraction)*r0[k]+fraction*r1[k])*difference;
                grad_r0[k]+=(1.-fraction)*mass[i]*difference;
                grad_r1[k]+=fraction*mass[i]*difference;
            }
            if (d>=2) {
                const int64_t i=row+(d==2 ? 0 : n-1), k=d*cells+i;
                const double difference=h*(exit-adjoint[i]);
                grad_mass[i]+=((1.-fraction)*r0[k]+fraction*r1[k])*difference;
                grad_r0[k]+=(1.-fraction)*mass[i]*difference;
                grad_r1[k]+=fraction*mass[i]*difference;
            }
        }
    }
}

static std::vector<Tensor> forward(Tensor mass, Tensor rates, int64_t substeps, double dt) {
    check(mass, rates, substeps, dt);
    const int64_t cells = mass.numel(), count = rates.size(0)-1;
    Tensor output = mass.clone(), exits = at::zeros({count, 3}, mass.options());
    std::vector<double> predicted(cells), advanced(cells);
    double minimum = *std::min_element(mass.data_ptr<double>(), mass.data_ptr<double>()+cells);
    double* m = output.data_ptr<double>();
    const double* r = rates.data_ptr<double>();
    for (int64_t k = 0; k < count; ++k) for (int64_t j = 0; j < substeps; ++j) {
        const double *r0 = r+k*4*cells, *r1 = r0+4*cells;
        double f0[3], f1[3];
        euler(m, r0, r1, double(j)/substeps, cells, dt/substeps, predicted.data(), f0);
        euler(predicted.data(), r0, r1, double(j+1)/substeps, cells, dt/substeps, advanced.data(), f1);
        for (int64_t i = 0; i < cells; ++i) {
            m[i] = .5*(m[i]+advanced[i]);
            minimum = std::min(minimum, m[i]);
        }
        for (int c = 0; c < 3; ++c) exits.data_ptr<double>()[3*k+c] += .5*(f0[c]+f1[c]);
    }
    return {output, exits, at::full({}, minimum, mass.options())};
}

static std::vector<Tensor> backward(Tensor mass, Tensor rates, Tensor adjoint, Tensor adj_exits,
                                    int64_t substeps, double dt) {
    check(mass, rates, substeps, dt);
    TORCH_CHECK(adjoint.sizes() == mass.sizes() && adj_exits.dim() == 2 &&
                adj_exits.size(0) == rates.size(0)-1 && adj_exits.size(1) == 3, "invalid adjoint shapes");
    const int64_t cells = mass.numel(), count = rates.size(0)-1, total = count*substeps;
    const double* r = rates.data_ptr<double>();
    // Recompute block trajectories instead of retaining a full-trial graph.
    std::vector<double> history((2*total+1)*cells), scratch(cells);
    std::copy(mass.data_ptr<double>(), mass.data_ptr<double>()+cells, history.data());
    for (int64_t s = 0; s < total; ++s) {
        const int64_t k = s/substeps, j = s%substeps;
        const double *r0 = r+k*4*cells, *r1 = r0+4*cells;
        double* original = history.data()+2*s*cells;
        double* predicted = original+cells;
        double* advanced = predicted+cells;
        double ignored[3];
        euler(original, r0, r1, double(j)/substeps, cells, dt/substeps, predicted, ignored);
        euler(predicted, r0, r1, double(j+1)/substeps, cells, dt/substeps, advanced, ignored);
        for (int64_t i = 0; i < cells; ++i) advanced[i] = .5*(original[i]+advanced[i]);
    }
    Tensor gm = adjoint.clone(), gr = at::zeros_like(rates);
    std::vector<double> half(cells), predicted_adjoint(cells);
    for (int64_t s = total; s-- > 0;) {
        const int64_t k = s/substeps, j = s%substeps;
        const double *r0 = r+k*4*cells, *r1 = r0+4*cells;
        double *gr0 = gr.data_ptr<double>()+k*4*cells, *gr1 = gr0+4*cells;
        const double* original = history.data()+2*s*cells;
        const double* predicted = original+cells;
        double af[3];
        for (int c = 0; c < 3; ++c) af[c] = .5*adj_exits.data_ptr<double>()[3*k+c];
        for (int64_t i = 0; i < cells; ++i) half[i] = .5*gm.data_ptr<double>()[i];
        euler_vjp(predicted, r0, r1, double(j+1)/substeps, cells, dt/substeps,
                  half.data(), af, predicted_adjoint.data(), gr0, gr1);
        euler_vjp(original, r0, r1, double(j)/substeps, cells, dt/substeps,
                  predicted_adjoint.data(), af, scratch.data(), gr0, gr1);
        for (int64_t i = 0; i < cells; ++i) gm.data_ptr<double>()[i] = scratch[i]+half[i];
    }
    return {gm, gr};
}

// Fused coefficients for the sigmoid LCA finite-volume operator. This is kept
// separate from the generic absorbing time loop above. Its VJP avoids retaining
// the many grid-sized intermediate tensors of the reference Torch expression.
static std::array<double, 2> bernoulli(double z) {
    if (std::abs(z) < 1.e-4)
        return {1.-z/2.+z*z/12.-z*z*z*z/720., -.5+z/6.-z*z*z/180.};
    const double a = std::abs(z), e = std::exp(-a), denominator = -std::expm1(-a);
    const double b = a*e/denominator, derivative = e/denominator*(1.-a/denominator);
    return z >= 0 ? std::array<double, 2>{b, derivative} : std::array<double, 2>{b-z, -derivative-1.};
}

static std::vector<Tensor> coefficients(Tensor inputs, Tensor gain, Tensor bias,
        Tensor boundary, Tensor boundary_rate, int64_t n, double low, double noise,
        double leak, double competition, Tensor adjoint) {
    TORCH_CHECK(n >= 2 && gain.dim() == 1 && bias.numel() == 1, "invalid coefficient dimensions");
    const int64_t count = gain.numel(), cells = n*n;
    TORCH_CHECK(inputs.dim() == 2 && inputs.size(0) == count && inputs.size(1) == 2 &&
                boundary.sizes() == gain.sizes() && boundary_rate.sizes() == gain.sizes(), "invalid coefficient shapes");
    for (const auto& a : {inputs, gain, bias, boundary, boundary_rate})
        TORCH_CHECK(a.device().is_cpu() && a.scalar_type() == at::kDouble && a.is_contiguous(), "CPU contiguous float64 required");
    const bool reverse = adjoint.defined();
    Tensor rates, gi, gg, gb, ga, gad;
    if (reverse) {
        TORCH_CHECK(adjoint.device().is_cpu() && adjoint.scalar_type() == at::kDouble && adjoint.is_contiguous() &&
                    adjoint.dim() == 4 && adjoint.size(0) == count && adjoint.size(1) == 4 &&
                    adjoint.size(2) == n && adjoint.size(3) == n, "invalid coefficient adjoint");
        gi=at::zeros_like(inputs); gg=at::zeros_like(gain); gb=at::zeros_like(bias);
        ga=at::zeros_like(boundary); gad=at::zeros_like(boundary_rate);
    } else rates=at::empty({count,4,n,n}, gain.options());
    const double h=1./n, diffusion=.5*noise*noise, b=bias.item<double>();
    for (int64_t k=0; k<count; ++k) {
        const double w=boundary.data_ptr<double>()[k]-low, g=gain.data_ptr<double>()[k];
        const double speed=boundary_rate.data_ptr<double>()[k];
        std::vector<double> activation(n), derivative(n);
        for (int64_t j=0; j<n; ++j) {
            const double argument=g*(low+w*(j+.5)*h+b), e=std::exp(-std::abs(argument));
            activation[j]=argument>=0 ? 1./(1.+e) : e/(1.+e);
            derivative[j]=e/((1.+e)*(1.+e));
        }
        // One face calculation supplies both neighboring outgoing rates.
        for (int axis=0; axis<2; ++axis) for (int64_t i=0; i<=n; ++i) for (int64_t j=0; j<n; ++j) {
            const double face=i*h, center=(j+.5)*h;
            const bool edge=i==0 || i==n;
            const double drift=inputs.data_ptr<double>()[2*k+axis]-leak*(low+w*face)-competition*activation[j]-face*speed;
            const double factor=(edge ? .5 : 1.)*h/diffusion, q=factor*drift*w;
            const auto bd=bernoulli(std::abs(q));
            const double down=q>=0 ? bd[0] : bd[0]-q, up=q>=0 ? bd[0]+q : bd[0];
            const double ddown=q>=0 ? bd[1] : -bd[1]-1., dup=q>=0 ? bd[1]+1. : -bd[1];
            const double prefactor=(edge ? 2. : 1.)*diffusion/(w*w*h*h);
            const int64_t di=(k*4+2*axis)*cells+(axis ? j*n+i : i*n+j);
            const int64_t ui=(k*4+2*axis+1)*cells+(axis ? j*n+i-1 : (i-1)*n+j);
            if (!reverse) {
                if (i<n) rates.data_ptr<double>()[di]=prefactor*down;
                if (i>0) rates.data_ptr<double>()[ui]=prefactor*up;
                continue;
            }
            const double adown=i<n ? adjoint.data_ptr<double>()[di] : 0.;
            const double aup=i>0 ? adjoint.data_ptr<double>()[ui] : 0.;
            const double weighted_rate=prefactor*(adown*down+aup*up);
            const double dq=prefactor*(adown*ddown+aup*dup)*factor, dr=dq*w;
            gi.data_ptr<double>()[2*k+axis]+=dr;
            gg.data_ptr<double>()[k]-=dr*competition*derivative[j]*(low+w*center+b);
            gb.data_ptr<double>()[0]-=dr*competition*derivative[j]*g;
            ga.data_ptr<double>()[k]+=-2.*weighted_rate/w+dq*drift
                                      +dr*(-leak*face-competition*derivative[j]*g*center);
            gad.data_ptr<double>()[k]-=dr*face;
        }

    }
    return reverse ? std::vector<Tensor>{gi,gg,gb,ga,gad} : std::vector<Tensor>{rates};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward);
    module.def("backward", &backward);
    module.def("coefficients", [](Tensor i, Tensor g, Tensor b, Tensor a, Tensor ad,
            int64_t n, double low, double noise, double leak, double competition) {
        return coefficients(i,g,b,a,ad,n,low,noise,leak,competition,Tensor())[0];
    });
    module.def("coefficients_vjp", &coefficients);
}
