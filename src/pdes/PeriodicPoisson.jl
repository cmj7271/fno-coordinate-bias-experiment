module PeriodicPoisson

using FFTW
using LinearAlgebra
using Statistics

"""
    solve_periodic_poisson(f)

Solves -u''(x) = f(x) with periodic boundary conditions using spectral methods.
"""
function solve_periodic_poisson(f::Vector{Float32})::Vector{Float32}
    N = length(f)
    
    # 1. FFT the forcing function f
    f_hat = fft(f)
    
    # 2. Correct frequencies
    k = fftfreq(N, Float32(N))
    
    # 3. Calculate u_hat(k) = f_hat(k) / (2πk)^2
    # k=0 mode is 0 (mean-zero condition)
    u_hat_final = zeros(ComplexF32, N)
    for i in 1:N
        if k[i] != 0
            u_hat_final[i] = f_hat[i] / (2 * pi * k[i])^2
        end
    end
    
    # 4. Inverse FFT to get the final state u(x)
    u_final = real.(ifft(u_hat_final))
    
    # Ensure mean is zero
    return u_final .- mean(u_final)
end

end # module PeriodicPoisson
