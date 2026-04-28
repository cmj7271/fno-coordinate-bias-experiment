module PeriodicHeat

using FFTW
using LinearAlgebra
using Statistics

"""
    solve_periodic_heat(u0, nu, T)

Solves u_t = nu * u_xx with periodic boundary conditions using spectral methods.
"""
function solve_periodic_heat(u0::Vector{Float32}, nu::Float32, T::Float32)::Vector{Float32}
    N = length(u0)
    
    # 1. FFT the initial condition u0
    u0_hat = fft(u0)
    
    # 2. Correct frequencies for FFTW: [0, 1, ..., N/2, -N/2+1, ..., -1]
    k = fftfreq(N, Float32(N))
    
    # 3. Evolve the modes: u_hat(k, T) = u0_hat(k) * exp(-nu * (2πk)^2 * T)
    eigenvalues = @. -nu * (2 * pi * k)^2 * T
    u_hat_final = u0_hat .* exp.(eigenvalues)
    
    # 4. Inverse FFT to get the final state u(x, T)
    return real.(ifft(u_hat_final))
end

end # module PeriodicHeat
