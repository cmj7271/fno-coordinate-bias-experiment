module DirichletHeat

using LinearAlgebra

"""
    solve_dirichlet_heat(u0, nu, T)

Solves u_t = nu * u_xx with Dirichlet boundary u(0,t)=u(1,t)=0 using sine series.
"""
function solve_dirichlet_heat(u0::Vector{Float32}, nu::Float32, T::Float32)::Vector{Float32}
    N = length(u0)
    grid = collect(range(0.0f0, 1.0f0, length=N))
    
    n_max = 20 # Sufficient for small T
    u_final = zeros(Float32, N)
    h = 1.0f0 / N
    
    for n in 1:n_max
        # Inner product a_n = 2 * ∫ u0(x) sin(nπx) dx
        a_n = 0.0f0
        for i in 1:N
            a_n += u0[i] * sin(pi * n * grid[i])
        end
        a_n *= (2.0f0 * h)
        
        # Evolution
        u_final .+= a_n * exp(-nu * (pi * n)^2 * T) .* sin.(pi * n .* grid)
    end
    
    return u_final
end

end # module DirichletHeat
