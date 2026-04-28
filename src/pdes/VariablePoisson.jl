module VariablePoisson

using LinearAlgebra

"""
    solve_variable_poisson(a, grid)

Solves -(a(x)u'(x))' = 1 with Dirichlet boundary u(0)=u(1)=0 using finite differences.
"""
function solve_variable_poisson(a::Vector{Float32}, grid::Vector{Float32})::Vector{Float32}
    N = length(grid)
    if N < 3
        return zeros(Float32, N)
    end
    
    h = grid[2] - grid[1]
    n_int = N - 2
    
    A = zeros(Float32, n_int, n_int)
    b = ones(Float32, n_int)
    
    for i in 1:n_int
        idx = i + 1 # Actual index in full grid
        
        # Staggered grid coefficients
        a_plus = (a[idx] + a[idx+1]) / 2.0f0
        a_minus = (a[idx] + a[idx-1]) / 2.0f0
        
        A[i, i] = (a_plus + a_minus) / h^2
        
        if i > 1
            A[i, i-1] = -a_minus / h^2
        end
        if i < n_int
            A[i, i+1] = -a_plus / h^2
        end
    end
    
    u_internal = A \ b
    
    u_final = zeros(Float32, N)
    u_final[2:end-1] .= u_internal
    return u_final
end

end # module VariablePoisson
