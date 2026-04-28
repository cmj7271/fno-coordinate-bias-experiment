# ==============================================================================
# DataGenerators.jl: Orchestrates the generation of all PDE datasets.
# This module is responsible for calling the specific PDE solvers and ensuring
# the data shape contract (X: (nx, c, samples), Y: (nx, 1, samples)) is met.
# ==============================================================================

module DataGenerators

using Random

# Include solvers and random fields
using ..PeriodicHeat
using ..DirichletHeat
using ..PeriodicPoisson
using ..VariablePoisson
using ..RandomFields

# Helper function to ensure data is correctly shaped (nx, channels, samples)
function assert_data_shape!(data, expected_channels::Int, samples::Int)
    @assert ndims(data) == 2 || ndims(data) == 3 "Data must be 2D or 3D."
    if ndims(data) == 3
        @assert size(data, 2) == expected_channels "Expected channels dimension mismatch."
        @assert size(data, 3) == samples "Expected sample count mismatch."
    end
end

# Helper to stack vector of vectors into (nx, 1, samples)
function stack_to_3d(vec_of_vecs)
    nx = length(vec_of_vecs[1])
    samples = length(vec_of_vecs)
    # reduce(hcat) gives (nx, samples)
    mat = reduce(hcat, vec_of_vecs)
    return reshape(mat, nx, 1, samples)
end

"""
    generate_periodic_heat_dataset(config)
"""
function generate_periodic_heat_dataset(config::Dict)
    config_data = config[:data]
    config_pde = config[:pde]
    
    N = config_data[:nx]
    grid = Float32.(collect(range(0.0, 1.0, length=N)))
    samples = config_data[:train_samples]
    
    rng = Random.seed!(config[:seed])
    
    X_list = []
    Y_list = []
    
    for s in 1:samples
        u0 = random_periodic_field(grid; max_mode=Int(config_data[:max_mode]), amplitude=Float32(config_data[:amplitude]), rng=rng)
        u_target = PeriodicHeat.solve_periodic_heat(u0, Float32(config_pde[:nu]), Float32(config_pde[:T]))
        push!(X_list, u0)
        push!(Y_list, u_target)
    end
    
    X = stack_to_3d(X_list)
    Y = stack_to_3d(Y_list)
    
    assert_data_shape!(X, 1, samples)
    assert_data_shape!(Y, 1, samples)
    return Dict(:X => X, :Y => Y)
end

"""
    generate_dirichlet_heat_dataset(config)
"""
function generate_dirichlet_heat_dataset(config::Dict)
    config_data = config[:data]
    config_pde = config[:pde]
    
    N = config_data[:nx]
    grid = Float32.(collect(range(0.0, 1.0, length=N)))
    samples = config_data[:train_samples]
    
    rng = Random.seed!(config[:seed])
    
    X_list = []
    Y_list = []
    
    for s in 1:samples
        u0 = random_sine_field(grid; max_mode=Int(config_data[:max_mode]), amplitude=Float32(config_data[:amplitude]), rng=rng)
        u_target = DirichletHeat.solve_dirichlet_heat(u0, Float32(config_pde[:nu]), Float32(config_pde[:T]))
        push!(X_list, u0)
        push!(Y_list, u_target)
    end
    
    X = stack_to_3d(X_list)
    Y = stack_to_3d(Y_list)
    
    assert_data_shape!(X, 1, samples)
    assert_data_shape!(Y, 1, samples)
    return Dict(:X => X, :Y => Y)
end

"""
    generate_periodic_poisson_dataset(config)
"""
function generate_periodic_poisson_dataset(config::Dict)
    config_data = config[:data]
    
    N = config_data[:nx]
    grid = Float32.(collect(range(0.0, 1.0, length=N)))
    samples = config_data[:train_samples]
    
    rng = Random.seed!(config[:seed])
    
    X_list = []
    Y_list = []
    
    for s in 1:samples
        f = random_periodic_field(grid; max_mode=Int(config_data[:max_mode]), amplitude=Float32(config_data[:amplitude]), rng=rng)
        u_target = PeriodicPoisson.solve_periodic_poisson(f)
        push!(X_list, f)
        push!(Y_list, u_target)
    end
    
    X = stack_to_3d(X_list)
    Y = stack_to_3d(Y_list)
    
    assert_data_shape!(X, 1, samples)
    assert_data_shape!(Y, 1, samples)
    return Dict(:X => X, :Y => Y)
end

"""
    generate_variable_poisson_dataset(config)
"""
function generate_variable_poisson_dataset(config::Dict)
    config_data = config[:data]
    
    N = config_data[:nx]
    grid = Float32.(collect(range(0.0, 1.0, length=N)))
    samples = config_data[:train_samples]
    
    rng = Random.seed!(config[:seed])
    
    X_list = []
    Y_list = []
    
    for s in 1:samples
        a = random_positive_coefficient(grid; max_mode=Int(config_data[:max_mode]), amplitude=Float32(config_data[:amplitude]), rng=rng)
        u_target = VariablePoisson.solve_variable_poisson(a, grid)
        push!(X_list, a)
        push!(Y_list, u_target)
    end
    
    X = stack_to_3d(X_list)
    Y = stack_to_3d(Y_list)
    
    assert_data_shape!(X, 1, samples)
    assert_data_shape!(Y, 1, samples)
    return Dict(:X => X, :Y => Y)
end

end # module DataGenerators
