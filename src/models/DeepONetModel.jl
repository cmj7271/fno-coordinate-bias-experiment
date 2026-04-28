module DeepONetModel

using Lux
using LinearAlgebra
using Statistics

"""
    DeepONet(branch, trunk)

Custom Lux layer for Deep Operator Network.
"""
struct DeepONet <: Lux.AbstractLuxContainerLayer{(:branch, :trunk)}
    branch::Lux.AbstractLuxLayer
    trunk::Lux.AbstractLuxLayer
end

function (m::DeepONet)(x, ps, st)
    X, grid = x

    nx = size(X, 1)
    in_channels = size(X, 2)
    batch = size(X, 3)

    u = reshape(X, nx * in_channels, batch)
    y = reshape(grid, 1, nx)
    
    b, st_b = m.branch(u, ps.branch, st.branch) # (latent_dim, batch)
    t, st_t = m.trunk(y, ps.trunk, st.trunk)    # (latent_dim, nx)
    
    # Output: (nx, batch)
    res = transpose(t) * b
    
    # Reshape to (nx, 1, batch)
    y_hat = reshape(res, size(res, 1), 1, size(res, 2))
    return y_hat, (branch=st_b, trunk=st_t)
end

"""
    build_deeponet_model(config; nx, in_channels)
"""
function build_deeponet_model(config::Dict{Symbol, Any}; nx::Int, in_channels::Int)
    branch_width = config[:branch_width]
    trunk_width = config[:trunk_width]
    latent_dim = config[:latent_dim]
    depth = config[:depth]
    activation = config[:activation] == "tanh" ? tanh : gelu
    
    @info "Building DeepONet model with Branch-Trunk architecture."

    branch = Chain(
        Dense(nx * in_channels => branch_width, activation),
        [Dense(branch_width => branch_width, activation) for _ in 1:(depth-1)]...,
        Dense(branch_width => latent_dim),
    )
    
    trunk = Chain(
        Dense(1 => trunk_width, activation),
        [Dense(trunk_width => trunk_width, activation) for _ in 1:(depth-1)]...,
        Dense(trunk_width => latent_dim)
    )
    
    return DeepONet(branch, trunk)
end

end # module DeepONetModel
