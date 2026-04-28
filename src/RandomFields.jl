module RandomFields

using Random
using Statistics

export random_periodic_field, random_sine_field, random_positive_coefficient

"""
    random_periodic_field(grid; max_mode, amplitude, rng)

Generates a random, periodic, approximately mean-zero field using sine/cosine modes.
"""
function random_periodic_field(
    grid::Vector{Float32};
    max_mode::Int,
    amplitude::Float32,
    rng::Random.AbstractRNG,
)::Vector{Float32}
    field = zeros(Float32, length(grid))

    for k in 1:max_mode
        a = amplitude * randn(rng, Float32)
        b = amplitude * randn(rng, Float32)

        angle = Float32(2π * k) .* grid
        field .+= a .* sin.(angle)
        field .+= b .* cos.(angle)
    end

    # Numerical centering. Helpful for periodic Poisson forcing.
    field .-= mean(field)

    return field
end

"""
    random_sine_field(grid; max_mode, amplitude, rng)

Generates a random sine-basis field for Dirichlet problems.
The result is zero at x=0 and x=1 if the grid includes those endpoints.
"""
function random_sine_field(
    grid::Vector{Float32};
    max_mode::Int,
    amplitude::Float32,
    rng::Random.AbstractRNG,
)::Vector{Float32}
    field = zeros(Float32, length(grid))

    for n in 1:max_mode
        a = amplitude * randn(rng, Float32)
        angle = Float32(π * n) .* grid
        field .+= a .* sin.(angle)
    end

    return field
end

"""
    random_positive_coefficient(grid; max_mode, amplitude, rng)

Generates a smooth positive coefficient a(x) for variable Poisson problems.
"""
function random_positive_coefficient(
    grid::Vector{Float32};
    max_mode::Int,
    amplitude::Float32,
    rng::Random.AbstractRNG,
)::Vector{Float32}
    raw = random_periodic_field(
        grid;
        max_mode=max_mode,
        amplitude=amplitude,
        rng=rng,
    )

    # Strictly positive coefficient. The small offset avoids values too close to zero.
    return exp.(Float32(0.5) .* raw) .+ Float32(1.0e-3)
end

end # module RandomFields