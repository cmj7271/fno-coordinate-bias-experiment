using Test
using Random
using LinearAlgebra
using Statistics

using FNOExperiment.DataGenerators
using FNOExperiment.Config
using FNOExperiment.DatasetIO

# Test Data Generation Logic (test_data_generation.jl)

@testset "Test random_periodic_field" begin
    @test typeof(DataGenerators.random_periodic_field(collect(range(0.0f0, 1.0f0, length=10)); max_mode=2, amplitude=1.0f0, rng=Random.default_rng())) == Vector{Float32}
end

@testset "Test random_sine_field" begin
    @test typeof(DataGenerators.random_sine_field(collect(range(0.0f0, 1.0f0, length=10)); max_mode=2, amplitude=1.0f0, rng=Random.default_rng())) == Vector{Float32}
end

@testset "Test random_positive_coefficient" begin
    @test typeof(DataGenerators.random_positive_coefficient(collect(range(0.0f0, 1.0f0, length=10)); max_mode=2, amplitude=1.0f0, rng=Random.default_rng())) == Vector{Float32}
end

@testset "Test Data Generation Pipeline" begin
    # Setup: Use a minimal dummy config since we can't access the file system easily in the test
    dummy_config = Dict(
        :name => "periodic_heat",
        :seed => 123,
        :data => Dict(
            :nx => 32,
            :train_samples => 1, # Test with 1 sample to simplify setup
            :valid_samples => 1,
            :test_samples => 1,
            :max_mode => 8,
            :amplitude => 1.0
        ),
        :pde => Dict(
            :nu => 0.01,
            :T => 0.1,
            :boundary => "periodic",
            :has_initial_condition => true,
            :is_translation_relative => true
        )
    )

    # Mock the RNG seed for determinism
    Random.seed!(123)

    # Test periodic_heat generation
    test_data = DataGenerators.generate_periodic_heat_dataset(dummy_config)
    X = test_data[:X]
    Y = test_data[:Y]

    # Assertions:
    # 1. Check shapes (nx, 1, 1) for single sample
    @test size(X) == (32, 1, 1)
    @test size(Y) == (32, 1, 1)
    
    # 2. Check for NaN or Inf
    @test all(isfinite, X)
    @test all(isfinite, Y)
    
    # 3. Check periodicity/mean (should be near zero for the target)
    # Since we only generated one sample, we check the mean of the target Y
    @test isapprox(mean(Y[:, :, 1]), 0.0, atol=1e-5)
end