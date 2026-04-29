using Test
using FNOExperiment.Metrics
using LinearAlgebra
using Statistics

# Test Metrics Logic (test_metrics.jl)

@testset "Test relative_l2" begin
    # Perfect match
    @test Metrics.relative_l2(ones(5), ones(5)) ≈ 0.0
    
    # No match
    @test Metrics.relative_l2(zeros(5), ones(5)) > 0.5 # Should be roughly 1.0
end

@testset "Test batch_relative_l2" begin
    # Test perfect match across a batch
    y_pred = ones(5, 1, 3)
    y_true = ones(5, 1, 3)
    @test all(Metrics.batch_relative_l2(y_pred, y_true) .≈ 0.0)
    
    # Test non-match
    y_pred = ones(5, 1, 2)
    y_true = zeros(5, 1, 2)
    # Expected error for a single sample: sqrt(5) / sqrt(5) = 1.0
    # Average over 2 samples: 1.0
    @test mean(Metrics.batch_relative_l2(y_pred, y_true)) > 0.95
end

@testset "Test initial_sensitivity_error" begin
    # Perfect match: all adjacent differences are zero
    y_pred = ones(5, 1, 3)
    y_true = ones(5, 1, 3)
    @test Metrics.initial_sensitivity_error(y_pred, y_true) ≈ 0.0
    
    # Significant difference
    # True diff (S_true) will be non-zero
    y_true = reshape(Float32[
        1, 1, 1, 1, 1,  
        2, 2, 2, 2, 2,  
        3, 3, 3, 3, 3,  
        4, 4, 4, 4, 4
    ], 5, 1, 4)
    
    # Pred diff (S_pred): We make the prediction match the true difference
    y_pred_diff_match = y_true
    y_pred = y_pred_diff_match 
    
    # Error should be close to 0
    @test Metrics.initial_sensitivity_error(y_pred, y_true) < 0.1
end