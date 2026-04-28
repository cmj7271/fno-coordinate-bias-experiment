# ==============================================================================
# Metrics.jl: Collection of performance evaluation metrics for PDE solvers.
# ==============================================================================

module Metrics

using Statistics
using LinearAlgebra

"""
    relative_l2(y_pred, y_true)

Calculates the relative L2 norm between prediction and ground truth.
Formula: ||pred - true||₂ / ||true||₂

# Arguments
- `y_pred`: Predicted value array (Float32).
- `y_true`: True value array (Float32).

# Returns
Float32: The relative L2 error.
"""
function relative_l2(y_pred::AbstractArray{<:Real}, y_true::AbstractArray{<:Real})::Float32
    # Calculate squared difference and sum
    diff_sq = sum((y_pred .- y_true).^2)
    
    # Calculate L2 norm of the true values
    true_l2_sq = sum(y_true .^ 2)
    
    if true_l2_sq < 1e-12
        # Avoid division by zero if the true solution is nearly zero everywhere.
        # In this case, error is 0 if prediction is also near zero, otherwise high.
        return sqrt(diff_sq) < 1e-6 ? 0.0 : Inf
    end
    
    return sqrt(diff_sq) / sqrt(true_l2_sq)
end

function batch_relative_l2(y_pred::AbstractArray{<:Real, 3}, y_true::AbstractArray{<:Real, 3})
    @assert size(y_pred) == size(y_true) "batch_relative_l2 size mismatch: pred=$(size(y_pred)), true=$(size(y_true))"

    S = size(y_pred, 3)
    values = Vector{Float32}(undef, S)

    @views for s in 1:S
        values[s] = relative_l2(y_pred[:, :, s], y_true[:, :, s])
    end

    return values
end


function initial_sensitivity_error(y_pred::AbstractArray{<:Real, 3}, y_true::AbstractArray{<:Real, 3})
    @assert size(y_pred) == size(y_true) "initial_sensitivity_error size mismatch"

    S = size(y_pred, 3)
    num_pairs = div(S, 2)

    if num_pairs == 0
        return missing
    end

    errors = Float32[]

    @views for pair_idx in 1:num_pairs
        i = 2 * pair_idx - 1
        j = 2 * pair_idx

        true_sens = norm(vec(y_true[:, :, i] .- y_true[:, :, j]))
        pred_sens = norm(vec(y_pred[:, :, i] .- y_pred[:, :, j]))

        push!(errors, Float32(abs(pred_sens - true_sens) / (true_sens + eps(Float32))))
    end

    return mean(errors)
end

function shift_physical_channels(X::AbstractArray{<:Real, 3}, shift_steps::Int; has_coord::Bool=false)
    @assert ndims(X) == 3 "X must be (nx, channels, samples)."

    if has_coord && size(X, 2) >= 2
        X_shifted = copy(X)

        # 마지막 채널을 coordinate라고 보고, 물리 입력 채널만 shift
        X_shifted[:, 1:(end - 1), :] .= circshift(X[:, 1:(end - 1), :], (shift_steps, 0, 0))
        X_shifted[:, end:end, :] .= X[:, end:end, :]

        return X_shifted
    else
        return circshift(X, (shift_steps, 0, 0))
    end
end

function shift_equivariance_error(
    predict_fn,
    X::AbstractArray{<:Real, 3};
    shift_steps::Int,
    has_coord::Bool=false,
)
    X_shifted = shift_physical_channels(X, shift_steps; has_coord=has_coord)

    Y_pred = predict_fn(X)
    Y_pred_from_shifted = predict_fn(X_shifted)

    expected_shifted_Y = circshift(Y_pred, (shift_steps, 0, 0))

    values = batch_relative_l2(Y_pred_from_shifted, expected_shifted_Y)

    return mean(values)
end

function boundary_error(Y_pred::AbstractArray{<:Real, 3})
    @assert ndims(Y_pred) == 3
    @assert size(Y_pred, 2) == 1

    left = abs.(Y_pred[1, 1, :])
    right = abs.(Y_pred[end, 1, :])

    return Float32(mean(left .+ right))
end

end # module Metrics
