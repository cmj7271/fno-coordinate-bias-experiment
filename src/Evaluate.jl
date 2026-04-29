# ==============================================================================
# Evaluate.jl: Handles the evaluation process and final metric calculation.
# ==============================================================================

module Evaluate

using Lux
using Statistics
using LinearAlgebra
using Random
using JLD2

using ..Config
using ..DatasetIO
using ..Metrics
using ..FNOModel
using ..DeepONetModel

export evaluate_dataset

function model_forward(model_name::Symbol, model, X, grid_points, ps, st)
    if model_name == :deeponet
        return model((X, grid_points), ps, st)
    elseif model_name == :fno
        return model(X, ps, st)
    else
        error("Unsupported model name: $model_name")
    end
end

"""
    evaluate_dataset(config_path::String; model_name::Symbol=:deeponet, use_coord::Bool=false)

Evaluates a trained model against a dataset. 
It loads the dataset using DatasetIO.PDEData (containing X, Y, Config, Grid) 
and calculates various metrics like relative L2 error and shift equivariance error.

# Arguments
- `config_path`: The TOML file path specifying the dataset type.
- `model_name`: The model architecture symbol (:fno or :deeponet).
- `use_coord`: Boolean indicating whether to use the coordinate channel.

# Returns
Dict containing the evaluation metrics.
"""
function evaluate_dataset(
    config_path::String;
    model_name::Symbol=:deeponet,
    use_coord::Bool=false,
)::Dict
    @info "--- Starting Dataset Evaluation for $config_path ---"
    
    # 1. Load config
    config = Config.load_config(config_path)
    
    nx = Int(config[:data][:nx])
    grid_points = Float32.(range(0, 1, length=nx))
    
    # 2. Use PDEData struct for robust data handling
    dataset = DatasetIO.load_and_preprocess_dataset(config_path, use_coord, grid_points)
    
    X_full = dataset.X
    Y_true = dataset.Y
    
    @info "Loaded evaluation data: X=$(size(X_full)), Y=$(size(Y_true))"
    
    @assert ndims(X_full) == 3 "X must be (nx, channels, samples). Got $(size(X_full))."
    @assert ndims(Y_true) == 3 "Y must be (nx, 1, samples). Got $(size(Y_true))."
    @assert size(X_full, 1) == size(Y_true, 1) "X/Y nx mismatch."
    @assert size(Y_true, 2) == 1 "Y must have one output channel."
    @assert size(X_full, 3) == size(Y_true, 3) "X/Y sample count mismatch."
    
    in_channels = size(X_full, 2)
    
    # 3. Build same model architecture
    if model_name == :fno
        model_config = config[:model][:fno]
        model = FNOModel.build_fno_model(
            model_config;
            in_channels=in_channels,
            out_channels=1,
        )
    elseif model_name == :deeponet
        model_config = config[:model][:deeponet]
        model = DeepONetModel.build_deeponet_model(
            model_config;
            nx=size(X_full, 1),
            in_channels=in_channels,
        )
    else
        error("Unsupported model name: $model_name")
    end
    
    checkpoint_path = DatasetIO.checkpoint_path(config, model_name, use_coord)
    @info "Loading checkpoint from $checkpoint_path"
    
    checkpoint = JLD2.load(checkpoint_path, "checkpoint")
    
    if !haskey(checkpoint, :ps) || !haskey(checkpoint, :st)
        error("Checkpoint does not contain trained parameters :ps and :st.")
    end
    
    ps = checkpoint[:ps]
    st = checkpoint[:st]
    
    # 4. Forward pass
    Y_pred, st = model_forward(model_name, model, X_full, grid_points, ps, st)
    
    @assert size(Y_pred) == size(Y_true) "Prediction size $(size(Y_pred)) does not match target size $(size(Y_true))."
    
    # 5. Relative L2 metrics
    rel_l2_values = Metrics.batch_relative_l2(Y_pred, Y_true)
    
    relative_l2_mean = mean(rel_l2_values)
    relative_l2_std = length(rel_l2_values) > 1 ? std(rel_l2_values) : 0.0f0
    
    # 6. Initial sensitivity metric
    initial_sensitivity = if get(config[:pde], :has_initial_condition, false)
        Metrics.initial_sensitivity_error(Y_pred, Y_true)
    else
        missing
    end
    
    # 7. Shift equivariance metric
    predict_fn = function(X_input)
        Y_tmp, _ = model_forward(model_name, model, X_input, grid_points, ps, st)
        return Y_tmp
    end
    
    shift_error = if get(config[:pde], :is_translation_relative, false)
        Metrics.shift_equivariance_error(
            predict_fn,
            X_full;
            shift_steps=Int(config[:eval][:shift_steps]),
            has_coord=use_coord,
        )
    else
        missing
    end
    
    # 8. Boundary error
    boundary_err = if get(config[:pde], :boundary, "") == "dirichlet"
        Metrics.boundary_error(Y_pred)
    else
        missing
    end
    
    @info "Evaluation completed."
    @info "Relative L2 mean=$relative_l2_mean"
    @info "Initial sensitivity error=$initial_sensitivity"
    @info "Shift equivariance error=$shift_error"
    @info "Boundary error=$boundary_err"
    
    return Dict(
        :dataset => config[:name],
        :model => model_name,
        :use_coord => use_coord,
        :relative_l2_mean => relative_l2_mean,
        :relative_l2_std => relative_l2_std,
        :initial_sensitivity_error => initial_sensitivity,
        :shift_equivariance_error => shift_error,
        :boundary_error => boundary_err,
        :coordinate_benefit => missing,
        :parameter_count => missing,
        :train_time_seconds => missing,
    )
end

end # module Evaluate