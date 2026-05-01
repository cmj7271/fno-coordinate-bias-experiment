# ==============================================================================
# Train.jl: Handles the entire training lifecycle for a single PDE/Model.
# ==============================================================================

module Train

using Random
using Lux
using Optimisers
using Zygote
using Statistics
using LinearAlgebra
using JLD2

using ..Config
using ..DatasetIO
using ..FNOModel
using ..DeepONetModel
using ..Metrics

export train_model

function model_forward(model_name::Symbol, model, X, grid_points, ps, st)
    if model_name == :deeponet
        return model((X, grid_points), ps, st)
    elseif model_name == :fno
        return model(X, ps, st)
    else
        error("Unsupported model name: $model_name")
    end
end

function mse_loss(model_name::Symbol, model, X, Y, grid_points, ps, st)
    Y_hat, _ = model_forward(model_name, model, X, grid_points, ps, st)
    return mean(abs2, Y_hat .- Y)
end

function make_batches(indices::Vector{Int}, batch_size::Int)
    batches = Vector{Vector{Int}}()

    for start_idx in 1:batch_size:length(indices)
        stop_idx = min(start_idx + batch_size - 1, length(indices))
        push!(batches, indices[start_idx:stop_idx])
    end

    return batches
end

function evaluate_relative_l2(model_name::Symbol, model, X, Y, grid_points, ps, st)
    Y_hat, st = model_forward(model_name, model, X, grid_points, ps, st)

    @assert size(Y_hat) == size(Y) "Prediction size $(size(Y_hat)) does not match target size $(size(Y))."

    values = Metrics.batch_relative_l2(Y_hat, Y)
    l2_mean = mean(values)
    l2_std = length(values) > 1 ? std(values) : 0.0f0

    return l2_mean, l2_std, st
end

"""
    train_model(config_path::String, model_name::Symbol, use_coord::Bool)

Train one model for one dataset configuration.
"""
function train_model(config_path::String, model_name::Symbol, use_coord::Bool)::Dict
    @info "Starting training for $model_name model using $config_path..."

    # 1. Load config and dataset
    config = Config.load_config(config_path)

    nx = Int(config[:data][:nx])
    grid_points = Float32.(range(0, 1, length=nx))

    dataset = DatasetIO.load_and_preprocess_dataset(config_path, use_coord, grid_points)

    X_train = dataset.X_train
    Y_train = dataset.Y_train
    X_valid = dataset.X_valid
    Y_valid = dataset.Y_valid

    @info "Loaded dataset shapes: X_train=$(size(X_train)), Y_train=$(size(Y_train))"

    @assert ndims(X_train) == 3 "X must be (nx, channels, samples). Got $(size(X_train))."
    @assert ndims(Y_train) == 3 "Y must be (nx, 1, samples). Got $(size(Y_train))."
    @assert size(X_train, 1) == size(Y_train, 1) "X/Y nx mismatch."
    @assert size(Y_train, 2) == 1 "Y must have one output channel."
    @assert size(X_train, 3) == size(Y_train, 3) "X/Y sample count mismatch."

    in_channels = size(X_train, 2)
    n_train = size(X_train, 3)

    # 2. Build model
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
            nx=nx,
            in_channels=in_channels,
        )

    else
        @error "Unsupported model name: $model_name"
        return Dict(
            :status => "Failed",
            :message => "Unsupported model",
        )
    end

    # 3. Initialize parameters/state
    seed = Int(config[:seed])
    rng = Random.Xoshiro(seed)

    ps, st = Lux.setup(rng, model)

    # 4. Train/validation split
    train_idx = collect(1:n_train)

    # 5. Optimizer
    epochs = Int(config[:train][:epochs])
    batch_size = Int(config[:train][:batch_size])
    learning_rate = Float32(config[:train][:learning_rate])

    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    best_valid_l2 = Inf
    best_valid_std = Inf
    best_ps = deepcopy(ps)
    best_st = deepcopy(st)
    best_epoch = 0

    train_losses = Float32[]
    valid_l2_history = Float32[]

    @info "--- Starting actual training for $model_name ---"
    @info "epochs=$epochs, batch_size=$batch_size, lr=$learning_rate, train_samples=$(size(X_train, 3)), valid_samples=$(size(X_valid, 3))"

    # 6. Training loop
    for epoch in 1:epochs
        shuffled_train_idx = shuffle(rng, train_idx)
        batches = make_batches(shuffled_train_idx, batch_size)

        epoch_losses = Float32[]

        for batch_idx in batches
            X_batch = X_train[:, :, batch_idx]
            Y_batch = Y_train[:, :, batch_idx]

            loss_value, back = Zygote.pullback(ps) do current_ps
                mse_loss(model_name, model, X_batch, Y_batch, grid_points, current_ps, st)
            end

            grads = first(back(one(loss_value)))

            opt_state, ps = Optimisers.update!(opt_state, ps, grads)

            push!(epoch_losses, Float32(loss_value))
        end

        epoch_train_loss = mean(epoch_losses)
        push!(train_losses, Float32(epoch_train_loss))

        valid_l2_mean, valid_l2_std, st = evaluate_relative_l2(
            model_name,
            model,
            X_valid,
            Y_valid,
            grid_points,
            ps,
            st,
        )

        push!(valid_l2_history, Float32(valid_l2_mean))

        if valid_l2_mean < best_valid_l2
            best_valid_l2 = valid_l2_mean
            best_valid_std = valid_l2_std
            best_ps = deepcopy(ps)
            best_st = deepcopy(st)
            best_epoch = epoch
        end

        if epoch == 1 || epoch == epochs || epoch % max(1, epochs ÷ 10) == 0
            @info "epoch=$epoch train_mse=$epoch_train_loss valid_relative_l2=$valid_l2_mean"
        end
    end

    # 7. Save best checkpoint
    dataset_name = String(config[:name])
    best_checkpoint_path = DatasetIO.checkpoint_path(config, model_name, use_coord)
    mkpath(dirname(best_checkpoint_path))

    checkpoint = Dict(
        :status => "trained",
        :dataset_name => dataset_name,
        :model_name => model_name,
        :use_coord => use_coord,
        :ps => best_ps,
        :st => best_st,
        :best_epoch => best_epoch,
        :best_valid_l2 => best_valid_l2,
        :best_valid_l2_std => best_valid_std,
        :train_losses => train_losses,
        :valid_l2_history => valid_l2_history,
        :train_shape => size(X_train),
        :valid_shape => size(X_valid),
        :test_shape => size(dataset.X_test),
        :message => "Trained checkpoint with best validation relative L2.",
    )

    JLD2.save(best_checkpoint_path, "checkpoint", checkpoint)

    @info "Training completed. Best checkpoint saved to $best_checkpoint_path"
    @info "Best epoch=$best_epoch, best valid relative L2=$best_valid_l2"

    return Dict(
        :status => "Success",
        :best_checkpoint_path => best_checkpoint_path,
        :metrics => Dict(
            :relative_l2_mean => best_valid_l2,
            :relative_l2_std => best_valid_std,
            :best_epoch => best_epoch,
        ),
    )
end

end # module Train
