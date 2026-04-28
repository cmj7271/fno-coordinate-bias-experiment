# ==============================================================================
# DatasetIO.jl: Handles loading, preprocessing, and artifact management.
# ==============================================================================

module DatasetIO

using JLD2
using LinearAlgebra
using Statistics

using ..Config

function checkpoint_path(config::Dict, model_name::Symbol, use_coord::Bool)
    dataset_name = String(config[:name])
    coord_tag = use_coord ? "coord" : "no_coord"

    return joinpath(
        "checkpoints",
        String(model_name),
        "$(dataset_name)_$(coord_tag).jld2",
    )
end

function dataset_path(config::Dict)
    dataset_name = String(config[:name])
    return joinpath("data", "raw", "$(dataset_name).jld2")
end

"""
    load_dataset(config_path)

Loads the raw (X, Y) dataset from the specified JLD2 file path.

# Arguments
- `config_path`: The TOML file path specifying the dataset type.

# Returns
Dict containing the loaded data X and Y arrays.
"""
function load_dataset(config_path::String)
    # NOTE: In a real scenario, data_path should be constructed from config_path/dataset_name.
    # For the MVP, we assume a single, central data directory artifact.
    # The actual path must be adjusted by the user based on the final artifact location.
    config = Config.load_config(config_path)
    dataset_name = String(config[:name])
    data_path = joinpath("data", "raw", "$(dataset_name).jld2")
    
    try
        # Load the data using JLD2
        data = JLD2.load(data_path, "data")
        
        # Verify required keys exist
        validate_xy(data[:X], data[:Y])
        
        return Dict(:X => data[:X], :Y => data[:Y])
    catch e
        @error "Failed to load dataset from $data_path. Check if data has been generated first. Error: $e"
        rethrow(e)
    end
end

"""
    add_coordinate_channel(X_original, grid::Vector{Float32})

Adds the absolute spatial coordinate 'x' as a second channel to the input feature vector X.
This is necessary for comparing FNO (with/without coordinate) fairly.

# Arguments
- `X_original`: Input feature array (nx, c, samples).
- `grid`: The spatial grid vector.

# Returns
Array{Float32, 3} with shape (nx, c+1, samples).
"""
function add_coordinate_channel(X_original::Array{Float32}, grid::Vector{Float32})
    N, C, S = size(X_original)
    
    # Create the coordinate channel array: (N, 1, S)
    # We repeat the grid vector 'S' times along the sample dimension.
    # This assumes the grid is the same for all samples, which is true for PDE operator learning.
    coord_channel_reshaped = reshape(repeat(grid, 1, S), N, 1, S)
    
    # Concatenate the coordinate channel along the channel dimension (dim=2)
    X_with_coord = hcat(X_original, coord_channel_reshaped)
    
    return X_with_coord
end

"""
    load_and_preprocess_dataset(config_path::String, add_coord::Bool, grid::Vector{Float32})

Loads the dataset and performs optional preprocessing (coordinate channel addition).

# Arguments
- `config_path`: The TOML file path specifying the dataset type.
- `add_coord`: Boolean indicating whether to add the coordinate channel.
- `grid`: The spatial grid vector (used for coordinate channel construction).

# Returns
Dict containing the preprocessed data X and Y arrays.
"""
function load_and_preprocess_dataset(config_path::String, add_coord::Bool, grid::Vector{Float32})
    @info "--- Preprocessing Dataset: $config_path ---"
    raw_data = load_dataset(config_path)
    X_raw = raw_data[:X]
    Y = raw_data[:Y]
    
    if add_coord
        @info "Adding coordinate channel to input X."
        X = add_coordinate_channel(X_raw, grid)
    else
        X = X_raw
    end
    
    @info "Data successfully preprocessed. X shape: $(size(X)), Y shape: $(size(Y))"
    return Dict(:X => X, :Y => Y)
end

function validate_xy(X, Y)
    @assert ndims(X) == 3 "X must be 3D: (nx, channels, samples). Got $(size(X))."
    @assert ndims(Y) == 3 "Y must be 3D: (nx, 1, samples). Got $(size(Y))."
    @assert size(X, 1) == size(Y, 1) "X/Y nx mismatch: X=$(size(X)), Y=$(size(Y))."
    @assert size(Y, 2) == 1 "Y must have one output channel. Got $(size(Y, 2))."
    @assert size(X, 3) == size(Y, 3) "X/Y sample count mismatch: X=$(size(X)), Y=$(size(Y))."
    @assert all(isfinite, X) "X contains NaN or Inf."
    @assert all(isfinite, Y) "Y contains NaN or Inf."
    return nothing
end

end # module DatasetIO
