using FNOExperiment.Train
using FNOExperiment.Config
using FNOExperiment.DatasetIO

# ==============================================================================
# scripts/train_one.jl: Script to train one model configuration.
# Usage: julia --project=. scripts/train_one.jl <config_path> <model_name> [true|false]
# ==============================================================================

function main(config_path::String, model_name::Symbol, use_coord_str::String)
    @info "==============================================="
    @info "  STARTING TRAINING PIPELINE"
    @info "==============================================="
    
    # 1. Parse Inputs
    use_coord = lowercase(use_coord_str) == "true"
    
    @info "Configuration: $config_path"
    @info "Model: $model_name, Use Coordinate: $use_coord"
    
    # 2. Run Training
    try
        results = Train.train_model(config_path, model_name, use_coord)
        
        @info "================================================="
        @info "✅ TRAINING SUCCESSFUL."
        @info "Status: $(results[:status])"
        @info "Best Checkpoint Saved: $(results[:best_checkpoint_path])"
        @info "================================================="
    catch e
        @error "Training failed due to critical error: $e"
    end
end

# Entry point for calling the script
if length(ARGS) < 3
    @warn "Usage: julia --project=. scripts/train_one.jl <config_path> <model_name> [true|false]"
end

# Run the main function with provided arguments
if length(ARGS) >= 3
    main(ARGS[1], Symbol(ARGS[2]), ARGS[3])
end
