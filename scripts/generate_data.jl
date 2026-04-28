using JLD2

using FNOExperiment.DataGenerators
using FNOExperiment.Config
using FNOExperiment.DatasetIO

# ==============================================================================
# scripts/generate_data.jl: Script to generate all raw dataset artifacts.
# Usage: julia --project=. scripts/generate_data.jl configs/periodic_heat.toml
# ==============================================================================

function main(config_path::String)
    @info "==============================================="
    @info "  STARTING DATA GENERATION PIPELINE"
    @info "==============================================="
    
    # 1. Load Configuration
    config = Config.load_config(config_path)
    if config === nothing
        @error "Failed to load configuration. Exiting."
        return
    end

    @info "Configuration loaded successfully from $config_path."
    
    # 2. Target Artifact Path
    artifact_path = DatasetIO.dataset_path(config)
    mkpath(dirname(artifact_path))

    # 3. Select Generator based on PDE type (Simple switch for MVP)
    dataset = nothing
    
    if startswith(config[:name], "periodic_heat")
        @info "Generating Periodic Heat dataset..."
        dataset = DataGenerators.generate_periodic_heat_dataset(config)
    elseif startswith(config[:name], "dirichlet_heat")
        @info "Generating Dirichlet Heat dataset..."
        dataset = DataGenerators.generate_dirichlet_heat_dataset(config)
    elseif startswith(config[:name], "periodic_poisson")
        @info "Generating Periodic Poisson dataset..."
        dataset = DataGenerators.generate_periodic_poisson_dataset(config)
    elseif startswith(config[:name], "variable_poisson")
        @info "Generating Variable Poisson dataset..."
        dataset = DataGenerators.generate_variable_poisson_dataset(config)
    else
        @error "Unknown PDE type for configuration: $(config[:name])"
        return
    end
    
    # 4. Save Artifact
    try
        JLD2.save(artifact_path, "data", dataset)
        @info "================================================="
        @info "✅ DATA GENERATION SUCCESSFUL."
        @info "Artifact saved to: $artifact_path"
        @info "================================================="
    catch e
        @error "Failed to save artifact to $artifact_path. Check JLD2 usage and memory. Error: $e"
    end
end

# Entry point for calling the script
if length(ARGS) < 1
    @warn "Usage: julia --project=. scripts/generate_data.jl <config_path>"
end

# Run the main function with the first argument as the config path
if length(ARGS) >= 1
    main(ARGS[1])
end
