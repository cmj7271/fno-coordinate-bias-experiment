module Config

using TOML
using LinearAlgebra
using Statistics
using JLD2

# ==============================================================================
# Config.jl: Configuration Management
# Handles loading, merging, and validating experiment configuration from TOML files.
# ==============================================================================

"""
    load_config(config_path)

Loads and returns a dictionary containing all settings from a specified TOML file.

# Arguments
- `config_path`: File path to the TOML configuration file.

# Returns
Dict of loaded configuration values.
"""
function load_config(config_path::String)
    try
        # Read the TOML file and parse it into a Julia Dictionary
        config = TOML.parsefile(config_path)
        
        # Simple recursive function to convert TOML structure to standard Julia Dict
        function to_julia_dict(t)
            if isa(t, Dict)
                d = Dict{Symbol, Any}()
                for (k, v) in pairs(t)
                    d[Symbol(k)] = to_julia_dict(v)
                end
                return d
            elseif isa(t, Array)
                # Handle arrays of tables (if necessary, but usually not for this setup)
                return [to_julia_dict(e) for e in t]
            else
                return t
            end
        end
        
        return to_julia_dict(config)
        
    catch e
        @error "Failed to load config from $config_path: $e"
        return nothing
    end
end

"""
    merge_configs(base_config, overrides...)

Merges multiple configuration dictionaries. Later dictionaries override earlier ones.
"""
function merge_configs(base_config::Dict, overrides...)
    config = deepcopy(base_config)
    for override in overrides
        for (k, v) in pairs(override)
            if haskey(config, k) && isa(config[k], Dict) && isa(v, Dict)
                # If both are dictionaries, recursively merge them
                config[k] = merge_configs(config[k], v)
            else
                # Otherwise, overwrite or set the value
                config[k] = v
            end
        end
    end
    return config
end

end # module Config