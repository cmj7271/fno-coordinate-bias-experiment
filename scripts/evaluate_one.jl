using FNOExperiment.Evaluate
using FNOExperiment.Config

# ==============================================================================
# scripts/evaluate_one.jl: Script to run the evaluation pipeline.
#
# Usage:
#   julia --project=. scripts/evaluate_one.jl configs/periodic_heat.toml fno false
#   julia --project=. scripts/evaluate_one.jl configs/periodic_heat.toml fno true
#   julia --project=. scripts/evaluate_one.jl configs/periodic_heat.toml deeponet false
# ==============================================================================

function value_to_string(x)
    if x === missing
        return ""
    else
        return string(x)
    end
end

function append_summary_csv(summary_path::String, row::Dict)
    mkpath(dirname(summary_path))

    header = [
        :dataset,
        :model,
        :use_coord,
        :relative_l2_mean,
        :relative_l2_std,
        :initial_sensitivity_error,
        :shift_equivariance_error,
        :boundary_error,
        :parameter_count,
        :train_time_seconds,
    ]

    file_exists = isfile(summary_path)

    open(summary_path, "a") do io
        if !file_exists
            println(io, join(string.(header), ","))
        end

        values = [value_to_string(get(row, key, missing)) for key in header]
        println(io, join(values, ","))
    end
end

function main(config_path::String, model_name_str::String, use_coord_str::String)
    @info "==============================================="
    @info "  STARTING EVALUATION PIPELINE"
    @info "==============================================="

    config = Config.load_config(config_path)
    if config === nothing
        @error "Failed to load configuration. Exiting."
        return false
    end

    model_name = Symbol(model_name_str)
    use_coord = lowercase(use_coord_str) == "true"

    if !(model_name in (:fno, :deeponet))
        @error "Unsupported model name: $model_name. Use :fno or :deeponet."
        return false
    end

    @info "Configuration loaded successfully from $config_path."
    @info "Evaluating model=$model_name, use_coord=$use_coord"

    try
        results = Evaluate.evaluate_dataset(
            config_path;
            model_name=model_name,
            use_coord=use_coord,
        )

        @info "================================================="
        @info "✅ EVALUATION SUCCESSFUL."
        @info "Results Summary:"
        @info "  Dataset: $(config[:name])"
        @info "  Model: $model_name"
        @info "  Use coord: $use_coord"
        @info "  Mean Relative L2 Error: $(results[:relative_l2_mean])"
        @info "  Std Relative L2 Error: $(get(results, :relative_l2_std, missing))"
        @info "  Initial Sensitivity Error: $(get(results, :initial_sensitivity_error, missing))"
        @info "  Shift Equivariance Error: $(get(results, :shift_equivariance_error, missing))"
        @info "================================================="

        summary_row = Dict(
            :dataset => config[:name],
            :model => String(model_name),
            :use_coord => use_coord,
            :relative_l2_mean => results[:relative_l2_mean],
            :relative_l2_std => get(results, :relative_l2_std, missing),
            :initial_sensitivity_error => get(results, :initial_sensitivity_error, missing),
            :shift_equivariance_error => get(results, :shift_equivariance_error, missing),
            :boundary_error => get(results, :boundary_error, missing),
            :parameter_count => get(results, :parameter_count, missing),
            :train_time_seconds => get(results, :train_time_seconds, missing),
        )

        summary_path = joinpath("results", "tables", "summary.csv")
        append_summary_csv(summary_path, summary_row)

        @info "Summary row appended to: $summary_path"

        return true
    catch e
        @error "Evaluation failed due to critical error: $e"
        return false
    end
end

if length(ARGS) < 3
    @warn "Usage: julia --project=. scripts/evaluate_one.jl <config_path> <model_name> <use_coord>"
    @warn "Example: julia --project=. scripts/evaluate_one.jl configs/periodic_heat.toml fno false"
else
    main(ARGS[1], ARGS[2], ARGS[3])
end