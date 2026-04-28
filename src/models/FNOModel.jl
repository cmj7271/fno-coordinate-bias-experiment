# ==============================================================================
# FNOModel.jl: FNO model implementation using NeuralOperators.jl.
# ==============================================================================

module FNOModel

using NeuralOperators
using NNlib: gelu

function get_fno_activation(name)
    if name == :gelu || name == "gelu"
        return gelu
    else
        error("Unsupported FNO activation: $name. MVP currently supports only gelu.")
    end
end

"""
    build_fno_model(config; in_channels, out_channels=1)

Builds a 1D FNO model wrapper using NeuralOperators.jl.

# Arguments
- `config`: Configuration dictionary containing model hyperparameters.
- `in_channels`: Number of input channels (c).
- `out_channels`: Number of output channels (must be 1 for PDE solutions).

# Returns
A function/closure representing the trained FNO model.
"""
function build_fno_model(config::Dict{Symbol, Any}; in_channels::Int, out_channels::Int=1)
    # Hyperparameters from config
    width = config[:width]
    modes = config[:modes]
    layers = config[:layers]
    activation = get_fno_activation(config[:activation])
    
    chs = Tuple([in_channels; fill(width, layers); out_channels])

    model = FourierNeuralOperator(
        activation;
        chs=chs,
        modes=(modes,),
    )

    return model
end

end # module FNOModel