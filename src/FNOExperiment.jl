module FNOExperiment

include("Config.jl")
include("RandomFields.jl")

include("pdes/PeriodicHeat.jl")
include("pdes/DirichletHeat.jl")
include("pdes/PeriodicPoisson.jl")
include("pdes/VariablePoisson.jl")

include("models/FNOModel.jl")
include("models/DeepONetModel.jl")

include("DataGenerators.jl")
include("DatasetIO.jl")
include("Metrics.jl")
include("Train.jl")
include("Evaluate.jl")

end