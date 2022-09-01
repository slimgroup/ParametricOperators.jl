using Flux
using LinearAlgebra
using MLDatasets
using Tao

W = MatrixOperator{Float32}(3, 4)
b = BiasOperator(W)
σ = x -> sigmoid.(x)
F = σ ∘ (W + b)
θ = init(F)
x = rand(DDT(F), Domain(F))