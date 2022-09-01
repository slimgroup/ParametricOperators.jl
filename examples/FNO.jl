using Flux
using LinearAlgebra
using MLDatasets
using Tao
using Zygote

IdentityOperator(T, m::Int64, n::Int64) = FunctionOperator{T,T,Linear}(m, n, identity)