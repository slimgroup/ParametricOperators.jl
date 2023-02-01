export ParBias

"""
Elementwise bias operator.
"""
struct ParBias{T} <: ParNonLinearOperator{T,T,Parametric,External}
    n::Int
    dim::Int
    shape::Vector{Int}
    ParBias(T::Type, n::Int; dim=1, shape=[n]) = new{T}(n, dim, shape)
    ParBias(n::Int; dim=1, shape=[n]) = ParBias(Float64, n; dim=dim, shape=shape)
end

Domain(A::ParBias) = prod(A.shape)
Range(A::ParBias) = Domain(A)
nparams(A::ParBias) = 1
function init(A::ParBias{T}) where {T}
    s = ones(Int, length(A.shape))
    s[A.dim] = A.n
    return [zeros(T, s...)]
end

(A::ParParameterized{T,T,NonLinear,ParBias{T},V})(x::X) where {T,V,X<:AbstractVector{T}} = vec(A.params[1] .+ reshape(x, A.op.shape...))
function (A::ParParameterized{T,T,NonLinear,ParBias{T},V})(x::X) where {T,V,X<:AbstractMatrix{T}}
    batch_size = size(x)[2]
    y = A.params[1] .+ reshape(x, A.op.shape..., batch_size)
    return reshape(y, Domain(A.op), batch_size)
end