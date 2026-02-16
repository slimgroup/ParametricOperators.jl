export ParSampling

"""
    ParSampling{T,N}

NonParametric operator for arbitrary index sampling of a vectorized tensor.

Domain = prod(sz), Range = length(indices).
Forward: reshape input to tensor, extract entries at CartesianIndex positions.
Adjoint: scatter values back to zero vector.
"""
struct ParSampling{T,N} <: ParLinearOperator{T,T,NonParametric,External}
    indices::Vector{CartesianIndex{N}}
    sz::NTuple{N,Int}
    n::Int   # prod(sz) = Domain
    m::Int   # length(indices) = Range
    function ParSampling(::Type{T}, indices::Vector{CartesianIndex{N}},
                         sz::NTuple{N,Int}) where {T,N}
        new{T,N}(indices, sz, prod(sz), length(indices))
    end
end

Domain(A::ParSampling) = A.n
Range(A::ParSampling) = A.m

# Forward: extract entries at observed indices
function (A::ParSampling{T,N})(x::AbstractVector{T}) where {T,N}
    X = reshape(x, A.sz)
    return T[X[idx] for idx in A.indices]
end

# Adjoint: scatter values back to zero vector
function (A::ParAdjoint{T,T,NonParametric,ParSampling{T,N}})(
    y::AbstractVector{T}) where {T,N}
    op = A.op
    x = zeros(T, op.sz)
    for (val, idx) in zip(y, op.indices)
        x[idx] += val
    end
    return vec(x)
end

# rrule for forward
function ChainRulesCore.rrule(A::ParSampling{T,N},
                              x::AbstractVector{T}) where {T,N}
    y = A(x)
    function parsampling_pullback(dy)
        return (NoTangent(), A'(collect(dy)))
    end
    return y, parsampling_pullback
end

# rrule for adjoint
function ChainRulesCore.rrule(A::ParAdjoint{T,T,NonParametric,ParSampling{T,N}},
                              y::AbstractVector{T}) where {T,N}
    x = A(y)
    function parsampling_adj_pullback(dx)
        return (NoTangent(), A.op(collect(dx)))
    end
    return x, parsampling_adj_pullback
end
