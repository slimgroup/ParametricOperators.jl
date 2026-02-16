export ParHTTucker, to_htparams, from_htparams, to_parht

"""
    ParHTTucker{T,N}

Parametric operator representing HT Tucker tensor reconstruction.

Domain = 1 (scalar input), Range = prod(sz).
Parameters are `(fmat=..., trten=...)` stored in the parameter dict.
Forward: `ht_reconstruct_vec(tree, fmat, trten, sz) * x[1]`
"""
struct ParHTTucker{T,N} <: ParLinearOperator{T,T,Parametric,External}
    tree::DimensionTree
    sz::NTuple{N,Int}
    max_rank::Int
    id::Any
    function ParHTTucker(::Type{T}, tree::DimensionTree, sz::NTuple{N,Int};
                         max_rank::Int=5) where {T,N}
        @assert tree.d == N "Tree dimensions must match tensor order"
        new{T,N}(tree, sz, max_rank, uuid4(Random.GLOBAL_RNG))
    end
end

Domain(::ParHTTucker) = 1
Range(A::ParHTTucker) = prod(A.sz)

function init!(A::ParHTTucker{T,N}, d::Parameters) where {T,N}
    X = randn_httensor(T, A.tree, A.sz, A.max_rank)
    d[A] = (fmat=X.fmat, trten=X.trten)
    return nothing
end

# Forward: ParParameterized wrapping ParHTTucker
function (A::ParParameterized{T,T,Linear,ParHTTucker{T,N},V})(
    x::AbstractVector{T}) where {T,N,V}
    op = A.op
    p = A.params
    v = ht_reconstruct_vec(op.tree, p.fmat, p.trten, op.sz)
    return v .* x[1]
end

# Adjoint forward: ParParameterized wrapping ParAdjoint{ParHTTucker}
function (A::ParParameterized{T,T,Linear,
    ParAdjoint{T,T,Parametric,ParHTTucker{T,N}},V})(
    x::AbstractVector{T}) where {T,N,V}
    op = A.op.op  # unwrap ParAdjoint to get ParHTTucker
    p = A.params[op]
    v = ht_reconstruct_vec(op.tree, p.fmat, p.trten, op.sz)
    return T[dot(v, x)]
end

# --- Conversion helpers ---

"""
    to_htparams(X::HTTensor)

Extract the HT parameters from an HTTensor as a NamedTuple.
"""
function to_htparams(X::HTTensor{T}) where T
    return (fmat=copy.(X.fmat), trten=copy.(X.trten))
end

"""
    from_htparams(A::ParHTTucker{T,N}, params) -> HTTensor{T}

Construct an HTTensor from a ParHTTucker operator and its parameters.
"""
function from_htparams(A::ParHTTucker{T,N}, params) where {T,N}
    return HTTensor(A.tree, params.trten, params.fmat, A.sz; isorth=false)
end

"""
    to_parht(X::HTTensor{T}; max_rank) -> (ParHTTucker, Dict)

Create a ParHTTucker operator and parameter dict from an existing HTTensor.
"""
function to_parht(X::HTTensor{T}; max_rank::Int=maximum(values(hrank(X)))) where T
    N = ndims_ht(X)
    A = ParHTTucker(T, X.tree, X.sz; max_rank=max_rank)
    d = Dict{ParOperator,Any}()
    d[A] = to_htparams(X)
    return A, d
end
