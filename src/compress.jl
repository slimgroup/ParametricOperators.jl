export ht_compress, compression_ratio, relative_error

"""
    ht_compress(A::AbstractArray{T,N}; max_rank, tree) where {T,N}

Compress a dense tensor to HT format.

Wrapper around `htrunc` that constructs a default tree if not provided.
"""
function ht_compress(A::AbstractArray{T,N}; max_rank::Int,
                     tree::Union{Nothing,DimensionTree}=nothing) where {T,N}
    X = htrunc(A; max_rank=max_rank, tree=tree)
    return orthogonalize(X)
end

"""
    compression_ratio(X::HTTensor)

Compute the compression ratio: full storage / HT storage.
A ratio > 1 means the HT format uses less storage.
"""
function compression_ratio(X::HTTensor)
    full_storage = prod(X.sz)
    ht_storage = sum(length(f) for f in X.fmat) + sum(length(b) for b in X.trten)
    return full_storage / ht_storage
end

"""
    relative_error(X::HTTensor{T}, A::AbstractArray{T}) where T

Compute the relative error ||full(X) - A|| / ||A||.
"""
function relative_error(X::HTTensor{T}, A::AbstractArray{T}) where T
    return norm(full(X) - A) / norm(A)
end
