export HTTensor
export hrank, tensor_size, ndims_ht
export trten2mat, trten2ten, randn_httensor

"""
    HTTensor{T}

Hierarchical Tucker tensor.

Stores a tensor in the HT format consisting of:
- `tree`: a `DimensionTree` defining the binary hierarchy over modes
- `trten`: transfer tensors at internal nodes (3D arrays of shape k_left × k_right × k_parent)
- `fmat`: leaf/frame matrices at leaf nodes (2D arrays of shape n_i × k_i)
- `isorth`: whether the tensor is in orthogonal form
- `sz`: the full tensor size (n_1, ..., n_d)

The transfer tensors are indexed by internal node order (via `node2ind`),
and leaf matrices are indexed by leaf dimension number.
"""
mutable struct HTTensor{T<:Number}
    tree::DimensionTree
    trten::Vector{Array{T,3}}     # transfer tensors, indexed by internal node order
    fmat::Vector{Matrix{T}}       # leaf matrices, indexed by dimension (1..d)
    isorth::Bool
    sz::NTuple                    # full tensor dimensions
end

"""
    HTTensor(tree, trten, fmat, sz; isorth=false)

Construct an HT tensor from components.
"""
function HTTensor(tree::DimensionTree, trten::Vector{<:Array{T,3}},
                  fmat::Vector{<:Matrix{T}}, sz::NTuple{N,Int};
                  isorth::Bool=false) where {T<:Number, N}
    @assert length(trten) == num_internal(tree) "Need one transfer tensor per internal node"
    @assert length(fmat) == tree.d "Need one leaf matrix per dimension"
    @assert N == tree.d "Tensor size must match number of dimensions"

    # Validate dimensions
    for (idx, node) in enumerate(internal_nodes(tree))
        l = left_child(tree, node)
        r = right_child(tree, node)

        kl = _rank_at(tree, trten, fmat, l)
        kr = _rank_at(tree, trten, fmat, r)
        kp = is_root(tree, node) ? size(trten[idx], 3) : size(trten[idx], 3)

        @assert size(trten[idx], 1) == kl "Transfer tensor $idx: left rank mismatch ($(size(trten[idx],1)) vs $kl)"
        @assert size(trten[idx], 2) == kr "Transfer tensor $idx: right rank mismatch ($(size(trten[idx],2)) vs $kr)"
    end

    HTTensor{T}(tree, Vector{Array{T,3}}(trten), Vector{Matrix{T}}(fmat), isorth, sz)
end

"""
Helper to get the rank at a node (leaf or internal).
"""
function _rank_at(tree::DimensionTree, trten::Vector{<:Array{<:Number,3}},
                  fmat::Vector{<:Matrix}, node::Int)
    if is_leaf(tree, node)
        dim = leaf_dim(tree, node)
        return size(fmat[dim], 2)
    else
        idx = node2ind(tree, node)
        return size(trten[idx], 3)
    end
end

"""
    hrank(X::HTTensor)

Return the hierarchical rank at each node.
Returns a Dict mapping node index to rank.
"""
function hrank(X::HTTensor)
    ranks = Dict{Int,Int}()
    tree = X.tree

    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        ranks[node] = size(X.fmat[dim], 2)
    end

    for (idx, node) in enumerate(internal_nodes(tree))
        ranks[node] = size(X.trten[idx], 3)
    end

    return ranks
end

"""
    tensor_size(X::HTTensor)

Return the full tensor size.
"""
tensor_size(X::HTTensor) = X.sz

"""
    ndims_ht(X::HTTensor)

Return the number of dimensions of the tensor.
"""
ndims_ht(X::HTTensor) = X.tree.d

"""
    trten2mat(B::Array{T,3}) where T

Reshape a transfer tensor from (k_l, k_r, k_p) to (k_l * k_r, k_p).
"""
function trten2mat(B::Array{T,3}) where T
    kl, kr, kp = size(B)
    return reshape(B, kl * kr, kp)
end

"""
    trten2ten(M::Matrix{T}, kl::Int, kr::Int) where T

Reshape a matrix (k_l * k_r, k_p) back to a transfer tensor (k_l, k_r, k_p).
"""
function trten2ten(M::Matrix{T}, kl::Int, kr::Int) where T
    kp = size(M, 2)
    return reshape(M, kl, kr, kp)
end

function Base.show(io::IO, X::HTTensor{T}) where T
    ranks = hrank(X)
    max_rank = maximum(values(ranks))
    print(io, "HTTensor{$T}(sz=$(X.sz), max_rank=$max_rank, isorth=$(X.isorth))")
end

"""
    Base.eltype(X::HTTensor{T}) where T

Return the element type.
"""
Base.eltype(::HTTensor{T}) where T = T

"""
    randn_httensor(T::Type, tree::DimensionTree, sz::NTuple{N,Int}, ranks::Dict{Int,Int}) where N

Create a random HT tensor with specified ranks at each node.
"""
function randn_httensor(::Type{T}, tree::DimensionTree, sz::NTuple{N,Int},
                        ranks::Dict{Int,Int}) where {T<:Number, N}
    @assert N == tree.d

    fmat = Vector{Matrix{T}}(undef, tree.d)
    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        k = ranks[node]
        fmat[dim] = randn(T, sz[dim], k)
    end

    trten = Vector{Array{T,3}}(undef, num_internal(tree))
    for (idx, node) in enumerate(internal_nodes(tree))
        l = left_child(tree, node)
        r = right_child(tree, node)
        kl = ranks[l]
        kr = ranks[r]
        kp = ranks[node]
        trten[idx] = randn(T, kl, kr, kp)
    end

    HTTensor(tree, trten, fmat, sz; isorth=false)
end

"""
    randn_httensor(T::Type, tree::DimensionTree, sz::NTuple{N,Int}, max_rank::Int) where N

Create a random HT tensor with uniform rank at all nodes.
"""
function randn_httensor(::Type{T}, tree::DimensionTree, sz::NTuple{N,Int},
                        max_rank::Int) where {T<:Number, N}
    ranks = Dict{Int,Int}()
    for node in 1:num_nodes(tree)
        if is_root(tree, node)
            ranks[node] = 1  # Root always has rank 1 for a single tensor
        else
            ranks[node] = max_rank
        end
    end
    randn_httensor(T, tree, sz, ranks)
end
