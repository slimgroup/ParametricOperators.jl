export htrunc, recompress

"""
    htrunc(A::AbstractArray{T,N}; max_rank, tree) where {T,N}

Truncate a full dense tensor to HT format using HT-SVD.

For each node in the dimension tree, computes the truncated SVD of the
appropriate matricization to obtain frame matrices. Then extracts transfer
tensors by projecting into the children subspaces.

# Arguments
- `A`: dense tensor
- `max_rank`: maximum rank at any non-root node
- `tree`: dimension tree (default: balanced binary tree)

# Returns
- `HTTensor` approximating `A`
"""
function htrunc(A::AbstractArray{T,N}; max_rank::Int,
                tree::Union{Nothing,DimensionTree}=nothing) where {T,N}
    if isnothing(tree)
        tree = DimensionTree(N)
    end
    @assert tree.d == N

    sz = size(A)

    # Step 1: Compute frame matrices at each non-root node via SVD of matricizations
    frames = Dict{Int, Matrix{T}}()

    for node in postorder(tree)
        if is_root(tree, node)
            continue
        end

        node_dims = dims_at_node(tree, node)
        A_mat = _node_matricize(A, node_dims)
        F = svd(A_mat)

        # Determine rank: min of max_rank and number of significant singular values
        k = min(max_rank, length(F.S), size(A_mat, 1), size(A_mat, 2))
        k = max(k, 1)
        frames[node] = F.U[:, 1:k]
    end

    # Step 2: Build leaf matrices from frames
    fmat = Vector{Matrix{T}}(undef, tree.d)
    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        fmat[dim] = frames[node]
    end

    # Step 3: Build transfer tensors
    trten = Vector{Array{T,3}}(undef, num_internal(tree))

    for (idx, node) in enumerate(internal_nodes(tree))
        l = left_child(tree, node)
        r = right_child(tree, node)

        Ul = frames[l]
        Ur = frames[r]
        kl = size(Ul, 2)
        kr = size(Ur, 2)

        left_dims = dims_at_node(tree, l)
        right_dims = dims_at_node(tree, r)
        prod_left = prod(sz[i] for i in left_dims)
        prod_right = prod(sz[i] for i in right_dims)

        if is_root(tree, node)
            # Root: B = Ul' * A_lr * Ur, shape (kl, kr)
            A_lr = _bipartite_matricize(A, left_dims, right_dims)
            B_2d = Ul' * A_lr * Ur
            trten[idx] = reshape(B_2d, kl, kr, 1)
        else
            # Non-root: B_mat[:, j] = vec(Ul' * reshape(U_t[:, j], prod_l, prod_r) * Ur)
            U_t = frames[node]
            kp = size(U_t, 2)
            B_mat = zeros(T, kl * kr, kp)

            for j in 1:kp
                u_col = U_t[:, j]
                u_reshaped = reshape(u_col, prod_left, prod_right)
                B_mat[:, j] = vec(Ul' * u_reshaped * Ur)
            end

            trten[idx] = trten2ten(B_mat, kl, kr)
        end
    end

    HTTensor(tree, trten, fmat, sz; isorth=false)
end

"""
    _node_matricize(A::AbstractArray, row_dims::Vector{Int})

Matricize tensor A with `row_dims` as row dimensions and the complement as columns.
Row dimensions are ordered as given (lower dims vary fastest in flattening).
"""
function _node_matricize(A::AbstractArray{T,N}, row_dims::Vector{Int}) where {T,N}
    col_dims = setdiff(1:N, row_dims)
    perm = vcat(row_dims, col_dims)
    A_perm = permutedims(A, perm)
    row_size = prod(size(A, d) for d in row_dims)
    col_size = prod(size(A, d) for d in col_dims)
    return reshape(A_perm, row_size, col_size)
end

"""
    _bipartite_matricize(A::AbstractArray, left_dims, right_dims)

Matricize tensor A into (prod of left dim sizes, prod of right dim sizes),
with left_dims as rows (varying fastest first) and right_dims as columns.
"""
function _bipartite_matricize(A::AbstractArray{T,N}, left_dims::Vector{Int},
                              right_dims::Vector{Int}) where {T,N}
    perm = vcat(left_dims, right_dims)
    A_perm = permutedims(A, perm)
    left_size = prod(size(A, d) for d in left_dims)
    right_size = prod(size(A, d) for d in right_dims)
    return reshape(A_perm, left_size, right_size)
end

"""
    recompress(X::HTTensor{T}; max_rank::Int) where T

Recompress an HT tensor to lower rank by converting to full and re-truncating.
"""
function recompress(X::HTTensor{T}; max_rank::Int) where T
    return htrunc(full(X); max_rank=max_rank, tree=X.tree)
end
