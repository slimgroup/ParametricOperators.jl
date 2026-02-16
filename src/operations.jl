export full, innerprod, norm_ht, gramians, plus, scale, ttm

"""
    full(X::HTTensor{T}) where T

Reconstruct the full dense tensor from HT format.

Traverses the dimension tree bottom-up. At each internal node, computes:
    U_node = kron(U_right, U_left) * trten2mat(B_node)

Returns an Array of the original tensor dimensions.
"""
function full(X::HTTensor{T}) where T
    tree = X.tree

    # Store the subspace matrix at each node
    U = Dict{Int, Matrix{T}}()

    # Initialize leaves
    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        U[node] = X.fmat[dim]
    end

    # Process internal nodes bottom-up (postorder, skip leaves)
    for node in postorder(tree)
        if is_leaf(tree, node)
            continue
        end

        l = left_child(tree, node)
        r = right_child(tree, node)
        idx = node2ind(tree, node)

        B_mat = trten2mat(X.trten[idx])  # (kl*kr, kp)
        K = kron(U[r], U[l])             # (nl*nr, kl*kr) -- note: Julia kron convention
        U[node] = K * B_mat              # (nl*nr, kp)
    end

    # Root node: U[root] is (prod(sz), 1) for rank-1 root
    root_mat = U[root(tree)]
    return Array(reshape(root_mat[:, 1], X.sz))
end

"""
    gramians(X::HTTensor{T}) where T

Compute the Gramian matrices G_t = U_t' * U_t for each node t.

For leaves: G_leaf = fmat[dim]' * fmat[dim]
For internal nodes (bottom-up):
    G_node = trten2mat(B)' * kron(G_right, G_left) * trten2mat(B)

Returns Dict{Int, Matrix{T}} mapping node index to Gramian.
"""
function gramians(X::HTTensor{T}) where T
    tree = X.tree
    G = Dict{Int, Matrix{T}}()

    # Leaves
    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        U = X.fmat[dim]
        G[node] = U' * U
    end

    # Internal nodes bottom-up
    for node in postorder(tree)
        if is_leaf(tree, node)
            continue
        end

        l = left_child(tree, node)
        r = right_child(tree, node)
        idx = node2ind(tree, node)

        B_mat = trten2mat(X.trten[idx])  # (kl*kr, kp)
        Gkron = kron(G[r], G[l])         # (kl*kr, kl*kr)
        G[node] = B_mat' * Gkron * B_mat # (kp, kp)
    end

    return G
end

"""
    innerprod(X::HTTensor{T}, Y::HTTensor{T}) where T

Compute the inner product <X, Y> between two HT tensors with the same tree.

Uses the Gramian-like approach:
- At leaves: M[leaf] = X.fmat[dim]' * Y.fmat[dim]
- At internal nodes: M[node] = trten2mat(Bx)' * kron(M[right], M[left]) * trten2mat(By)

Returns a scalar.
"""
function innerprod(X::HTTensor{T}, Y::HTTensor{T}) where T
    @assert X.tree.d == Y.tree.d "Tensors must have same number of dimensions"
    tree = X.tree

    M = Dict{Int, Matrix{T}}()

    # Leaves
    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        M[node] = X.fmat[dim]' * Y.fmat[dim]
    end

    # Internal nodes bottom-up
    for node in postorder(tree)
        if is_leaf(tree, node)
            continue
        end

        l = left_child(tree, node)
        r = right_child(tree, node)
        idx_x = node2ind(tree, node)
        idx_y = node2ind(Y.tree, node)

        Bx_mat = trten2mat(X.trten[idx_x])
        By_mat = trten2mat(Y.trten[idx_y])
        Mkron = kron(M[r], M[l])
        M[node] = Bx_mat' * Mkron * By_mat
    end

    return M[root(tree)][1, 1]
end

"""
    norm_ht(X::HTTensor{T}) where T

Compute the norm of an HT tensor.

If the tensor is orthogonalized, the norm equals the Frobenius norm of the
root transfer tensor. Otherwise, compute via inner product.
"""
function norm_ht(X::HTTensor{T}) where T
    if X.isorth
        return norm(X.trten[node2ind(X.tree, root(X.tree))])
    else
        val = innerprod(X, X)
        return sqrt(max(real(val), zero(real(T))))
    end
end

"""
    plus(X::HTTensor{T}, Y::HTTensor{T}) where T

Add two HT tensors with the same dimension tree.

Concatenates leaf matrices horizontally, block-diagonalizes transfer tensors,
and assembles a structured root transfer tensor.
"""
function plus(X::HTTensor{T}, Y::HTTensor{T}) where T
    @assert X.sz == Y.sz "Tensor sizes must match"
    tree = X.tree

    # New leaf matrices: horizontal concatenation
    new_fmat = Vector{Matrix{T}}(undef, tree.d)
    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        new_fmat[dim] = hcat(X.fmat[dim], Y.fmat[dim])
    end

    # New transfer tensors: block diagonal for non-root, structured for root
    new_trten = Vector{Array{T,3}}(undef, num_internal(tree))
    for (idx, node) in enumerate(internal_nodes(tree))
        Bx = X.trten[idx]
        By = Y.trten[idx]
        kxl, kxr, kxp = size(Bx)
        kyl, kyr, kyp = size(By)

        if is_root(tree, node)
            # Root: assemble [Bx; By] pattern reshaped appropriately
            # Result has rank 1 at root: sum of X and Y
            new_B = zeros(T, kxl + kyl, kxr + kyr, 1)
            new_B[1:kxl, 1:kxr, 1] = Bx[:, :, 1]
            new_B[kxl+1:end, kxr+1:end, 1] = By[:, :, 1]
            new_trten[idx] = new_B
        else
            # Non-root: block diagonal
            new_B = zeros(T, kxl + kyl, kxr + kyr, kxp + kyp)
            new_B[1:kxl, 1:kxr, 1:kxp] = Bx
            new_B[kxl+1:end, kxr+1:end, kxp+1:end] = By
            new_trten[idx] = new_B
        end
    end

    HTTensor(tree, new_trten, new_fmat, X.sz; isorth=false)
end

"""
    scale(alpha::Number, X::HTTensor{T}) where T

Scale an HT tensor by a scalar. Only modifies the root transfer tensor.
"""
function scale(alpha::Number, X::HTTensor{T}) where T
    new_trten = copy.(X.trten)
    root_idx = node2ind(X.tree, root(X.tree))
    new_trten[root_idx] = T(alpha) .* new_trten[root_idx]
    HTTensor(X.tree, new_trten, copy.(X.fmat), X.sz; isorth=X.isorth)
end

"""
    ttm(X::HTTensor{T}, M::Matrix{T}, mode::Int) where T

n-mode matrix product: multiply the leaf matrix at `mode` by M.
Only modifies fmat[mode], tree and transfer tensors unchanged.
"""
function ttm(X::HTTensor{T}, M::AbstractMatrix, mode::Int) where T
    @assert 1 <= mode <= X.tree.d
    new_fmat = copy.(X.fmat)
    new_fmat[mode] = M * X.fmat[mode]
    new_sz = ntuple(i -> i == mode ? size(M, 1) : X.sz[i], X.tree.d)
    HTTensor(X.tree, copy.(X.trten), new_fmat, new_sz; isorth=false)
end

Base.:+(X::HTTensor, Y::HTTensor) = plus(X, Y)
Base.:-(X::HTTensor, Y::HTTensor) = plus(X, scale(-1, Y))
Base.:*(alpha::Number, X::HTTensor) = scale(alpha, X)
Base.:*(X::HTTensor, alpha::Number) = scale(alpha, X)
