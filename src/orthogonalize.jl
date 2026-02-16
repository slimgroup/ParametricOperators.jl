export orthogonalize, orthogonalize!

"""
    orthogonalize(X::HTTensor{T}) where T

Return an orthogonalized copy of the HT tensor.

QR-decomposes each leaf matrix, propagates R factors upward through
transfer tensors via kron(R_right, R_left) * trten2mat(B), then QR
at each internal node, continuing up to the root.

After orthogonalization:
- All leaf matrices have orthonormal columns
- All transfer tensors (except root) satisfy: trten2mat(B)'*trten2mat(B) = I
- norm(X) == norm(root_transfer_tensor)
"""
function orthogonalize(X::HTTensor{T}) where T
    if X.isorth
        return X
    end

    tree = X.tree
    new_fmat = copy.(X.fmat)
    new_trten = copy.(X.trten)

    # R factors to propagate upward, keyed by node index
    R_factors = Dict{Int, Matrix{T}}()

    # Step 1: QR at leaves
    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        F = qr(new_fmat[dim])
        Q = Matrix(F.Q)
        R = Matrix(F.R)
        new_fmat[dim] = Q
        R_factors[node] = R
    end

    # Step 2: Process internal nodes bottom-up (postorder), absorb R factors
    for node in postorder(tree)
        if is_leaf(tree, node)
            continue
        end

        l = left_child(tree, node)
        r = right_child(tree, node)
        idx = node2ind(tree, node)

        Rl = R_factors[l]
        Rr = R_factors[r]

        # Absorb R factors: B_new_mat = kron(Rr, Rl) * trten2mat(B)
        B_mat = trten2mat(new_trten[idx])
        Rkron = kron(Rr, Rl)
        B_new_mat = Rkron * B_mat

        if !is_root(tree, node)
            # QR on the reshaped transfer tensor
            kl = size(Rl, 1)
            kr = size(Rr, 1)
            F = qr(B_new_mat)
            Q = Matrix(F.Q)
            R = Matrix(F.R)
            new_trten[idx] = trten2ten(Q, kl, kr)
            R_factors[node] = R
        else
            # At root, just store the result (no further QR needed)
            kl = size(Rl, 1)
            kr = size(Rr, 1)
            new_trten[idx] = trten2ten(B_new_mat, kl, kr)
        end
    end

    HTTensor(tree, new_trten, new_fmat, X.sz; isorth=true)
end

"""
    orthogonalize!(X::HTTensor{T}) where T

In-place orthogonalization of the HT tensor.
"""
function orthogonalize!(X::HTTensor{T}) where T
    Y = orthogonalize(X)
    X.fmat .= Y.fmat
    X.trten .= Y.trten
    X.isorth = true
    return X
end
