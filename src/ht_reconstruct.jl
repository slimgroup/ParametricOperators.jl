export ht_reconstruct_vec

"""
    ht_reconstruct_vec(tree::DimensionTree, fmat::Vector{Matrix{T}},
                       trten::Vector{Array{T,3}}, sz::NTuple{N,Int}) -> Vector{T}

Reconstruct the full tensor from HT components and return as a flat vector.
This is equivalent to `vec(full(HTTensor(...)))` but works with raw arrays
so it can be differentiated through by ChainRulesCore.

Arguments:
- `tree`: dimension tree
- `fmat`: leaf matrices indexed by dimension (1..d)
- `trten`: transfer tensors indexed by internal node order
- `sz`: full tensor dimensions
"""
function ht_reconstruct_vec(tree::DimensionTree, fmat::Vector{Matrix{T}},
                            trten::Vector{Array{T,3}},
                            sz::NTuple{N,Int}) where {T<:Number, N}
    U = Dict{Int, Matrix{T}}()

    for node in leaves(tree)
        dim = leaf_dim(tree, node)
        U[node] = fmat[dim]
    end

    for node in postorder(tree)
        is_leaf(tree, node) && continue
        l = left_child(tree, node)
        r = right_child(tree, node)
        idx = node2ind(tree, node)
        B_mat = trten2mat(trten[idx])
        K = kron(U[r], U[l])
        U[node] = K * B_mat
    end

    root_mat = U[root(tree)]
    return root_mat[:, 1]
end

function ChainRulesCore.rrule(::typeof(ht_reconstruct_vec),
                              tree::DimensionTree,
                              fmat::Vector{Matrix{T}},
                              trten::Vector{Array{T,3}},
                              sz::NTuple{N,Int}) where {T<:Number, N}
    # Forward pass
    y = ht_reconstruct_vec(tree, fmat, trten, sz)

    function ht_reconstruct_vec_pullback(dy)
        grad_dense = reshape(collect(dy), sz)

        # Build a temporary HTTensor for gradient computation
        X_tmp = HTTensor(tree, trten, fmat, sz; isorth=false)
        X_full = full(X_tmp)
        Phi = _compute_all_bases(X_tmp)

        d = tree.d

        # Leaf gradients
        dfmat = Vector{Matrix{T}}(undef, d)
        for node in leaves(tree)
            dim = leaf_dim(tree, node)
            V_i = _leaf_complement(X_tmp, dim, Phi)
            G_mat = _mode_unfold(grad_dense, dim)
            dfmat[dim] = G_mat * V_i
        end

        # Transfer tensor gradients
        dtrten = Vector{Array{T,3}}(undef, num_internal(tree))
        for (idx, node) in enumerate(internal_nodes(tree))
            l = left_child(tree, node)
            r = right_child(tree, node)
            kl = size(trten[idx], 1)
            kr = size(trten[idx], 2)
            kp = size(trten[idx], 3)
            Phi_l = Phi[l]
            Phi_r = Phi[r]

            if is_root(tree, node)
                left_dims = sort(dims_at_node(tree, l))
                right_dims = sort(dims_at_node(tree, r))
                G_bip = _bipartite_unfold(grad_dense, left_dims, right_dims)
                grad_B = Phi_l' * G_bip * Phi_r
                dtrten[idx] = reshape(grad_B, kl, kr, 1)
            else
                node_dims = sort(dims_at_node(tree, node))
                G_mat_node = _node_unfold(grad_dense, node_dims)
                X_mat = _node_unfold(X_full, node_dims)
                Phi_t = Phi[node]
                C_t = X_mat' * Phi_t

                projected = G_mat_node * C_t
                left_dims = sort(dims_at_node(tree, l))
                right_dims = sort(dims_at_node(tree, r))

                grad_B_mat = zeros(T, kl * kr, kp)
                for j in 1:kp
                    col = projected[:, j]
                    col_mat = _bipartite_reshape(col, left_dims, right_dims, node_dims, sz)
                    grad_B_mat[:, j] = vec(Phi_l' * col_mat * Phi_r)
                end
                dtrten[idx] = trten2ten(grad_B_mat, kl, kr)
            end
        end

        return (NoTangent(), NoTangent(), dfmat, dtrten, NoTangent())
    end

    return y, ht_reconstruct_vec_pullback
end
