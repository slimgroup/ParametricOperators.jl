export ht_optimize, HTOptConfig

using LinearAlgebra: norm, dot, I, Diagonal, svd

"""
    HTOptConfig

Configuration for HT tensor optimization.
"""
struct HTOptConfig
    max_iter::Int
    tol::Float64
    max_rank::Int
    method::Symbol       # :sd (steepest descent) or :cg (conjugate gradient)
    verbose::Bool
    line_search_max::Int
    armijo_c::Float64
    armijo_rho::Float64
end

function HTOptConfig(; max_iter::Int=100, tol::Float64=1e-6,
                      max_rank::Int=10, method::Symbol=:cg,
                      verbose::Bool=false, line_search_max::Int=20,
                      armijo_c::Float64=1e-4, armijo_rho::Float64=0.5)
    HTOptConfig(max_iter, tol, max_rank, method, verbose,
                line_search_max, armijo_c, armijo_rho)
end

"""
    SamplingOperator{T,N}

Represents a subsampling/restriction operator for tensor completion.
Stores observed indices and the full tensor size.
"""
struct SamplingOperator{T<:Number, N}
    indices::Vector{CartesianIndex{N}}
    sz::NTuple{N,Int}
end

"""
Apply the sampling operator: extract entries at observed indices.
"""
function (A::SamplingOperator{T,N})(X::AbstractArray{T}) where {T,N}
    return T[X[idx] for idx in A.indices]
end

"""
Apply the adjoint: scatter values back to a zero tensor.
"""
function adjoint_apply(A::SamplingOperator{T,N}, b::Vector{T}) where {T,N}
    Y = zeros(T, A.sz...)
    for (val, idx) in zip(b, A.indices)
        Y[idx] = val
    end
    return Y
end

"""
    ht_optimize(b, indices, sz; config, X0, tree)

Riemannian optimization on the HT manifold for tensor completion.

Minimizes ||A * full(X) - b||^2 where A is a sampling operator,
using steepest descent or conjugate gradient on the HT manifold.

Returns (X, history) where history contains convergence information.
"""
function ht_optimize(b::Vector{T}, indices::Vector{CartesianIndex{N}},
                     sz::NTuple{N,Int};
                     config::HTOptConfig=HTOptConfig(),
                     X0::Union{Nothing,HTTensor{T}}=nothing,
                     tree::Union{Nothing,DimensionTree}=nothing) where {T<:Number, N}
    if isnothing(tree)
        tree = DimensionTree(N)
    end

    if isnothing(X0)
        X0 = randn_httensor(T, tree, sz, config.max_rank)
    end

    A = SamplingOperator{T,N}(indices, sz)

    # Orthogonalize initial guess
    X = orthogonalize(X0)

    # Compute initial residual
    r = A(full(X)) - b
    obj = T(0.5) * dot(r, r)

    history = Dict{String, Vector{Float64}}(
        "objective" => Float64[obj],
        "residual" => Float64[norm(r)],
        "rel_residual" => Float64[norm(r) / norm(b)]
    )

    if config.verbose
        println("Iter 0: obj = $(obj), rel_res = $(norm(r)/norm(b))")
    end

    prev_grad_fmat = nothing
    prev_grad_trten = nothing
    prev_dir_fmat = nothing
    prev_dir_trten = nothing

    for iter in 1:config.max_iter
        # Euclidean gradient: A'(A*full(X) - b)
        grad_dense = adjoint_apply(A, r)

        # Project gradient to tangent space of HT manifold
        grad_fmat, grad_trten = _project_gradient(X, grad_dense)

        # Compute search direction
        if config.method == :sd || iter == 1 || isnothing(prev_dir_fmat)
            dir_fmat = [-g for g in grad_fmat]
            dir_trten = [-g for g in grad_trten]
        else
            # Fletcher-Reeves CG
            num = _tangent_inner(grad_fmat, grad_trten, grad_fmat, grad_trten)
            den = _tangent_inner(prev_grad_fmat, prev_grad_trten, prev_grad_fmat, prev_grad_trten)
            beta = den > 0 ? num / den : zero(T)
            beta = min(beta, T(10))

            dir_fmat = [-g + T(beta) * d for (g, d) in zip(grad_fmat, prev_dir_fmat)]
            dir_trten = [-g + T(beta) * d for (g, d) in zip(grad_trten, prev_dir_trten)]
        end

        # Line search (Armijo backtracking)
        alpha = _line_search(X, dir_fmat, dir_trten, A, b, obj, grad_fmat, grad_trten, config)

        # Retraction: update parameters and re-orthogonalize
        X_new = _retract(X, dir_fmat, dir_trten, alpha)
        X_new = orthogonalize(X_new)

        # Recompress if ranks grew
        max_current_rank = maximum(values(hrank(X_new)))
        if max_current_rank > config.max_rank
            X_new = recompress(X_new; max_rank=config.max_rank)
            X_new = orthogonalize(X_new)
        end

        # Update residual
        r = A(full(X_new)) - b
        obj_new = T(0.5) * dot(r, r)

        push!(history["objective"], obj_new)
        push!(history["residual"], norm(r))
        push!(history["rel_residual"], norm(r) / norm(b))

        if config.verbose
            println("Iter $iter: obj = $(obj_new), rel_res = $(norm(r)/norm(b)), alpha = $alpha")
        end

        # Check convergence
        rel_change = abs(obj - obj_new) / max(abs(obj), one(T))
        if rel_change < config.tol
            if config.verbose
                println("Converged at iteration $iter")
            end
            X = X_new
            break
        end

        prev_grad_fmat = grad_fmat
        prev_grad_trten = grad_trten
        prev_dir_fmat = dir_fmat
        prev_dir_trten = dir_trten

        X = X_new
        obj = obj_new
    end

    return X, history
end

# --- Gradient projection ---

"""
Project the Euclidean gradient onto the tangent space of the HT manifold at X.

Assumes X is orthogonalized.

For leaf i:
    grad_fmat[i] = mode_i_unfold(G) * V_i
    where V_i is the complement basis (computed via partial reconstruction).

For transfer tensor at node t:
    grad_trten[t] = reshape(kron(Phi_r', Phi_l') * grad_(t) * C_t, kl, kr, kp)
    where grad_(t) is the t-matricization of G, and C_t is the complement fiber.
"""
function _project_gradient(X::HTTensor{T}, grad_dense::Array{T}) where T
    tree = X.tree
    d = tree.d
    sz = X.sz

    # Precompute: full tensor and reconstructed bases at all nodes
    X_full = full(X)
    Phi = _compute_all_bases(X)

    # --- Leaf gradients ---
    grad_fmat = Vector{Matrix{T}}(undef, d)
    for node in leaves(tree)
        dim = leaf_dim(tree, node)

        # Compute complement basis via partial reconstruction
        V_i = _leaf_complement(X, dim, Phi)  # (prod_others, k_i)

        # Mode-dim matricization of gradient
        G_mat = _mode_unfold(grad_dense, dim)  # (n_dim, prod_others)

        grad_fmat[dim] = G_mat * V_i
    end

    # --- Transfer tensor gradients ---
    grad_trten = Vector{Array{T,3}}(undef, num_internal(tree))
    for (idx, node) in enumerate(internal_nodes(tree))
        l = left_child(tree, node)
        r = right_child(tree, node)

        kl = size(X.trten[idx], 1)
        kr = size(X.trten[idx], 2)
        kp = size(X.trten[idx], 3)

        Phi_l = Phi[l]
        Phi_r = Phi[r]

        if is_root(tree, node)
            # Root: grad_B = kron(Phi_r', Phi_l') * vec(G) reshaped
            left_dims = sort(dims_at_node(tree, l))
            right_dims = sort(dims_at_node(tree, r))
            G_bip = _bipartite_unfold(grad_dense, left_dims, right_dims)
            grad_B = Phi_l' * G_bip * Phi_r  # (kl, kr)
            grad_trten[idx] = reshape(grad_B, kl, kr, 1)
        else
            # Non-root: use complement fiber
            node_dims = sort(dims_at_node(tree, node))

            # t-matricization of gradient and full tensor
            G_mat = _node_unfold(grad_dense, node_dims)  # (node_prod, comp_prod)
            X_mat = _node_unfold(X_full, node_dims)       # (node_prod, comp_prod)

            # Reconstructed basis at this node
            Phi_t = Phi[node]  # (node_prod, kp)

            # Complement: C_t = X_(t)' * Phi_t / (Phi_t' * Phi_t) ≈ X_(t)' * Phi_t (orthogonal)
            C_t = X_mat' * Phi_t  # (comp_prod, kp)

            # Contract gradient with complement
            projected = G_mat * C_t  # (node_prod, kp)

            # Contract each column with child bases via bipartite reshape
            left_dims = sort(dims_at_node(tree, l))
            right_dims = sort(dims_at_node(tree, r))
            prod_left = prod(sz[i] for i in left_dims)
            prod_right = prod(sz[i] for i in right_dims)

            grad_B_mat = zeros(T, kl * kr, kp)
            for j in 1:kp
                col = projected[:, j]
                # Reshape to (left_prod, right_prod) matching kron(Phi_r, Phi_l) layout
                col_mat = _bipartite_reshape(col, left_dims, right_dims, node_dims, sz)
                grad_B_mat[:, j] = vec(Phi_l' * col_mat * Phi_r)
            end
            grad_trten[idx] = trten2ten(grad_B_mat, kl, kr)
        end
    end

    return grad_fmat, grad_trten
end

"""
Compute reconstructed basis matrices at all nodes (bottom-up).
"""
function _compute_all_bases(X::HTTensor{T}) where T
    tree = X.tree
    Phi = Dict{Int, Matrix{T}}()

    for node in leaves(tree)
        Phi[node] = X.fmat[leaf_dim(tree, node)]
    end

    for node in postorder(tree)
        if is_leaf(tree, node) continue end
        l = left_child(tree, node)
        r = right_child(tree, node)
        idx = node2ind(tree, node)
        B_mat = trten2mat(X.trten[idx])
        K = kron(Phi[r], Phi[l])
        Phi[node] = K * B_mat
    end

    return Phi
end

"""
Compute the complement basis for a leaf dimension.

Replaces fmat[dim] with I(k,k) and reconstructs bottom-up.
Returns V_i of shape (prod_others, k_i).
"""
function _leaf_complement(X::HTTensor{T}, exclude_dim::Int, Phi::Dict{Int, Matrix{T}}) where T
    tree = X.tree
    d = tree.d

    # Build reconstruction with identity at the excluded leaf
    U = Dict{Int, Matrix{T}}()
    for node in leaves(tree)
        dm = leaf_dim(tree, node)
        if dm == exclude_dim
            k = size(X.fmat[dm], 2)
            U[node] = Matrix{T}(I, k, k)
        else
            U[node] = X.fmat[dm]
        end
    end

    for node in postorder(tree)
        if is_leaf(tree, node) continue end
        l = left_child(tree, node)
        r = right_child(tree, node)
        idx = node2ind(tree, node)
        B_mat = trten2mat(X.trten[idx])
        K = kron(U[r], U[l])
        U[node] = K * B_mat
    end

    # U[root] has shape (k_i * prod_other_n_j, 1) — a modified-size tensor
    root_vec = U[root(tree)][:, 1]
    k_i = size(X.fmat[exclude_dim], 2)
    mod_sz = ntuple(j -> j == exclude_dim ? k_i : X.sz[j], d)
    mod_tensor = reshape(root_vec, mod_sz)

    # Mode-exclude_dim unfold to get V_i' = (k_i, prod_others)
    other_dims = sort(setdiff(1:d, [exclude_dim]))
    perm = vcat([exclude_dim], other_dims)
    T_perm = permutedims(mod_tensor, perm)
    prod_others = prod(X.sz[j] for j in other_dims)
    V_iT = reshape(T_perm, k_i, prod_others)

    return V_iT'  # (prod_others, k_i)
end

"""
Mode-k unfolding of a tensor: puts dimension k as rows, sorted others as columns.
Returns matrix of shape (n_k, prod_others).
"""
function _mode_unfold(X::AbstractArray{T}, k::Int) where T
    d = ndims(X)
    other_dims = sort(setdiff(1:d, [k]))
    perm = vcat([k], other_dims)
    X_perm = permutedims(X, perm)
    return reshape(X_perm, size(X, k), :)
end

"""
Node-matricization: rows indexed by sorted node_dims, cols by sorted complement.
Returns matrix of shape (prod_node, prod_complement).
"""
function _node_unfold(X::AbstractArray{T}, node_dims::Vector{Int}) where T
    d = ndims(X)
    other_dims = sort(setdiff(1:d, node_dims))
    perm = vcat(node_dims, other_dims)
    X_perm = permutedims(X, perm)
    row_size = prod(size(X, i) for i in node_dims)
    col_size = isempty(other_dims) ? 1 : prod(size(X, i) for i in other_dims)
    return reshape(X_perm, row_size, col_size)
end

"""
Bipartite unfolding: reshape tensor as (left_prod, right_prod) where
left_dims and right_dims together span all dimensions.
Follows the convention: left dims inner (fast), right dims outer (slow),
matching kron(Phi_r, Phi_l) column ordering.
"""
function _bipartite_unfold(X::AbstractArray{T}, left_dims::Vector{Int}, right_dims::Vector{Int}) where T
    perm = vcat(left_dims, right_dims)
    X_perm = permutedims(X, perm)
    left_prod = prod(size(X, i) for i in left_dims)
    right_prod = prod(size(X, i) for i in right_dims)
    return reshape(X_perm, left_prod, right_prod)
end

"""
Reshape a vector from node-matricization row ordering into (left_prod, right_prod)
matching the bipartite layout (left dims inner, right dims outer).
"""
function _bipartite_reshape(v::Vector{T}, left_dims::Vector{Int}, right_dims::Vector{Int},
                             node_dims::Vector{Int}, sz::NTuple) where T
    node_sizes = Tuple(sz[i] for i in node_dims)
    combined_order = vcat(left_dims, right_dims)

    # v is in node_dims (sorted ascending) column-major order
    # We need it in combined_order column-major order
    if combined_order == node_dims
        # No permutation needed
        left_prod = prod(sz[i] for i in left_dims)
        right_prod = prod(sz[i] for i in right_dims)
        return reshape(v, left_prod, right_prod)
    end

    # Need to permute
    v_tensor = reshape(v, node_sizes)
    local_perm = [findfirst(==(d), node_dims) for d in combined_order]
    v_perm = permutedims(v_tensor, local_perm)
    left_prod = prod(sz[i] for i in left_dims)
    right_prod = prod(sz[i] for i in right_dims)
    return reshape(v_perm, left_prod, right_prod)
end

# --- Tangent space operations ---

"""Inner product in tangent space."""
function _tangent_inner(fmat1, trten1, fmat2, trten2)
    s = 0.0
    for (a, b) in zip(fmat1, fmat2)
        s += dot(vec(a), vec(b))
    end
    for (a, b) in zip(trten1, trten2)
        s += dot(vec(a), vec(b))
    end
    return s
end

"""Armijo backtracking line search."""
function _line_search(X::HTTensor{T}, dir_fmat, dir_trten, A, b, obj,
                      grad_fmat, grad_trten, config) where T
    alpha = one(T)
    slope = _tangent_inner(grad_fmat, grad_trten, dir_fmat, dir_trten)

    for _ in 1:config.line_search_max
        X_trial = _retract(X, dir_fmat, dir_trten, alpha)
        r_trial = A(full(X_trial)) - b
        obj_trial = T(0.5) * dot(r_trial, r_trial)

        if obj_trial <= obj + config.armijo_c * alpha * slope
            return alpha
        end

        alpha *= T(config.armijo_rho)
    end

    return alpha
end

"""
Retraction: additive update of HT parameters.
"""
function _retract(X::HTTensor{T}, dir_fmat, dir_trten, alpha) where T
    tree = X.tree

    new_fmat = Vector{Matrix{T}}(undef, tree.d)
    for dim in 1:tree.d
        new_fmat[dim] = X.fmat[dim] + T(alpha) * dir_fmat[dim]
    end

    new_trten = Vector{Array{T,3}}(undef, num_internal(tree))
    for idx in 1:num_internal(tree)
        new_trten[idx] = X.trten[idx] + T(alpha) * dir_trten[idx]
    end

    HTTensor(tree, new_trten, new_fmat, X.sz; isorth=false)
end
