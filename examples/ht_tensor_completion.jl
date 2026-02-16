using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ParametricOperators
using ChainRulesCore
using LinearAlgebra
using Random

Random.seed!(42)

# === 1. Define the ground truth low-rank tensor ===
sz = (6, 8, 10)
tree = DimensionTree(3)
X_true = randn_httensor(Float64, tree, sz, 2)
X_full = full(X_true)
println("Ground truth tensor: size=$(sz), HT rank=2")
println("  norm = $(norm(X_full))")

# === 2. Sample 50% of entries (tensor completion setup) ===
all_indices = vec([CartesianIndex(i,j,k) for i in 1:sz[1], j in 1:sz[2], k in 1:sz[3]])
n_obs = div(length(all_indices), 2)
indices = all_indices[randperm(length(all_indices))[1:n_obs]]
b = Float64[X_full[idx] for idx in indices]
println("\nObserved $(n_obs)/$(prod(sz)) entries ($(round(100*n_obs/prod(sz)))%)")

# === 3. Build the parametric forward model: S ∘ Phi ===
Phi = ParHTTucker(Float64, tree, sz; max_rank=2)
S = ParSampling(Float64, indices, sz)
F = S ∘ Phi

println("\nOperator pipeline: S ∘ Phi")
println("  Phi: Domain=$(Domain(Phi)) → Range=$(Range(Phi))  (HT reconstruction)")
println("  S:   Domain=$(Domain(S)) → Range=$(Range(S))  (sampling)")

# === 4. Initialize parameters (perturbed ground truth for demo) ===
theta = init(Phi)
# For a meaningful demo, start near the true solution with added noise.
# (Random init on this non-convex problem rarely converges with simple GD.)
true_params = (fmat=copy.(X_true.fmat), trten=copy.(X_true.trten))
noise_level = 0.3
for dim in 1:tree.d
    theta[Phi].fmat[dim] .= true_params.fmat[dim] .+ noise_level .* randn(size(true_params.fmat[dim]))
end
for idx in 1:num_internal(tree)
    theta[Phi].trten[idx] .= true_params.trten[idx] .+ noise_level .* randn(size(true_params.trten[idx]))
end
v0 = ht_reconstruct_vec(tree, theta[Phi].fmat, theta[Phi].trten, sz)
println("\nInitialized parameters (perturbed ground truth, noise=$(noise_level))")
println("  initial rel_error = $(round(norm(v0 - vec(X_full)) / norm(vec(X_full)), sigdigits=4))")

# === 5. Simple gradient descent loop ===
n_iter = 500

println("\nRunning gradient descent ($(n_iter) iters)...")
for iter in 1:n_iter
    p = theta[Phi]

    # Forward pass via rrule (gives us the pullback for free)
    v, pullback = ChainRulesCore.rrule(ht_reconstruct_vec, tree, p.fmat, p.trten, sz)

    # Compute residual at sampled indices
    X_tensor = reshape(v, sz)
    sampled = Float64[X_tensor[idx] for idx in indices]
    residual = sampled - b
    obj = 0.5 * sum(abs2, residual)

    # Backprop: scatter residual gradient into full tensor shape
    grad_v = zeros(Float64, sz)
    for (r, idx) in zip(residual, indices)
        grad_v[idx] = r
    end

    # Get parameter gradients via pullback
    _, _, dfmat, dtrten, _ = pullback(vec(grad_v))

    # Adaptive step size: scale lr so parameter updates are bounded
    grad_norm = sum(norm(g) for g in dfmat) + sum(norm(g) for g in dtrten)
    lr = min(5e-3, 1.0 / (grad_norm + 1e-12))

    # Gradient step
    for dim in 1:tree.d
        theta[Phi].fmat[dim] .-= lr .* dfmat[dim]
    end
    for idx in 1:num_internal(tree)
        theta[Phi].trten[idx] .-= lr .* dtrten[idx]
    end

    if iter == 1 || iter % 100 == 0
        # Reconstruct and measure full error
        v_new = ht_reconstruct_vec(tree, theta[Phi].fmat, theta[Phi].trten, sz)
        rel_err = norm(v_new - vec(X_full)) / norm(vec(X_full))
        println("  iter $(lpad(iter, 3)): obj = $(round(obj, sigdigits=4)),  rel_error = $(round(rel_err, sigdigits=4)),  lr = $(round(lr, sigdigits=2))")
    end
end

# === 6. Final result ===
v_final = ht_reconstruct_vec(tree, theta[Phi].fmat, theta[Phi].trten, sz)
rel_err = norm(v_final - vec(X_full)) / norm(vec(X_full))
println("\nFinal relative error: $(round(rel_err, sigdigits=4))")

# === 7. Demonstrate composition with the PO framework ===
println("\n--- PO framework features ---")
Ft = F(theta)
y = Ft * ones(Float64, 1)
println("Composed forward (S ∘ Phi)(θ) * [1.0]:  $(length(y)) sampled values")
println("  residual norm = $(round(norm(y - b), sigdigits=4))")

# Adjoint dot-product test: <Phi*x, w> == <x, Phi'*w>
x = ones(Float64, 1)
Phi_t = Phi(theta)
y_phi = Phi_t * x
w = randn(Float64, prod(sz))
z = Phi'(theta) * w
lhs = dot(y_phi, w)
rhs = dot(x, z)
println("Adjoint Phi'(θ) maps $(Range(Phi))→$(Domain(Phi)):")
println("  <Phi*x, w> = $(round(lhs, sigdigits=6))")
println("  <x, Phi'*w> = $(round(rhs, sigdigits=6))")
println("  match: $(isapprox(lhs, rhs, atol=1e-10))")

println("\nDone.")
