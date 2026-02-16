using ParametricOperators
using Test
using LinearAlgebra
using Random
using ChainRulesCore

Random.seed!(42)

@testset "ParHTTucker operator" begin
    @testset "Construction and init" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)
        A = ParHTTucker(Float64, tree, sz; max_rank=3)

        @test Domain(A) == 1
        @test Range(A) == prod(sz)

        theta = init(A)
        @test haskey(theta, A)
        @test length(theta[A].fmat) == 3
        @test length(theta[A].trten) == num_internal(tree)
    end

    @testset "Forward matches full()" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)

        X = randn_httensor(Float64, tree, sz, 3)
        A, theta = to_parht(X)

        At = A(theta)
        y = At * ones(Float64, 1)

        X_full_vec = vec(full(X))
        @test length(y) == prod(sz)
        @test y ≈ X_full_vec atol=1e-10
    end

    @testset "Scalar scaling" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)
        X = randn_httensor(Float64, tree, sz, 3)
        A, theta = to_parht(X)

        At = A(theta)
        alpha = 3.7
        y_scaled = At * [alpha]

        @test y_scaled ≈ alpha .* vec(full(X)) atol=1e-10
    end

    @testset "Adjoint dot-product test" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)
        X = randn_httensor(Float64, tree, sz, 3)
        A, theta = to_parht(X)

        At = A(theta)
        x = ones(Float64, 1)
        y = At * x

        w = randn(Float64, prod(sz))
        lhs = dot(y, w)

        Aat = A'(theta)
        z = Aat * w
        rhs = dot(x, z)

        @test lhs ≈ rhs atol=1e-10
    end

    @testset "Conversion roundtrip" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)
        X = randn_httensor(Float64, tree, sz, 3)

        A, theta = to_parht(X)
        X2 = from_htparams(A, theta[A])

        @test full(X) ≈ full(X2) atol=1e-12
    end

    @testset "4D tensor" begin
        tree = DimensionTree(4)
        sz = (3, 4, 5, 6)
        X = randn_httensor(Float64, tree, sz, 2)
        A, theta = to_parht(X)

        At = A(theta)
        y = At * ones(Float64, 1)

        @test length(y) == prod(sz)
        @test y ≈ vec(full(X)) atol=1e-10
    end
end

@testset "ParSampling operator" begin
    @testset "Construction" begin
        sz = (4, 5, 6)
        indices = [CartesianIndex(1,1,1), CartesianIndex(2,3,4), CartesianIndex(4,5,6)]
        S = ParSampling(Float64, indices, sz)

        @test Domain(S) == prod(sz)
        @test Range(S) == 3
    end

    @testset "Forward extraction" begin
        sz = (4, 5, 6)
        A_tensor = randn(Float64, sz...)
        indices = [CartesianIndex(1,2,3), CartesianIndex(4,5,6), CartesianIndex(2,3,1)]
        S = ParSampling(Float64, indices, sz)

        y = S * vec(A_tensor)
        @test length(y) == 3
        @test y[1] ≈ A_tensor[1,2,3]
        @test y[2] ≈ A_tensor[4,5,6]
        @test y[3] ≈ A_tensor[2,3,1]
    end

    @testset "Adjoint scatter" begin
        sz = (4, 5, 6)
        indices = [CartesianIndex(1,2,3), CartesianIndex(4,5,6)]
        S = ParSampling(Float64, indices, sz)

        y = [3.0, 7.0]
        x = S' * y
        @test length(x) == prod(sz)

        X_tensor = reshape(x, sz)
        @test X_tensor[1,2,3] ≈ 3.0
        @test X_tensor[4,5,6] ≈ 7.0
        @test sum(abs, x) ≈ 10.0
    end

    @testset "Dot-product test" begin
        sz = (4, 5, 6)
        indices = [CartesianIndex(i,j,k) for i in 1:4 for j in 1:5 for k in 1:6][1:30]
        S = ParSampling(Float64, indices, sz)

        x = randn(Float64, prod(sz))
        y = S * x

        w = randn(Float64, length(indices))
        z = S' * w

        @test dot(y, w) ≈ dot(x, z) atol=1e-10
    end
end

@testset "Composed S ∘ Phi" begin
    @testset "Composed forward" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)
        X = randn_httensor(Float64, tree, sz, 3)
        Phi, theta = to_parht(X)

        indices = [CartesianIndex(1,2,3), CartesianIndex(4,5,6),
                   CartesianIndex(2,3,1), CartesianIndex(3,4,5)]
        S = ParSampling(Float64, indices, sz)

        F = S ∘ Phi

        Ft = F(theta)
        y = Ft * ones(Float64, 1)

        X_full = full(X)
        expected = Float64[X_full[idx] for idx in indices]
        @test y ≈ expected atol=1e-10
    end

    @testset "Tensor completion objective" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)
        X_true = randn_httensor(Float64, tree, sz, 2)
        X_full = full(X_true)

        all_indices = vec([CartesianIndex(i,j,k) for i in 1:4, j in 1:5, k in 1:6])
        perm = randperm(length(all_indices))
        n_obs = div(length(all_indices), 2)
        indices = all_indices[perm[1:n_obs]]
        b = Float64[X_full[idx] for idx in indices]

        Phi = ParHTTucker(Float64, tree, sz; max_rank=2)
        S = ParSampling(Float64, indices, sz)
        F = S ∘ Phi

        theta = init(Phi)
        Ft = F(theta)
        y = Ft * ones(Float64, 1)

        @test length(y) == n_obs
        obj = 0.5 * sum(abs2, y - b)
        @test obj >= 0.0
        @test isfinite(obj)
    end

    @testset "Finite-difference gradient verification" begin
        tree = DimensionTree(3)
        sz = (4, 5, 6)
        X = randn_httensor(Float64, tree, sz, 2)

        indices = [CartesianIndex(1,1,1), CartesianIndex(2,3,4),
                   CartesianIndex(4,5,6), CartesianIndex(3,2,1)]
        b = randn(Float64, length(indices))

        Phi, theta = to_parht(X)

        function loss_from_params(fmat, trten)
            v = ht_reconstruct_vec(Phi.tree, fmat, trten, Phi.sz)
            X_tensor = reshape(v, sz)
            sampled = Float64[X_tensor[idx] for idx in indices]
            return 0.5 * sum(abs2, sampled - b)
        end

        p = theta[Phi]
        loss_val = loss_from_params(p.fmat, p.trten)
        @test isfinite(loss_val)

        # Get AD gradient via rrule
        y_fwd, pb = ChainRulesCore.rrule(ht_reconstruct_vec,
                                          Phi.tree, p.fmat, p.trten, Phi.sz)

        X_tensor = reshape(y_fwd, sz)
        sampled = Float64[X_tensor[idx] for idx in indices]
        residual = sampled - b
        grad_v = zeros(Float64, sz)
        for (r, idx) in zip(residual, indices)
            grad_v[idx] = r
        end
        _, _, dfmat, dtrten, _ = pb(vec(grad_v))

        # Finite difference on first leaf matrix
        eps_fd = 1e-6
        dim = 1
        fmat_pert = copy.(p.fmat)
        grad_fd = zeros(size(p.fmat[dim]))
        for i in eachindex(p.fmat[dim])
            fmat_pert[dim] = copy(p.fmat[dim])
            fmat_pert[dim][i] += eps_fd
            l_plus = loss_from_params(fmat_pert, p.trten)
            fmat_pert[dim][i] -= 2 * eps_fd
            l_minus = loss_from_params(fmat_pert, p.trten)
            grad_fd[i] = (l_plus - l_minus) / (2 * eps_fd)
            fmat_pert[dim] = copy(p.fmat[dim])
        end
        @test dfmat[dim] ≈ grad_fd rtol=1e-4

        # Finite difference on first transfer tensor
        tidx = 1
        trten_pert = copy.(p.trten)
        grad_trten_fd = zeros(size(p.trten[tidx]))
        for i in eachindex(p.trten[tidx])
            trten_pert[tidx] = copy(p.trten[tidx])
            trten_pert[tidx][i] += eps_fd
            l_plus = loss_from_params(p.fmat, trten_pert)
            trten_pert[tidx][i] -= 2 * eps_fd
            l_minus = loss_from_params(p.fmat, trten_pert)
            grad_trten_fd[i] = (l_plus - l_minus) / (2 * eps_fd)
            trten_pert[tidx] = copy(p.trten[tidx])
        end
        @test dtrten[tidx] ≈ grad_trten_fd rtol=1e-4
    end
end
