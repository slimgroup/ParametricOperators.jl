using LinearAlgebra, Statistics, Zygote, Printf, Test, Polynomials, ParametricOperators

function grad_test(misfit, x0::AbstractArray{T}, dx, g; maxiter=6, h0=5f-2, data=false, stol=1f-1) where T

    err1 = zeros(T, maxiter)
    err2 = zeros(T, maxiter)
    
    gdx = data ? g : dot(g, dx)
    f0 = misfit(x0)
    hs = zeros(T, maxiter)
    h = h0

    @printf("%11.5s, %11.5s, %11.5s, %11.5s, %11.5s, %11.5s \n", "h", "gdx", "e1", "e2", "rate1", "rate2")
    for j = 1:maxiter
        hs[j] = h
        f = misfit(x0 + h*dx)
        err1[j] = norm(f - f0, 1)
        err2[j] = norm(f - f0 - h*gdx, 1)
        j == 1 ? prev = 1 : prev = j - 1
        @printf("%5.5e, %5.5e, %5.5e, %5.5e, %5.5e, %5.5e \n", h, h*norm(gdx, 1), err1[j], err2[j], err1[prev]/err1[j], err2[prev]/err2[j])
        h = h * .8f0
    end

    rate1 = err1[1:end-1]./err1[2:end]
    rate2 = err2[1:end-1]./err2[2:end]
    println("")
    @show mean(rate1), extrema(rate1), mean(rate1) - T(1/.8)
    println("")
    @show mean(rate2), extrema(rate2), (mean(rate2) - T(1/.8)^2)/2

    @test isapprox(mean(rate1), 1.25f0; atol=stol)
    @test isapprox(mean(rate2), 1.5625f0; atol=stol)

    p1 = fit(log10.(hs), log10.(err1), 1)
    p2 = fit(log10.(hs), log10.(err2), 1)
    
    c1 = p1.coeffs[2]
    c2 = p2.coeffs[2]
    println("")
    @show c1 c2 c1-T(1) (c2-T(2))/2
end

function grad_test(F::ParOperator{D,R,Linear,Parametric,T}; kwargs...) where {D,R,T}

    θ = init(F)
    S = eltype(θ)
    θ = S(100).*(θ .- minimum(θ))./(maximum(θ))
    
    dθ = init(F)
    dθ = S(10).*(dθ .- minimum(dθ))./(maximum(dθ))

    x = DDT(F)(100).*rand(DDT(F), Domain(F))
    y = RDT(F)(100).*rand(RDT(F), Range(F))

    misfit(p) = norm(F(x, p) .- y, 1)
    g = gradient(p -> misfit(p), θ)[1]
    grad_test(misfit, θ, dθ, g; kwargs...)

    Fθ = F(θ)
    grad_test(Fθ)

end

function grad_test(F::ParOperator{D,R,Linear,<:Applicable,T}; kwargs...) where {D,R,T}

    x = DDT(F)(100).*rand(DDT(F), Domain(F))
    dx = DDT(F)(10).*rand(DDT(F), Domain(F))
    y = RDT(F)(100).*rand(RDT(F), Range(F))

    misfit(v) = norm(F(v) - y)
    g = gradient(v -> misfit(v), x)[1]
    grad_test(misfit, x, dx, g; kwargs...)

end