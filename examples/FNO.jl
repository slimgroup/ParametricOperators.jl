using ChainRulesCore: @ignore_derivatives
using ColorSchemes
using GLMakie
using LinearAlgebra
using MAT
using Tao
using Zygote

nc = 10
nx = 64
ny = 64
nt = 16
n  = nc*nx*ny*nt

mx = 8
my = 8
mt = 8

T = Float32
C_in = TaoConvert{T,Complex{T}}(nx*ny*nt)
C_out(x::V) where {V<:AbstractVector{Complex{T}}} = real.(x)

Is = TaoIdentity{Complex{T}}(nt*ny*nx)
Ic = TaoIdentity{Complex{T}}(nc)

function Block()
    Fx = TaoDFT{Complex{T}}(nx)
    Fy = TaoDFT{Complex{T}}(ny)
    Ft = TaoDFT{Complex{T}}(nt)

    Rx = TaoRestriction{Complex{T}}(nx, 1:mx, nx-mx+1:nx)
    Ry = TaoRestriction{Complex{T}}(ny, 1:my, ny-my+1:ny)
    Rt = TaoRestriction{Complex{T}}(nt, 1:mt)

    Wc1 = TaoMatrix{Complex{T}}(nc, nc)
    Wc2 = TaoMatrix{Complex{T}}(nc, nc)

    F = (Rt ⊗ Ry ⊗ Rx ⊗ Ic) * (Ft ⊗ Fy ⊗ Fx ⊗ Ic)
    D = TaoDiagonal{Complex{T}}(Range(Rt)*Range(Ry)*Range(Rx)) ⊗ Wc1
    S = F'*D*F
    W = Is ⊗ Wc2
    σ = x -> tanh.(x)
    return σ ∘ (S + W)
end

Q = Is ⊗ TaoMatrix{Complex{T}}(nc, 1)
P = Is ⊗ TaoMatrix{Complex{T}}(1, nc)

B1 = Block()
B2 = Block()
F = C_out ∘ P ∘ B2 ∘ B1 ∘ Q ∘ C_in

data = matread("data/ns_V1e-3_N5000_T50.mat")
u = data["u"]

n_train = 1000
n_test  = 100
x_train = repeat(u[1:n_train,:,:,1:1], 1, 1, 1, nt)
x_test  = repeat(u[n_train+1:n_train+n_test,:,:,1:1], 1, 1, 1, nt)
y_train = u[1:n_train,:,:,1:nt]
y_test  = u[1:n_train,:,:,1:nt]

# Init params and adam variables
θ  = init(F)
α  = T(1e-3)
ϵ  = T(1e-8)
β1 = T(0.9)
β2 = T(0.999)
m  = zero(θ)
v  = zero(θ)
m̂  = zero(θ)
v̂  = zero(θ)

L(ŷ::Y, y::Y) where {Y<:AbstractVector{T}} = norm(ŷ.-y)/norm(ŷ)
n_epoch  = 100
Ls_train = zeros(T, n_epoch)
Ls_test  = zeros(T, n_epoch)

for i in 1:n_epoch
    L_train = T(0)
    for i = 1:n_train
        @show i
        x = vec(x_train[i,:,:,:])
        y = vec(y_train[i,:,:,:])
        g = gradient(p -> begin
            ŷ = F(x, p)
            l = L(ŷ, y)
            @show l
            @ignore_derivatives L_train += l
            return l
        end, θ)[1]
        m .= (β1.*m + (1-β1).*g) 
        v .= β2.*v + (1-β2).*(g.^2)
        m̂ .= m./(1-β1^i)
        v̂ .= v./(1-β2^i)
        θ .-= α.*m̂./(sqrt.(v̂) .+ ϵ)
    end
    L_train ./= n_train
    Ls_train[i] = L_train
    @show L_train
end