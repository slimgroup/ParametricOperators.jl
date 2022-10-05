using ChainRulesCore, LinearAlgebra, JOLI, MAT, NNlib, ParametricOperators, PyPlot, Random, Statistics, Zygote

# Base datatype
T = Float32

# Load data
data_path = "NavierStokes_V1e-5_N1200_T20.mat"
data = matread(data_path)
u = T.(data["u"])
u = reshape(u, size(u)[1], 1, size(u)[2:end]...)

ntrain = 100
ntest  = 10

S = 2
x_train = repeat(u[1:ntrain,:,1:S:end,1:S:end,1], 1, 1, 1, 1, size(u)[end])
y_train = u[1:ntrain,:,1:S:end,1:S:end,:]
x_test  = repeat(u[ntrain+1:ntrain+ntest,:,1:S:end,1:S:end,1], 1, 1, 1, 1, size(u)[end])
y_test  = u[ntrain+1:ntrain+ntest,:,1:S:end,1:S:end,:]

# Size of data
(nx, ny, nt) = size(x_train)[3:end]
nc = 8
ns = nx*ny*nt

# Setup transform
C  = joWrapper(joCurvelet2D(nx, ny; DDT=Complex{T}, RDT=Complex{T}))
F  = ParDFT{Complex{T}}(nt)
Ic = ParIdentity{Complex{T}}(nc)
Is = ParIdentity{Complex{T}}(ns)
E  = F ⊗ C ⊗ Ic

# Setup blocks
σ(x::V) where {U<:Number, V<:AbstractVecOrMat{U}} = tanh.(x)

nb = 4

Ds = [ParDiagonal{Complex{T}}(ns) ⊗ ParMatrix{Complex{T}}(nc, nc) for _ in 1:nb]
Ws = [Is ⊗ ParMatrix{Complex{T}}(nc, nc) for _ in 1:nb]
Bs = [i == nb ? E'*D*E + W : σ ∘ (E'*D*E + W) for (i, (D, W)) in enumerate(zip(Ds, Ws))]
B  = ∘(Bs...)

# Setup network
fQ(x::V) where {U<:Number, V<:AbstractVecOrMat{U}} = complex.(x)
fP(x::V) where {U<:Number, V<:AbstractVecOrMat{U}} = x
Q = (Is ⊗ ParMatrix{Complex{T}}(nc, 1)) ∘ ParFunction(T,Complex{T},NonLinear,NonParametric,ns,ns,fQ)
P = ParFunction(Complex{T},T,NonLinear,NonParametric,ns,ns,fP) ∘ (Is ⊗ ParMatrix{Complex{T}}(1, nc)) 
G = P ∘ B ∘ Q

# Setup optimization parameters
L(ŷ::V, y::V) where {U<:Number,V<:AbstractVector{U}} = norm(ŷ.-y)/norm(ŷ)

α  = T(1e-3)
β1 = T(0.9)
β2 = T(0.999)
ϵ   = T(1e-8)

θ = init(G)
m = zero(θ)
m̂ = zero(θ)
v = zero(θ)
v̂ = zero(θ)

nepoch = 100

# Initialize network loss trackers
Ls_train = Vector{Vector{T}}()
Ls_test  = Vector{Vector{T}}()

# Training loop
for i in 1:nepoch
	schedule = shuffle(1:ntrain)
	L_train = Vector{T}()
	for j in schedule
		x = vec(x_train[j,:,:,:,:])
		y = vec(y_train[j,:,:,:,:])
		g = gradient(p -> begin
			ŷ = G(x, p)
			l = L(ŷ, y)
			@show l
			@ignore_derivatives push!(L_train, l)
			return l
		end, θ)[1]
		
		m .= β1.*m + (1-β1).*g
		v .= β2.*v + (1-β2).*(g.^2)
		m̂ .= m./(1-β1^i)
		v̂ .= v./(1-β2^i)
		θ .-= α.*m̂./(sqrt.(v̂) .+ ϵ)
	end
	push!(Ls_train, L_train)
	println("epoch = $i, avg train loss = $(mean(L_train))")
	
	L_test = Vector{T}()
	for j in 1:ntest
		x = vec(x_test[j,:,:,:,:])
		y = vec(y_test[j,:,:,:,:])
		ŷ = G(x, θ)
		l = L(ŷ, y)
		push!(L_test, l)
	end
	
	push!(Ls_test, L_test)
	println("epoch = $i, avg test loss = $(mean(L_test))")
	
end
