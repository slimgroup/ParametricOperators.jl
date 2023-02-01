# examples/fno.jl
#
# This file provides an example implementation of a Fourier neural operator
# (https://arxiv.org/abs/2010.08895) using ParametricOperators.jl. This
# implementation differs slightly from the original in that there is only
# a single affine transformation + nonlineairty representing the lifting
# operator.

using ChainRulesCore
using CUDA
using LinearAlgebra
using MAT
using NNlib
using NNlibCUDA
using Parameters
using ParametricOperators
using Printf
using ProgressBars
using Random
using Zygote

# Choose the namespace to use for doing array operations
namespace = CUDA.functional() ? CUDA : Base
dev_array = CUDA.functional() ? CuArray : identity

@with_kw struct FNOConfig
    T::Type{<:Real} = Float64
    shape::Vector{Int}
    modes::Vector{Int}; @assert length(modes) == length(shape)
    batch_size::Int = 1
    timesteps_in::Int = 1
    input_dim::Int = length(shape) + 1
    output_dim::Int = 1
    lifted_dim::Int = 20
    lifted_scale::Int = 4
    num_blocks::Int = 4
    act::Function = gelu
end

function FNO(config::FNOConfig)

    # == Type Information ==
    T = config.T

    # == Get shape information ==
    N = length(config.shape)
    timesteps_out = config.shape[end]
    lifted_shape = [config.lifted_dim, config.shape...]

    # == Declare constant operators ==
    Is  = ParIdentity(T, prod(config.shape[1:end-1])) # Identity on space dims
    It  = ParIdentity(T, config.shape[end])           # Identity on time dim
    Ic  = ParIdentity(T, config.lifted_dim)           # Identity on lifted (channel) dim
    CIc = ParIdentity(Complex{T}, config.lifted_dim)  # Complex identity on lifted (channel) dim

    σ  = ParActivationFunction(x -> config.act.(x), prod(lifted_shape), T)

    # == Lifting Operator ==

    # Weight and bias tensors
    Wc = ParMatrix(T, config.lifted_dim, config.input_dim)
    Wt = ParMatrix(T, timesteps_out, config.timesteps_in)
    bc = ParBias(T, config.lifted_dim; dim=1, shape=lifted_shape)
    bt = ParBias(T, timesteps_out; dim=N+1, shape=lifted_shape)

    Wp = Wt ⊗ Is ⊗ Wc
    bp = bc + bt
    P  = σ ∘ bp ∘ Wp

    # == Blocks ==

    # Fourier transform
    Fs = [ParDFT(Complex{T}, s) for s in config.shape[1:end-1]]
    Ft = ParDRFT(T, config.shape[end])
    F  = Ft ⊗ ParKron(Fs...) ⊗ Ic

    # Restriction
    Rs = ParKron([ParRestriction(Complex{T}, Range(op), [1:m, Range(op)-m+1:Range(op)]) for (op, m) in zip(Fs, config.modes[1:end-1])]...)
    Rt = ParRestriction(Complex{T}, Range(Ft), [1:config.modes[end]])
    R  = Rt ⊗ Rs ⊗ CIc

    # Fourier weighting
    Ds = [ParDiagonal(Complex{T}, Range(Rt)*Range(Rs)) ⊗ ParMatrix(Complex{T}, config.lifted_dim, config.lifted_dim) for _ in 1:config.num_blocks]
    
    # Spectral convolutions
    Cs = [F'*R'*D*R*F for D in Ds]

    # Lifted dimension mixing
    Ws = [It ⊗ Is ⊗ ParMatrix(T, config.lifted_dim, config.lifted_dim) for _ in 1:config.num_blocks]

    # Blocks
    Bs = [σ ∘ (C + W) for (C, W) in zip(Cs, Ws)]
    B  = ∘(Bs...)

    # == Projection Operator ==
    Q = It ⊗ Is ⊗ ParMatrix(T, config.output_dim, config.lifted_dim)
    
    # == Full Network ==
    G = Q ∘ B ∘ P

    return G
end

# Reproducibility
Random.seed!(1337)

# Load data
data_path = joinpath(homedir(), "data/NavierStokes_V1e-5_N1200_T20.mat");
u = Float64.(matread(data_path)["u"]);

# Reshape data
subsample = 1
u = permutedims(u, (2, 3, 4, 1));
u = u[1:subsample:end,1:subsample:end,:,:];
(nx, ny, nt, nb) = size(u);
u = reshape(u, 1, nx, ny, nt, nb);

# Separate training data
train_split = 0.8;
split_idx = Int(round(train_split*nb));
n_train = split_idx
n_test  = nb - n_train

x_train = u[:,:,:,1:1,1:split_idx];
y_train = u[:,:,:,:,1:split_idx];
x_test  = u[:,:,:,1:1,split_idx+1:end];
y_test  = u[:,:,:,:,split_idx+1:end];

# Setup network
config = FNOConfig(T = eltype(u), shape = [nx, ny, nt], modes = [4, 4, 4], input_dim = 1, batch_size = 1);
G = FNO(config);
θ = [dev_array(p) for p in init(G)];

# Setup optimization
num_epochs = 100

α  = 1e-3 # Learning rate
λ  = 1e-4 # Weight decay
β1 = 0.9   # First ADAM decay parameter
β2 = 0.999 # Second ADAM decay parameter
m  = [zero(p) for p in θ]; # First moment vector for ADAM
v  = [zero(p) for p in θ]; # Second moment vector for ADAM

function adam_update!(θ::AbstractArray{<:Number}, g, α, β1, β2, m, v, t; ϵ=1e-8)
    m .*= β1
    v .*= β2
    m .+= (1-β1).*g    
    v .+= (1-β2).*g.^2
    mh = m./(1-β1^t)
    vh = v./(1-β2^t)
    θ .-= α.*mh./(sqrt.(vh) .+ ϵ)
end

function adam_update!(θ::AbstractArray{<:AbstractArray}, g, α, β1, β2, m, v, t; ϵ=1e-8)
    n = length(θ)
    for i in 1:n
        adam_update!(θ[i], g[i], α, β1, β2, m[i], v[i], t; ϵ=ϵ)
    end
end

# Relative L2 loss function
relative_l2_loss(y, ŷ) = norm(vec(y) .- vec(ŷ))

# L1 loss on weights
weight_decay(p) = sum(map(θi -> norm(vec(θi), p=1), p))

# Workaround for Zygote having no adjoint for CUDA.zeros()
# https://discourse.julialang.org/t/error-this-intrinsic-must-be-compiled-to-be-called/52497
#Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

# Main training loop
for epoch in 1:num_epochs

    # Get batch schedule
    schedule = [i+1:i+config.batch_size for i in 0:config.batch_size:n_train-1]
    Random.shuffle!(schedule)

    # Training iterations
    pbar = ProgressBar(enumerate(schedule))
    for (i, batch) in pbar

        # Get training pair
        x = reshape(view(x_train, :,:,:,:,1:1), nx*ny, config.batch_size) |> dev_array;
        y = reshape(view(y_train, :,:,:,:,1:1), nx*ny*nt, config.batch_size) |> dev_array;

        # Compute gradient
        loss = nothing
        g = gradient(p -> begin
            ŷ = G(x, p);
            l = relative_l2_loss(y, ŷ)# + λ*weight_decay(p)
            @ignore_derivatives loss = l
            return l
        end, θ)[1];

        # ADAM parameter update
        adam_update!(θ, g, α, β1, β2, m, v, i)

        set_description(pbar, string(@sprintf("epoch = %03d, batch = %03d, loss = %.4f", epoch, i, loss)))
    end
    break
end