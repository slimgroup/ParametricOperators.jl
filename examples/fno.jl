using CUDA
using MAT
using LinearAlgebra
using NNlib
using NNlibCUDA
using ParametricOperators
using Zygote

function spectral_convolution(T, shape_spacetime, modes_spacetime, lifted_dim)

    # Fourier transform along time/space
    dft_time = ParDFT(T, shape_spacetime[end])
    dft_space = ParKron([ParDFT(Complex{T}, s) for s in shape_spacetime[1:end-1]]...)
    dft_spacetime = dft_time ⊗ dft_space

    # Restriction to low-frequency modes
    restrict = ParKron([ParRestriction(Complex{T}, Range(F), m) for (F, m) in zip(dft_spacetime.ops, reverse(modes_spacetime))]...)

    # Combination dft/restriction along each channel dim
    identity_channels = ParIdentity(T, lifted_dim)
    dft_restrict = restrict*dft_spacetime
    dft_all = dft_restrict ⊗ identity_channels

    # Elementwise multiplication and channel mixing in frequency space
    mix_channels = ParMatrix(Complex{T}, lifted_dim, lifted_dim)
    elementwise_weighting = ParDiagonal(Complex{T}, Range(dft_restrict))
    frequency_weight = elementwise_weighting ⊗ mix_channels

    # Spectral conv
    return dft_all'*frequency_weight*dft_all

end

function channel_mixing(T, shape_spacetime, lifted_dim)
    mix_channels = ParMatrix(T, lifted_dim, lifted_dim)
    identity_spacetime = ParIdentity(T, prod(shape_spacetime))
    return identity_spacetime ⊗ mix_channels
end

function fno_block(x, spectral_conv, channel_mix)
    y1 = spectral_conv(x)
    y2 = channel_mix(x)
    return gelu.(y1 .+ y2)
end

shape = [64, 64, 20]
modes = [[1:8, 57:64], [1:8, 57:64], [1:8]]
lifted_dim = 20

T = Float64
S = spectral_convolution(T, shape, modes, lifted_dim)
W = channel_mixing(T, shape, lifted_dim)
θ = init(S)
init!(W, θ)