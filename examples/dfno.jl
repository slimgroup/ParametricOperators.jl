using MPI
using ParametricOperators
using Parameters
using Profile
using MAT
using Shuffle
using Zygote
using Flux

@with_kw struct ModelConfig
    nx::Int
    ny::Int
    nz::Int
    nt_in::Int = 1
    nt_out::Int = 1
    nc_in::Int = 1
    nc_out::Int = 1
    nc_lift::Int = 20
    mx::Int
    my::Int
    mz::Int
    mt::Int
    n_blocks::Int = 4
    dtype::DataType = Float32
    partition::Vector{Int}
end

function dfno(config::ModelConfig)

    T = config.dtype

    function lifting(in_shape, lift_dim, out_features)

        layer = ParIdentity(T, 1) 

        for dim in eachindex(in_shape)
            if dim == lift_dim
                layer = layer ⊗ ParMatrix(T, out_features, in_shape[dim])
            else 
                layer = layer ⊗ ParIdentity(T, in_shape[dim]) 
            end
        end

        return layer
    end

    function sconv()

        # Build 4D Fourier transform with real-valued FFT along time
        fourier_x = ParDFT(Complex{T}, config.nx)
        fourier_y = ParDFT(Complex{T}, config.ny)
        fourier_z = ParDFT(Complex{T}, config.nz)
        fourier_t = ParDFT(T, config.nt_out)

        # Build restrictions to low-frequency modes
        restrict_x = ParRestriction(Complex{T}, Range(fourier_x), [1:config.mx, config.nx-config.mx+1:config.nx])
        restrict_y = ParRestriction(Complex{T}, Range(fourier_y), [1:config.my, config.ny-config.my+1:config.ny])
        restrict_z = ParRestriction(Complex{T}, Range(fourier_z), [1:config.mz])
        restrict_t = ParRestriction(Complex{T}, Range(fourier_t), [1:config.mt])

        # Setup FFT-restrict pattern with Kroneckers
        restrict_dft = (restrict_t * fourier_t) ⊗
                    ((restrict_z ⊗ restrict_y ⊗ restrict_x) * (fourier_z ⊗ fourier_y ⊗ fourier_x)) ⊗
                    ParIdentity(T, config.nc_lift)

        # Diagonal/mixing of modes on each channel
        weight_diag = ParDiagonal(Complex{T}, Range(restrict_dft))

        weight_mix = ParIdentity(Complex{T}, Range(weight_diag) ÷ config.nc_lift) ⊗
                    ParMatrix(Complex{T}, config.nc_lift, config.nc_lift)

        sconv = restrict_dft' * weight_mix * weight_diag * restrict_dft
        return sconv
    end

    shape = [config.nc_in, config.nx, config.ny, config.nz, config.nt_in]

    # Lift Time and Channel dimension
    lt = lifting(shape, 5, config.nt_out)
    shape[5] = config.nt_out

    lc = lifting(shape, 1, config.nc_lift)
    shape[1] = config.nc_lift

    network = lc * lt

    # Add all sconv blocks
    for _ in eachindex(config.n_blocks)
        sconv = sconv()
        network = sconv * network
    end

    # Project channel dimension
    pc = lifting(shape, 1, config.nc_out)
    shape[1] = config.nc_out

    network = pc * network

    return network
end

config = ModelConfig(n_blocks=1, nx=64, ny=64, nz=1, nt_in = 10, nt_out=40, nc_lift=10, mx=4, my=4, mz=1, mt=8, partition=[1])
network = dfno(config)

θ = init(network)

data = matread("./examples/ns_V1e-3_N5000_T50.mat")["u"]
data = reshape(data, :, config.nc_in, config.nx, config.ny, config.nz, config.nt_in + config.nt_out)

n_data = 1 # 5000
n_epochs = 1
batch_size = 1
step_length = 1e-3

for _ in 1:n_epochs
    batch_indexes = []
    for start in 1:batch_size:n_data
        push!(batch_indexes, start:start+batch_size-1)
    end
    shuffle!(batch_indexes)
    for batch in batch_indexes
        x_train = vec(data[batch, :, :, :, :, 1:config.nt_in])
        y_train = vec(data[batch, :, :, :, :, config.nt_in+1:end])

        grads = gradient(params -> Flux.mse(network(params)*x_train, y_train), θ)
        # scale!(step_length, grads)
        # update!(θ, grads)
    end
end