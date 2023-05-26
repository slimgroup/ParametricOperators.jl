using MPI
using ParametricOperators
using Parameters
using Profile
using MAT
using Shuffle
using Zygote
using Flux
using PyPlot
using DrWatson

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

n_data = 1
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

        grads = gradient(params -> Flux.mse(network(params)*x_train, y_train), θ)[1]
        
        scale!(step_length, grads)
        update!(θ, grads)
    end
end

fig = figure(figsize=(20, 12))

for i = 1:5

    j = i + n_data

    x_sample = vec(data[j, :, :, :, :, 1:config.nt_in])
    y_sample = network(θ)*x_sample
    y_sample = reshape(y_sample, 64, 64, 40)

    subplot(4,5,i)
    imshow(reshape(data[j, :, :, :, :, 1], 64, 64))
    title("x")

    subplot(4,5,i+5)
    imshow(reshape(data[j, :, :, :, :, 50], 64, 64), vmin=0, vmax=1)
    title("true y")

    subplot(4,5,i+10)
    imshow(reshape(y_sample[:, :, 40], 64, 64), vmin=0, vmax=1)
    title("predict y")

    subplot(4,5,i+15)
    imshow(5f0 .* reshape(abs.(vec(data[j, :, :, :, :, 50])-vec(y_sample[:, :, 40])), 64, 64), vmin=0, vmax=1)
    title("5X abs difference")

end
tight_layout()

sim_name = "2D_FNO"
exp_name = "navier_stokes"
save_dict = @strdict exp_name
plot_path = "./plots" # plotsdir(sim_name, savename(save_dict; digits=6))
fig_name = @strdict n_epochs batch_size # Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2D_dfno.png"), fig);
close(fig)
