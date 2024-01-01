export ParMatrixN

using OMEinsum
using Flux:batched_mul

"""
Dense N dimensional matrix operator.
"""
struct ParMatrixN{N,M,O,T} <: ParLinearOperator{T,T,Parametric,External}
    weight_order::Tuple{Vararg{Int,N}}
    weight_shape::Tuple{Vararg{Int,N}}

    input_order::Tuple{Vararg{Int,M}}
    input_shape::Tuple{Vararg{Int,M}}

    target_order::Tuple{Vararg{Int,O}}
    target_shape::Tuple{Vararg{Int,O}}
    id::Any

    ParMatrixN(T::DataType, wo::Tuple{Vararg{Int,N}}, ws::Tuple{Vararg{Int,N}}, io::Tuple{Vararg{Int,M}}, is::Tuple{Vararg{Int,M}}, to::Tuple{Vararg{Int,O}}, ts::Tuple{Vararg{Int,O}}, id) where {N, M, O} = new{N,M,O,T}(wo, ws, io, is, to, ts, id)
    ParMatrixN(wo::Tuple{Vararg{Int,N}}, ws::Tuple{Vararg{Int,N}}, io::Tuple{Vararg{Int,M}}, is::Tuple{Vararg{Int,M}}, to::Tuple{Vararg{Int,O}}, ts::Tuple{Vararg{Int,O}}, id) where {N, M, O} = new{N,M,O,Float64}(wo, ws, io, is, to, ts, id)
    ParMatrixN(T::DataType, wo::Tuple{Vararg{Int,N}}, ws::Tuple{Vararg{Int,N}}, io::Tuple{Vararg{Int,M}}, is::Tuple{Vararg{Int,M}}, to::Tuple{Vararg{Int,O}}, ts::Tuple{Vararg{Int,O}}) where {N, M, O} = new{N,M,O,T}(wo, ws, io, is, to, ts, uuid4(Random.GLOBAL_RNG))
    ParMatrixN(wo::Tuple{Vararg{Int,N}}, ws::Tuple{Vararg{Int,N}}, io::Tuple{Vararg{Int,M}}, is::Tuple{Vararg{Int,M}}, to::Tuple{Vararg{Int,O}}, ts::Tuple{Vararg{Int,O}}) where {N, M, O} = new{N,M,O,Float64}(wo, ws, io, is, to, ts, uuid4(Random.GLOBAL_RNG))
end

Domain(A::ParMatrixN) = prod(A.input_shape)
Range(A::ParMatrixN) = prod(A.target_shape)

# TODO: Remove seeds

function init!(A::ParMatrixN{N,M,O,T}, d::Parameters) where {N,M,O,T<:Real}
    Random.seed!(1234)
    d[A] = rand(T, A.weight_shape...) ./ convert(T, sqrt(prod(A.weight_shape)))
end

function init!(A::ParMatrixN{N,M,O,T}, d::Parameters) where {N,M,O,T<:Complex}
    Random.seed!(1234)
    d[A] = rand(T, A.weight_shape...) ./ convert(real(T), sqrt(prod(A.weight_shape)))
end

# TODO: Add rest of the functionality and abstract usage of OMEinsum to another controller ?

function (A::ParParameterized{T,T,Linear,ParMatrixN{6,M,O,T},V})(x::X) where {M,O,T,V,X<:AbstractMatrix{T}}
    # Hacky batched mul for Just ML4Seismic
    b = size(x)[2]
    oc = A.op.weight_shape[1]   
    ic = A.op.weight_shape[2]
    nx = A.op.weight_shape[3]
    ny = A.op.weight_shape[4]
    nz = A.op.weight_shape[5]
    nt = A.op.weight_shape[6]

    # input from ixyztb -> bi(xyzt)
    input = reshape(x, (A.op.input_shape..., b))
    input = permutedims(input, [6,1,2,3,4,5])
    input = reshape(input, b, ic, :)

    # params from oixyzt -> io(xyzt)
    params = permutedims(A.params, [2,1,3,4,5,6])
    params = reshape(params, ic, oc, :)

    # output from bo(xyzt) -> (oxyzt)b
    output = batched_mul(input, params)
    output = reshape(output, b, oc, nx, ny, nz, nt)
    output = permutedims(output, [2,3,4,5,6,1])
    output = reshape(output, :, b)

    return output
end

function (A::ParParameterized{T,T,Linear,ParMatrixN{5,M,O,T},V})(x::X) where {M,O,T,V,X<:AbstractMatrix{T}}
    # Hacky batched mul for Just ML4Seismic
    b = size(x)[2]
    oc = A.op.weight_shape[1]   
    ic = A.op.weight_shape[2]
    nx = A.op.weight_shape[3]
    ny = A.op.weight_shape[4]
    nt = A.op.weight_shape[5]

    # input from ixytb -> bi(xyt)
    input = reshape(x, (A.op.input_shape..., b))
    input = permutedims(input, [5,1,2,3,4])
    input = reshape(input, b, ic, :)

    # params from oixyt -> io(xyt)
    params = permutedims(A.params, [2,1,3,4,5])
    params = reshape(params, ic, oc, :)

    # output from bo(xyt) -> (oxyt)b
    output = batched_mul(input, params)
    output = reshape(output, b, oc, nx, ny, nt)
    output = permutedims(output, [2,3,4,5,1])
    output = reshape(output, :, b)

    return output
end

function (A::ParParameterized{T,T,Linear,ParMatrixN{4,M,O,T},V})(x::X) where {M,O,T,V,X<:AbstractMatrix{T}}
    # Hacky batched mul for Just ML4Seismic
    b = size(x)[2] 
    ic = A.op.weight_shape[1]
    oc = A.op.weight_shape[2]  
    nt = A.op.weight_shape[3]
    nxy = A.op.weight_shape[4]

    # input from it(xy)b -> bi(txy)
    input = reshape(x, (A.op.input_shape..., b))
    input = permutedims(input, [4,1,2,3])
    input = reshape(input, b, ic, :)

    # params from iot(xy) -> io(txy)
    params = reshape(A.params, ic, oc, :)

    # output from bo(txy) -> (otxy)b
    output = batched_mul(input, params)
    output = reshape(output, b, oc, nt, nxy)
    output = permutedims(output, [2,3,4,1])
    output = reshape(output, :, b)

    return output
end

# function (A::ParParameterized{T,T,Linear,ParMatrixN{N,M,O,T},V})(x::X) where {N,M,O,T,V,X<:AbstractMatrix{T}}
#     # GPU supported batched mult
#     b = size(x)[2]
#     nd = maximum([A.op.input_order..., A.op.weight_order...]) + 1

#     io = (A.op.input_order..., nd)
#     wo = A.op.weight_order
#     to = (A.op.target_order..., nd)

#     is = (A.op.input_shape..., b)

#     return reshape(einsum(EinCode((wo,io),to),(A.params |> cpu,reshape(x, is) |> cpu)), (:, b)) |> gpu
# end

# (A::ParParameterized{T,T,Linear,ParMatrixN{N,M,O,T},V})(x::X) where {N,M,O,T,V,X<:AbstractVector{T}} = vec(einsum(EinCode((A.op.weight_order,A.op.input_order),A.op.target_order),(A.params,reshape(x, A.op.input_shape))))
# (A::ParParameterized{T,T,Linear,ParMatrixN{N,M,O,T},V})(x::X) where {N,M,O,T,V,X<:AbstractVector{T}} = vec(einsum(EinCode((A.op.weight_order,A.op.input_order),A.op.target_order),(A.params |> cpu,reshape(x, A.op.input_shape) |> cpu))) |> gpu
# (A::ParParameterized{T,T,Linear,ParMatrixN{N,M,O,T},V})(x::X) where {N,M,O,T,V,X<:AbstractMatrix{T}} = A.params*x
# (A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrixN{N,M,O,T}},V})(x::X) where {N,M,O,T,V,X<:AbstractVector{T}} = A.params[A.op.op]'*x
# (A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrixN{N,M,O,T}},V})(x::X) where {N,M,O,T,V,X<:AbstractMatrix{T}} = A.params[A.op.op]'*x
# *(x::X, A::ParParameterized{T,T,Linear,ParMatrixN{N,M,O,T},V}) where {N,M,O,T,V,X<:AbstractVector{T}} = x*A.params
# *(x::X, A::ParParameterized{T,T,Linear,ParMatrixN{N,M,O,T},V}) where {N,M,O,T,V,X<:AbstractMatrix{T}} = x*A.params
# *(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrixN{N,M,O,T}},V}) where {N,M,O,T,V,X<:AbstractVector{T}} = x*A.params[A.op.op]'
# *(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrixN{N,M,O,T}},V}) where {N,M,O,T,V,X<:AbstractMatrix{T}} = x*A.params[A.op.op]'

function to_Dict(A::ParMatrixN{N,M,O,T}) where {N,M,O,T}
    rv = Dict{String, Any}(
        "type" => "ParMatrixN",
        "T" => string(T),
        "ws" => A.weight_shape,
        "wo" => A.weight_order,
        "is" => A.input_shape,
        "io" => A.input_order,
        "to" => A.target_order,
        "ts" => A.target_shape,
    )
    if typeof(A.id) == String
        rv["id"] = A.id
    elseif typeof(A.id) == UUID
        rv["id"] = "UUID:$(string(A.id))"
    else
        throw(ParException("I don't know how to encode id of type $(typeof(A.id))"))
    end
    rv
end

function from_Dict(::Type{ParMatrixN}, d)
    ts = d["T"]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    mid = d["id"]
    if startswith(mid, "UUID:")
        mid = UUID(mid[6:end])
    end
    ParMatrixN(dtype, d["ws"], d["wo"], d["is"], d["io"], d["to"], d["ts"], mid)
end

function distribute(A::ParMatrixN{N,M,O,T}, dims_in, comm::MPI.Comm = MPI.COMM_WORLD) where {N,M,O,T}
    # TODO: Rethink how to repartition this or completely rewrite this operator somehow (basically what distdl is doing?)
    @assert length(dims_in) == length(A.input_shape)

    comm_cart = MPI.Cart_create(comm, dims_in)
    coords = MPI.Cart_coords(comm_cart)

    # TODO: Also assert comm size and dims_in product

    combined_tuples = tuple(A.input_order..., A.weight_order..., A.target_order...)
    count_occurrences = (element_to_count) -> sum(element == element_to_count for element in combined_tuples)

    new_input_shape = collect(A.input_shape)
    new_target_shape = collect(A.target_shape)
    new_weight_shape = collect(A.weight_shape)

    for (i, dim) in enumerate(A.input_order)
        dist_across = dims_in[dim]
        if count_occurrences(dim) == 2
            # Do not distribute across the dimenions on which the convolution is performed
            @assert dist_across == 1
        end

        # TODO: For now, only support perfect distribution

        @assert A.input_shape[i] % dist_across == 0
        new_input_shape[i] = A.input_shape[i] ÷ dist_across

        for (j, dim_j) in enumerate(A.weight_order)
            if dim_j == dim
                new_weight_shape[j] = A.weight_shape[j] ÷ dist_across
                break
            end
        end

        for (j, dim_j) in enumerate(A.target_order)
            if dim_j == dim
                new_target_shape[j] = A.target_shape[j] ÷ dist_across
                break
            end
        end
    end

    return ParMatrixN(T, A.weight_order, tuple(new_weight_shape...), A.input_order, tuple(new_input_shape...), A.target_order, tuple(new_target_shape...), "$(A.id):($(join(coords, ',')))")
end
