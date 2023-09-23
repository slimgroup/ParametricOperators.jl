export ParMatrix, ParMatrixN

using Random
using OMEinsum

"""
Dense matrix operator.
"""
struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,External}
    m::Int
    n::Int
    id::Any
    state::Any
    ParMatrix(T::DataType, m::Int, n::Int, id) = new{T}(m, n, id, 0)
    ParMatrix(m::Int, n::Int, id) = new{Float64}(m, n, id, 0)
    ParMatrix(T::DataType, m::Int, n::Int) = new{T}(m, n, uuid4(Random.GLOBAL_RNG), 0)
    ParMatrix(m::Int, n::Int, state::Int) = new{Float64}(m, n, uuid4(Random.GLOBAL_RNG), state)
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m

complexity(A::ParMatrix{T}) where {T} = elementwise_multiplication_cost(T)*A.n*A.m

# TODO: Fix init scheme

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Real}

    # G = zeros(A.n, A.m)
    # if A.state == 0
    #     G[1, :] = [1, 2]
    #     G[2, :] = [3, 4]
    # else
    #     G[1, :] = [5, 10]
    #     G[2, :] = [15, 20]
    # end
    # G = Float64.(G)
    # d[A] = G

    rng = Random.seed!(1234)

    # glorot_uniform init for dfno, does a lot better for some reason

    scale = sqrt(24.0f0 / sum((A.n, A.m)))
    d[A] = (rand(rng, T, (A.n, A.m)) .- 0.5f0) .* scale

    # d[A] = rand(T, A.n, A.m)/convert(T, sqrt(A.m*A.n))

    d[A] = permutedims(d[A], [2, 1])
end

# TODO: Remove seeds. Currently exists to match implementation to Francis's FNO

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Complex}
    Random.seed!(1234)
    d[A] = rand(T, A.n, A.m)/convert(real(T), sqrt(A.m*A.n))
    d[A] = permutedims(d[A], [2, 1])
end

(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params*x
(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params[A.op.op]'*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params[A.op.op]'*x
*(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractVector{T}} = x*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractMatrix{T}} = x*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractVector{T}} = x*A.params[A.op.op]'
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x*A.params[A.op.op]'

function to_Dict(A::ParMatrix{T}) where {T}
    rv = Dict{String, Any}(
        "type" => "ParMatrix",
        "T" => string(T),
        "m" => A.m,
        "n" => A.n
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

function from_Dict(::Type{ParMatrix}, d)
    ts = d["T"]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    mid = d["id"]
    if startswith(mid, "UUID:")
        mid = UUID(mid[6:end])
    end
    ParMatrix(dtype, d["m"], d["n"], mid)
end

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

# TODO: Add rest of the functionality and abstract usage of einsum to another controller

(A::ParParameterized{T,T,Linear,ParMatrixN{N,M,O,T},V})(x::X) where {N,M,O,T,V,X<:AbstractVector{T}} = vec(einsum(EinCode((A.op.weight_order,A.op.input_order),A.op.target_order),(A.params,reshape(x, A.op.input_shape))))
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

    @assert length(dims_in) == length(A.input_shape)
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

        # TODO: For now, only support perfect distribution because we have no way to refer to the communicator and cartesian grid
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

    return ParMatrixN(T, A.weight_order, tuple(new_weight_shape...), A.input_order, tuple(new_input_shape...), A.target_order, tuple(new_target_shape...))
end
