export ParDiagonal

"""
Diagonal matrix (elementwise) operator.
"""
struct ParDiagonal{T} <: ParLinearOperator{T,T,Parametric,External}
    n::Int
    id::Any
    ParDiagonal(T::DataType, n::Int) = new{T}(n, uuid4(Random.GLOBAL_RNG))
    ParDiagonal(n::Int) = new{Float64}(n, uuid4(Random.GLOBAL_RNG))
    ParDiagonal(T::DataType, n::Int, id) = new{T}(n, id)
    ParDiagonal(n::Int, id) = new{Float64}(n, id)
end

Domain(A::ParDiagonal) = A.n
Range(A::ParDiagonal) = A.n

complexity(A::ParDiagonal{T}) where {T} = elementwise_multiplication_cost(T)*A.n

function init!(A::ParDiagonal{T}, d::Parameters) where {T}
    d[A] = rand(T, A.n)
end

(A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params.*x
(A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params.*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V})(x::X) where {T,V,X<:AbstractVector{T}} = conj(A.params[A.op.op]).*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = conj(A.params[A.op.op]).*x
*(x::X, A::ParParameterized{T,T,Linear,ParDiagonal{T},V}) where {T,V,X<:AbstractVector{T}} = x.*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParDiagonal{T},V}) where {T,V,X<:AbstractMatrix{T}} = x.*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V}) where {T,V,X<:AbstractVector{T}} = x.*conj(A.params[A.op.op])
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x.*conj(A.params[A.op.op])

function to_Dict(A::ParDiagonal{T}) where {T}
    rv = Dict{String, Any}(
        "type" => "ParDiagonal",
        "T" => string(T),
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

function from_Dict(::Type{ParDiagonal}, d)
    ts = d["T"]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    mid = d["id"]
    if startswith(mid, "UUID:")
        mid = UUID(mid[6:end])
    end
    ParDiagonal(dtype, d["n"], mid)
end

function distribute(A::ParDiagonal{T}, comm::MPI.Comm = MPI.COMM_WORLD) where {T}
    local_n = local_size(A.n, MPI.Comm_rank(comm), MPI.Comm_size(comm))
    return ParDiagonal(T, local_n)
end
