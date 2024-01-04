export ParMatrix

"""
Dense matrix operator.
"""
struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,External}
    m::Int
    n::Int
    id::Any
    ParMatrix(T::DataType, m::Int, n::Int, id) = new{T}(m, n, id)
    ParMatrix(m::Int, n::Int, id) = new{Float64}(m, n, id)
    ParMatrix(T::DataType, m::Int, n::Int) = new{T}(m, n, uuid4(Random.GLOBAL_RNG))
    ParMatrix(m::Int, n::Int) = new{Float64}(m, n, uuid4(Random.GLOBAL_RNG))
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m

complexity(A::ParMatrix{T}) where {T} = elementwise_multiplication_cost(T)*A.n*A.m

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Real}
    d[A] = rand(T, A.m, A.n)/convert(T, sqrt(A.m*A.n))
end

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Complex}
    d[A] = rand(T, A.m, A.n)/convert(real(T), sqrt(A.m*A.n))
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

function Base.getindex(A::ParMatrix{T}, rows, cols) where T
    row_range = isa(rows, Colon) ? (1:Range(A)) : (isa(rows, Integer) ? (rows:rows) : rows)
    col_range = isa(cols, Colon) ? (1:Domain(A)) : (isa(rows, Integer) ? (cols:cols) : cols)

    new_m = length(row_range)
    new_n = length(col_range)

    return ParMatrix(T, new_m, new_n)
end

function Base.getindex(A::ParParameterized{T,T,Linear,ParMatrix{T},V}, rows, cols) where {T,V}
    row_range = isa(rows, Colon) ? (1:Range(A)) : (isa(rows, Integer) ? (rows:rows) : rows)
    col_range = isa(cols, Colon) ? (1:Domain(A)) : (isa(rows, Integer) ? (cols:cols) : cols)

    new_m = length(row_range)
    new_n = length(col_range)
    
    new_params = reshape(A.params[rows, cols], new_m, new_n)
    new_matrix = ParMatrix(T, new_m, new_n)

    return ParParameterized(new_matrix, new_params)
end
