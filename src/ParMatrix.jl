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

# TODO: Fix init scheme

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Real}

    if A.n == 1
        d[A] = zeros(T, A.m, A.n)
        return
    end
    # G = zeros(A.n, A.m)
    # G[1, :] = [1, 2]
    # G[2, :] = [3, 4]
    # G = Float64.(G)
    # d[A] = G
    # return

    rng = Random.seed!(1234)

    # glorot_uniform init for dfno, does a lot better for some reason

    scale = sqrt(24.0f0 / sum((A.n, A.m)))
    d[A] = (rand(rng, T, (A.n, A.m)) .- 0.5f0) .* scale

    # d[A] = rand(T, A.n, A.m)/convert(T, sqrt(A.m*A.n))

    d[A] = permutedims(d[A], [2, 1])
end

# TODO: Remove seeds. Currently exists to match implementation to Francis's FNO

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Complex}
    if A.n == 1
        d[A] = zeros(T, A.m, A.n)
        return
    end
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

+(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractVector{T}} = x.+A.params
+(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractArray{T}} = x.+A.params
+(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractMatrix{T}} = x.+A.params
+(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractVector{T}} = x+A.params[A.op.op]'
+(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractArray{T}} = x+A.params[A.op.op]'
+(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x+A.params[A.op.op]'

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
