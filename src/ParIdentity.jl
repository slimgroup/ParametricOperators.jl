export ParIdentity

"""
Identity operator.
"""
struct ParIdentity{T} <: ParLinearOperator{T,T,NonParametric,External}
    n::Int
    ParIdentity(T, n) = new{T}(n)
    ParIdentity(n) = new{Float64}(n)
end

Domain(A::ParIdentity) = A.n
Range(A::ParIdentity) = A.n
adjoint(A::ParIdentity) = A
(A::ParIdentity{T})(x::X) where {T,X<:AbstractVector{T}} = x
(A::ParIdentity{T})(x::X) where {T,X<:AbstractMatrix{T}} = x

complexity(::ParIdentity) = 0

(A::ParDistributed{T,T,Linear,NonParametric,ParIdentity{T}})(x::X) where {T,X<:AbstractVector{T}} = x
(A::ParDistributed{T,T,Linear,NonParametric,ParIdentity{T}})(x::X) where {T,X<:AbstractMatrix{T}} = x
*(x::X, A::ParDistributed{T,T,Linear,NonParametric,ParIdentity{T}}) where {T,X<:AbstractMatrix{T}} = x

to_Dict(A::ParIdentity{T}) where {T} = Dict{String, Any}("type" => "ParIdentity", "T" => string(T), "n" => A.n)

function from_Dict(::Type{ParIdentity}, d)
    ts = d["T"]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    ParIdentity(dtype, d["n"])
end

kron(A::ParIdentity{T}, B::ParIdentity{T}) where {T} = ParIdentity(T,B.n*A.n)
