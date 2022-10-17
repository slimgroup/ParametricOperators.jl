export ID, Optional, ParException
export project_type

const ID = Union{UUID, String}
const Optional{T} = Union{T, Nothing}

struct ParException <: Exception
    msg::String
end

project_type(T, x) = convert(T, x)
project_type(::Type{T}, x::AbstractArray{T}) where {T} = x
project_type(::Type{T}, x::AbstractArray{U}) where {T,U} = T.(x)
project_type(U::Type{T}, x::AbstractArray{S}) where {T<:Real,S<:Number} = U.(real(x))

promote_optype(::Type{T}, ::Type{T}) where {T} = T
promote_optype(::Type{Nothing}, ::Type{T}) where {T} = T
promote_optype(::Type{T}, ::Type{Nothing}) where {T} = T

promote_opdim(::Nothing, d::I) where {I<:Integer} = d
promote_opdim(d::I, ::Nothing) where {I<:Integer} = d
promote_opdim(d1::I1, d2::I2) where {I1<:Integer,I2<:Integer} =
    d1 == d2 ? d1 : throw(ParException("Incompatible operator dimensions"))

embedding_type(::Type{T}, ::Type{T}) where {T} = T
embedding_type(::Type{T}, ::Type{Complex{T}}) where {T<:Real} = Complex{T}
embedding_type(::Type{Complex{T}}, ::Type{T}) where {T<:Real} = Complex{T}