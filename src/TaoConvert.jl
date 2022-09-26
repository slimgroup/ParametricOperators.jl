export TaoConvert

struct TaoConvert{D,R,L} <: TaoOperator{D,R,L,NonParametric,External}
    n::Int64
end

TaoConvert{D,R}(n::Int64) where {D,R} = TaoConvert{D,R,NonLinear}(n)
TaoConvert{T,Complex{T}}(n::Int64) where {T<:Real} = TaoConvert{T,Complex{T},Linear}(n)
TaoConvert{Complex{T},T}(n::Int64) where {T<:Real} = TaoConvert{Complex{T},T,Linear}(n)

Domain(C::TaoConvert) = C.n
Range(C::TaoConvert) = C.n
id(::TaoConvert{D,R}) where {D,R} = "convert_[$D,$R]"
adjoint(C::TaoConvert{D,R}) where {D,R} = TaoConvert{R,D}(C.n)

(C::TaoConvert{D,R})(x::X) where {D,R,X<:AbstractVector{D}} = R.(x)