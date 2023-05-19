export ParDFT, ParDFTN

"""
Fouier transform operator. Dispatchs real-valued fft if applicable.
"""
struct ParDFT{D,R} <: ParLinearOperator{D,R,NonParametric,External}
    m::Int
    n::Int
    function ParDFT(T, n)
        if T <: Real
            return new{T,Complex{T}}(n÷2+1, n)
        elseif T <: Complex
            return new{T,T}(n, n)
        else
            throw(ParException("Invalid type $T for DFT"))
        end
    end
    ParDFT(n) = ParDFT(ComplexF64, n)
end

Domain(A::ParDFT) = A.n
Range(A::ParDFT) = A.m

complexity(A::ParDFT{D,R}) where {D,R} = elementwise_multiplication_cost(R)*A.n*log2(A.n)

(A::ParDFT{D,R})(x::X) where {D<:Complex,R,X<:AbstractMatrix{D}} = convert(Matrix{R}, fft(x, 1) ./ sqrt(A.n))
(A::ParDFT{D,R})(x::X) where {D<:Real,R,X<:AbstractMatrix{D}} = convert(Matrix{R}, rfft(x, 1) ./ sqrt(A.n))
(A::ParDFT{D,R})(x::X) where {D,R,X<:AbstractVector{D}} = vec(A(reshape(x, length(x), 1)))

(A::ParAdjoint{D,R,NonParametric,ParDFT{D,R}})(x::X) where {D<:Complex,R,X<:AbstractMatrix{R}} = ifft(x, 1).*convert(real(D), sqrt(A.op.n))
(A::ParAdjoint{D,R,NonParametric,ParDFT{D,R}})(x::X) where {D<:Real,R,X<:AbstractMatrix{R}} = irfft(x, A.op.n, 1).*convert(D, sqrt(A.op.n))
(A::ParAdjoint{D,R,NonParametric,ParDFT{D,R}})(x::X) where {D,R,X<:AbstractVector{R}} = vec(A(reshape(x, length(x), 1)))

to_Dict(A::ParDFT{D,R}) where {D,R} = Dict{String, Any}("type" => "ParDFT", "T" => string(D), "n" => A.n, "m" => A.m)

function from_Dict(::Type{ParDFT}, d)
     ts = d["T"]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    A = ParDFT(dtype, d["n"])
    # the "m" is informational, but if it's present, make sure it matches
    if haskey(d, "m") && A.m != d["m"]
        expected_m = d["m"]
        throw(ParException("range mismatch: data says $expected_m, but parDFT calculated $(A.m)"))
    end
    A
end

"""
Fouier transform operator for n dimensional tensor.
"""
struct ParDFTN{N,T} <: ParLinearOperator{T,T,NonParametric,External}
    shape::NTuple{N, Int}
    n:: Int
    ParDFTN(ns...; T = ComplexF32) = new{length(ns),T}(ns, prod(ns))
end

Domain(A::ParDFTN) = A.n
Range(A::ParDFTN) = A.n

(A::ParDFTN{N,T})(x::X) where {N,T<:Complex,X<:AbstractVector{T}} = vec(fft(reshape(x, A.shape)) ./ T(sqrt(A.n)))

(A::ParAdjoint{D,R,NonParametric,ParDFTN{N,T}})(x::X) where {D<:Complex,R,N,T<:Complex,X<:AbstractVector{T}} = vec(ifft(reshape(x, A.op.shape)) .* T(sqrt(A.op.n)))

kron(A::ParDFT{T,T}, B::ParDFT{T,T}) where {T} = ParDFTN(B.n, A.n, T=T)
kron(A::ParDFT{T,T}, B::ParDFTN{N,T}) where {N,T} = ParDFTN(B.shape..., A.n, T=T)
kron(A::ParDFTN{N,T}, B::ParDFT{T,T}) where {N,T} = ParDFTN(B.n, A.shape..., T=T)
kron(A::ParDFTN{N,T}, B::ParDFTN{M,T}) where {M,N,T} = ParDFTN(B.shape..., A.shape..., T=T)
