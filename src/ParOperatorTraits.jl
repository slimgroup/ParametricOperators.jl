export DDT, RDT, linearity, parametricity, ast_location
export Domain, Range, nparams, children, init, id, optimize

DDT(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = D
RDT(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = R
linearity(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = L
parametricity(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = P
ast_location(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = T

#Domain(F::ParOperator)   = throw(ParException("Domain() is not implemented for $(typeof(F))"))
#Range(F::ParOperator)    = throw(ParException("Range() is not implemented for $(typeof(F))"))
#nparams(F::ParOperator)  = throw(ParException("nparams() is not implemented for $(typeof(F))"))
#children(F::ParOperator) = throw(ParException("children() is not implemented for $(typeof(F))"))
#init(F::ParOperator)     = throw(ParException("init() is not implemented for $(typeof(F))"))
#id(F::ParOperator)       = throw(ParException("id() is not implemented for $(typeof(F))"))

Domain(::ParOperator{Nothing,R,L,P,T}) where {R,L,P,T} = nothing
Range(::ParOperator{D,Nothing,L,P,T}) where {D,L,P,T} = nothing

children(::ParOperator{D,R,L,NonParametric,T}) where {D,R,L,T} = Vector{ParOperator}()

nparams(F::ParOperator{D,R,L,NonParametric,T}) where {D,R,L,T} = 0
nparams(F::ParOperator{D,R,L,Parametric,Internal}) where {D,R,L} = sum(map(nparams, children(F)))

params(F::ParOperator{D,R,L,NonParametric,T}) where {D,R,L,T} = []
params(F::ParOperator{D,R,L,Parametric,T}) where {D,R,L,T} = []
params(F::ParOperator{D,R,L,Parameterized,Internal}) where {D,R,L} =
    MultiTypeVector(Number, map(init, Iterators.filter(v -> parametricity(v) == Parameterized, children(F))))

init(::ParOperator{D,R,L,NonParametric,T}) where {D,R,L,T} = Vector{Number}()
init(F::ParOperator{D,R,L,Parametric,Internal}) where {D,R,L} =
    MultiTypeVector(Number, map(init, Iterators.filter(v -> parametricity(v) == Parametric, children(F)))...)

(F::ParOperator{D,R,L,P,T})(::X) where {D,R,L,P<:Applicable,T,X<:AbstractVector{D}} =
    throw(ParException("F(x) is not implemented for $(typeof(F))"))

function (A::ParOperator{D,R,L,P,T})(x::X) where {D,R,L,P<:Applicable,T,X<:AbstractMatrix{D}}
    nc = size(x)[2]
    cols = [x[:,i] for i in 1:nc]
    return reduce(hcat, broadcast(A, cols))
end

(F::ParOperator{D,R,L,Parametric,T})(x::X, θ) where {D,R,L,T,X<:AbstractVecOrMat{D}} = F(θ)(x)
*(A::ParOperator{D,R,Linear,P,T}, x::X) where {D,R,P<:Applicable,T,X<:AbstractVecOrMat{D}} = A(x)

function *(x::X, A::ParLinearOperator{D,R,P,T}) where {D,R,P<:Applicable,T,X<:AbstractMatrix{D}}
    nr = size(x)[1]
    rows = [x[i,:] for i in 1:nr]
    return reduce(vcat, broadcast(r -> transpose(A*r), rows))
end

optimize(F::ParOperator) = F