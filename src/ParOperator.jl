export ParOperator, ParLinearOperator
export DDT, RDT, Domain, Range, linearity, parametricity, ast_location
export init!, init, scale!, update!
export cpu, gpu
export children, params

# ==== Type Definitions ====

"""
Typeflag for whether a given operator is linear.
"""
abstract type Linearity end
struct Linear <: Linearity end
struct NonLinear <: Linearity end

"""
Linearity promotion rules.
"""
promote_linearity(::Type{Linear}, ::Type{Linear}) = Linear
promote_linearity(::Type{<:Linearity}, ::Type{<:Linearity}) = NonLinear

"""
Typeflag for whether a given operator is parametric, nonparametric, or parameterized
with some given parameters.
Note: A distinction is made for parametric vs. parameterized to allow for proper
method dispatch.
"""
abstract type Parametricity end
struct Parametric <: Parametricity end
struct NonParametric <: Parametricity end
struct Parameterized <: Parametricity end

"""
Applicable types can act on vectors.
"""
const Applicable = Union{NonParametric, Parameterized}

"""
HasParams types have parameter slots or associated parameters.
"""
const HasParams = Union{Parametric, Parameterized}

"""
Parametricity promotion rules.
"""
promote_parametricity(::Type{NonParametric}, ::Type{NonParametric}) = NonParametric
promote_parametricity(::Type{NonParametric}, ::Type{Parameterized}) = Parameterized
promote_parametricity(::Type{Parameterized}, ::Type{NonParametric}) = Parameterized
promote_parametricity(::Type{Parameterized}, ::Type{Parameterized}) = Parameterized
promote_parametricity(::Type{<:Parametricity}, ::Type{<:Parametricity}) = Parametric

"""
Typeflag for whether a given operator is an external or internal node in the AST
generated by combining operators together.
"""
abstract type ASTLocation end
struct Internal <: ASTLocation end
struct External <: ASTLocation end

"""
Base operator type.
"""
abstract type ParOperator{D,R,L<:Linearity,P<:Parametricity,T<:ASTLocation} end

"""
Linear operator type (defined for convenience).
"""
const ParLinearOperator{D,R,P,T} = ParOperator{D,R,Linear,P,T}

"""
Parametric operator type (defined for convenience).
"""
const ParParametricOperator{D,R,L,T} = ParOperator{D,R,L,Parametric,T}

# ==== Trait Definitions ====

"""
Domain datatype of the given operator.
"""
DDT(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = D

"""
Range datatype of the given operator.
"""
RDT(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = R

"""
Linearity of the given operator.
"""
linearity(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = L

"""
Parametricity of the given operator.
"""
parametricity(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = P

"""
AST location of the given operator.
"""
ast_location(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = T

"""
Domain of the given operator. In parallel computation, corresponds to the local
domain size.
"""
Domain(::ParOperator) = throw(ParException("Unimplemented"))

"""
Range of the given operator. In parallel computation, corresponds to the local
range size.
"""
Range(::ParOperator) = throw(ParException("Unimplemented"))

"""
Children of the given operator. For external nodes, this is an empty list.
"""
children(::ParOperator{D,R,L,P,External}) where {D,R,L,P} = []
children(::ParOperator{D,R,L,P,Internal}) where {D,R,L,P} = throw(ParException("Unimplemented"))

"""
Rebuild the given operator using a vector of new children.
"""
rebuild(A::ParOperator{D,R,L,P,External}, _) where {D,R,L,P} = A
rebuild(::ParOperator{D,R,L,P,Internal}, _) where {D,R,L,P} = throw(ParException("Unimplemented"))

"""
Parameter dict typedef.
"""
const Parameters = Dict{<:ParOperator,Any}

"""
Move objects to cpu.
"""
cpu(x::CuArray{<:Number}) = Array(x)
cpu(x::Vector{CuArray}) = [cpu(y) fpr y in x]
cpu(x::AbstractArray) = x
cpu(x::Parameters) = Dict(k => cpu(v) for (k, v) in pairs(x))

if CUDA.functional()
    """
    Move objects to gpu.
    """
    gpu(x::AbstractArray{<:Number}) = CuArray(x)
    gpu(x::Vector{<:AbstractArray}) = [gpu(y) fpr y in x]
    gpu(x::CuArray) = x
    gpu(x::Parameters) = Dict(k => gpu(v) for (k, v) in pairs(x))
end

for op in [:+, :-, :*, :/, :^]
    @eval $op(p0::Parameters, p1::Parameters) = mergewith(BroadcastFunction($op), p0, p1)
    @eval $op(p0::Parameters, p1::Dict) = mergewith(BroadcastFunction($op), p0, p1)
    @eval $op(p0::Dict, p1::Parameters) = mergewith(BroadcastFunction($op), p0, p1)
end

"""
Update parameters with gradients.
"""
update!(params::Parameters, grads::Dict) = mergewith!(BroadcastFunction(-), params, grads)

"""
Scale parameters with a number.
"""
function scale!(a::Number, params::Dict)
    for k in keys(params)
        scale!(a, params[k])
    end
end

function scale!(a::Number, x::AbstractArray{<:AbstractArray})
    for v in x
        scale!(a, v)
    end
end

function scale!(a::Number, x::AbstractArray{<:Number})
    x .*= a
end

"""
Initialize the given operator into the given dictionary.
"""
init!(::ParOperator{D,R,L,<:Applicable,T}, d::Parameters) where {D,R,L,T} = nothing

function init!(A::ParOperator{D,R,L,Parametric,Internal}, d::Parameters) where {D,R,L}
    for c in children(A)
        init!(c, d)
    end
    return d
end

"""
Initialize the given operator(s) creating a new dictionary.
"""
function init(As::ParOperator...)
    d = Dict{ParOperator,Any}()
    for A in As
        init!(A, d)
    end
    return d
end

"""
Parameterize the given operator
"""
(A::ParOperator{D,R,L,Parametric,Internal})(params) where {D,R,L} =
    rebuild(A, collect(map(c -> parametricity(c) == Parametric ? c(params) : c, children(A))))

"""
Apply the given operator on a vector.
"""
(A::ParOperator{D,R,L,<:Applicable,T})(::X) where {D,R,L,T,X<:AbstractVector{D}} = throw(ParException("Unimplemented"))

"""
Apply the given operator to a matrix. By default, apply to each of the columns.
"""
(A::ParOperator{D,R,L,<:Applicable,T})(x::X) where {D,R,L,T,X<:AbstractMatrix{D}} = mapreduce(col -> A(col), hcat, eachcol(x))

"""
Apply a linear operator to a vector or matrix through multiplication.
"""
*(A::ParOperator{D,R,L,<:Applicable,T}, x::X) where {D,R,L,T,X<:AbstractVector{D}} = A(x)
*(A::ParOperator{D,R,L,<:Applicable,T}, x::X) where {D,R,L,T,X<:AbstractMatrix{D}} = A(x)

"""
Apply a matrix to a linear operator. By default, use rules of the adjoint.
"""
*(x::X, A::ParLinearOperator{D,R,<:Applicable,T}) where {D,R,T,X<:AbstractMatrix{R}} = (A'*x')'

"""
Serialize the given operator to a Dict{String, Any}, suitable for encoding to json, toml, msgpack, yaml, etc
"""
to_Dict(::ParOperator) = throw(ParException("Unimplemented"))
