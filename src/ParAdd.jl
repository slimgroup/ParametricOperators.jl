export ParAdd

promote_optype_add(::Type{Nothing}, ::Type{Nothing}) = Nothing
promote_optype_add(::Type{T}, ::Type{Nothing}) where {T} = T
promote_optype_add(::Type{Nothing}, ::Type{T}) where {T} = T
promote_optype_add(::Type{T}, ::Type{T}) where {T} = T

struct ParAdd{D,R,L,P,F} <: ParOperator{D,R,L,P,HigherOrder}
    ops::F
    m::Int64
    n::Int64
    ranges::Vector{Option{UnitRange{Int64}}}
    id::ID
    
    function ParAdd(ops)

        D = foldr(promote_optype_add, map(DDT, ops); init=Nothing)
        R = foldr(promote_optype_add, map(RDT, ops); init=Nothing)
        L = foldr(promote_linearity, map(linearity, ops); init=Linear)
        P = foldr(promote_parametricity, map(parametricity, ops); init=NonParametric)

        m = foldl(promote_opdim, map(Range, ops); init=nothing)
        n = foldr(promote_opdim, map(Domain, ops); init=nothing)

        nps = collect(map(nparams, ops))
        offsets = [0, cumsum(nps[1:end-1])...]
        starts = offsets .+ 1
        stops = offsets .+ nps
        ranges = [start:stop for (start, stop) in zip(starts, stops)]
        ranges = collect(map(r -> length(r) == 0 ? nothing : r, ranges))

        return new{D,R,L,P,typeof(ops)}(ops, m, n, ranges, uuid4(GLOBAL_RNG))
    end
end

+(ops::ParOperator...) = ParAdd(collect(ops))

Domain(A::ParAdd) = A.n
Range(A::ParAdd) = A.m
children(A::ParAdd) = A.ops
id(A::ParAdd) = A.id

adjoint(A::ParAdd{D,R,Linear,P,F}) where {D,R,P,F} = ParAdd(collect(map(adjoint, A.ops)))

function (A::ParAdd{D,R,L,Parametric,F})(θ::V) where {D,R,L,F,V}
    ops_out = [isnothing(r) ? op : op(view(θ, r)) for (op, r) in zip(A.ops, A.ranges)]
    return ParAdd(ops_out)
end

function (A::ParAdd{D,R,L,P,F})(x::X) where {D,R,L,P<:Applicable,F,X<:AbstractVector{D}}
    return sum([op(x) for op in A.ops])
end

function (A::ParAdd{D,R,L,P,F})(x::X) where {D,R,L,P<:Applicable,F,X<:AbstractMatrix{D}}
    return sum([op(x) for op in A.ops])
end