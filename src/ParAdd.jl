export ParAdd

promote_optype_add(::Nothing, ::Nothing) = Nothing
promote_optype_add(::Type{T}, ::Nothing) where {T} = T
promote_optype_add(::Nothing, ::Type{T}) where {T} = T
promote_optype_add(::Type{T}, ::Type{T}) where {T} = T

struct ParAdd{D,R,L,P,F} <: ParOperator{D,R,L,P,HigherOrder}
    ops::F
    m::Int64
    n::Int64
    slots::Vector{Int64}
    ranges::Vector{UnitRange{Int64}}
    id::ID
    function ParAdd(ops)

        D = foldr(promote_optype_add, map(DDT, ops); init=Nothing)
        R = foldl(promote_optype_add, map(RDT, ops); init=Nothing)
        L = foldl(promote_linearity, map(linearity, ops); init=Linear)
        P = foldl(promote_parametricity, map(parametricity, ops); init=NonParametric)

        m = foldl(promote_opdim, map(Range, ops))
        n = foldr(promote_opdim, map(Domain, ops))

        slots = collect(map(first, filter(t -> nparams(second(t)) > 0, enumerate(ops))))
        nps = collect(map(nparams, ops[slots]))
        offsets = [0, cumsum(nps[1:end-1])...]
        starts = offsets .+ 1
        stops = offsets .+ nps
        ranges = [start:stop for (start, stop) in (starts, stops)]

        return new{D,R,L,P,typeof(ops)}(ops, m, n, slots, ranges, uuid4(GLOBAL_RNG))

    end
    ParAdd(ops...) = ParAdd(collect(ops))
end

+(ops::ParOperator...) = ParAdd(ops...)

Domain(A::ParAdd) = A.n
Range(A::ParAdd) = A.m
children(A::ParAdd) = A.ops
id(A::ParAdd) = A.id

function (A::ParAdd{D,R,L,Parametric,F})(θ) where {D,R,L,F}
    ops_out
end