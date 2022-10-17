export ParAdd

struct ParAdd{D,R,L,P,F} <: ParOperator{D,R,L,P,Internal}
    ops::F
    m::Int64
    n::Int64
    slots::Optional{Set{Int64}}
    ranges::Optional{Vector{UnitRange{Int64}}}
    id::ID
    function ParAdd(ops::ParOperator...)

        D = foldl(promote_optype, map(DDT, ops); init = Nothing)
        R = foldl(promote_optype, map(RDT, ops); init = Nothing)
        m = foldl(promote_opdim, map(Range, ops); init = nothing)
        n = foldl(promote_opdim, map(Domain, ops); init = nothing)

        L = foldl(promote_linearity, map(linearity, ops); init = Linear)
        P = foldl(promote_parametricity, map(parametricity, ops); init = NonParametric)

        if P <: Applicable
            return new{D,R,L,P,typeof(ops)}(ops, m, n, nothing, nothing, uuid4(GLOBAL_RNG))
        else
            pop_tups = Iterators.filter(tup -> parametricity(tup[2]) == Parametric, enumerate(ops))
            slots = Set(map(tup -> tup[1], pop_tups))

            nps = collect(map(tup -> nparams(tup[2]), pop_tups))
            offsets = [0, cumsum(nps[1:end-1])...]
            starts = offsets .+ 1
            stops = offsets .+ nps
            ranges = [start:stop for (start, stop) in zip(starts, stops)]
    
            return new{D,R,L,P,typeof(ops)}(ops, m, n, slots, ranges, uuid4(GLOBAL_RNG))
        end
    end
end

+(ops::ParOperator...) = ParAdd(ops...)

Domain(A::ParAdd) = A.n
Range(A::ParAdd) = A.m
children(A::ParAdd) = A.ops
id(A::ParAdd) = A.id

adjoint(A::ParAdd{D,R,Linear,P,F}) where {D,R,P,F} = ParAdd(map(adjoint, A.ops)...)

(A::ParAdd{D,R,L,Parametric,F})(θ) where {D,R,L,F} =
    ParAdd([i ∈ A.slots ? A.ops[i](@view θ[A.ranges[i]]) : A.ops[i] for i in 1:length(A.ops)]...)

(A::ParAdd{D,R,L,P,F})(x::X) where {D,R,L,P<:Applicable,F,X<:AbstractVector{D}} = +([op(x) for op in A.ops]...)
(A::ParAdd{D,R,L,P,F})(x::X) where {D,R,L,P<:Applicable,F,X<:AbstractMatrix{D}} = +([op(x) for op in A.ops]...)