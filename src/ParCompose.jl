export ParCompose

abstract type ParamOrder end
struct LeftFirst <: ParamOrder end
struct RightFirst <: ParamOrder end

struct ParCompose{D,R,L,P,F,O<:ParamOrder} <: ParOperator{D,R,L,P,Internal}
    ops::F
    m::Int64
    n::Int64
    slots::Optional{Set{Int64}}
    ranges::Optional{Vector{UnitRange{Int64}}}
    id::ID
    function ParCompose(ops::ParOperator...; order=nothing, slots=nothing, ranges=nothing)

        m = Range(ops[1])
        n = Domain(ops[end])
        for i in 1:length(ops)-1
            promote_opdim(Domain(ops[i]), Range(ops[i+1]))
        end

        D = DDT(ops[end])
        R = RDT(ops[1])
        L = foldl(promote_linearity, map(linearity, ops); init = Linear)
        P = foldl(promote_parametricity, map(parametricity, ops); init = NonParametric)
        O = isnothing(order) ? LeftFirst : order

        if P <: Applicable
            return new{D,R,L,P,typeof(ops),O}(ops, m, n, nothing, nothing, uuid4(GLOBAL_RNG))
        else
            pop_tups = Iterators.filter(tup -> parametricity(tup[2]) == Parametric, enumerate(ops))
            slots = isnothing(slots) ? Set(map(tup -> tup[1], pop_tups)) : slots
    
            nps = collect(map(tup -> nparams(tup[2]), pop_tups))
            offsets = [0, cumsum(nps[1:end-1])...]
            starts = offsets .+ 1
            stops = offsets .+ nps
            ranges = isnothing(ranges) ? [start:stop for (start, stop) in zip(starts, stops)] : ranges
    
            return new{D,R,L,P,typeof(ops),O}(ops, m, n, slots, ranges, uuid4(GLOBAL_RNG))
        end
    end
end

∘(ops::ParOperator...) = ParCompose(ops...)
*(ops::ParLinearOperator...) = ParCompose(ops...)

Domain(A::ParCompose) = A.n
Range(A::ParCompose) = A.m
children(A::ParCompose) = A.ops
id(A::ParCompose) = A.id

adjoint(A::ParCompose{D,R,Linear,P,F,LeftFirst}) where {D,R,P,F} = ParCompose(map(adjoint, reverse(A.ops))...; order = RightFirst, slots = A.slots, ranges = A.ranges)
adjoint(A::ParCompose{D,R,Linear,P,F,RightFirst}) where {D,R,P,F} = ParCompose(map(adjoint, reverse(A.ops))...; order = LeftFirst, slots = A.slots, ranges = A.ranges)

(A::ParCompose{D,R,L,Parametric,F,LeftFirst})(θ) where {D,R,L,F} =
    ParCompose([i ∈ A.slots ? A.ops[i](@view θ[A.ranges[i]]) : A.ops[i] for i in 1:length(A.ops)]...)

(A::ParCompose{D,R,L,Parametric,F,RightFirst})(θ) where {D,R,L,F} =
    ParCompose([i ∈ A.slots ? A.ops[i](@view θ[A.ranges[length(A.ops)-i+1]]) : A.ops[i] for i in 1:length(A.ops)]...)

function (A::ParCompose{D,R,L,P,F,O})(x::X) where {D,R,L,P<:Applicable,F,O,X<:AbstractVector{D}}
    y = x
    for i = length(A.ops):-1:1
        y = A.ops[i](y)
    end
    return y
end

function (A::ParCompose{D,R,L,P,F,O})(x::X) where {D,R,L,P<:Applicable,F,O,X<:AbstractMatrix{D}}
    y = x
    for i = length(A.ops):-1:1
        y = A.ops[i](y)
    end
    return y
end