export ParCompose

struct ParCompose{D,R,L,P,F} <: ParOperator{D,R,L,P,Internal}
    ops::F
    m::Int64
    n::Int64
    ranges::Vector{UnitRange{Int64}}
    slots::Set{Int64}
    id::ID
    function ParCompose(ops::ParOperator...; ranges=nothing)
        D_out = DDT(ops[end])
        R_out = RDT(ops[1])
        for i in 1:length(ops)-1
            @assert DDT(ops[i]) == RDT(ops[i+1])
            @assert Domain(ops[i]) == Range(ops[i+1]) "failed with $(typeof(ops[i])) $(typeof(ops[i+1]))"
        end
        L_out = foldl((l1, l2) -> promote_linearity(l1, l2), map(linearity, ops); init = Linear)
        P_out = foldl((p1, p2) -> promote_parametricity(p1, p2), map(parametricity, ops); init = NonParametric)
        if isnothing(ranges)
            offsets = [0, cumsum(map(nparams, ops[1:end-1]))...]
            starts = offsets .+ 1
            stops = [o+np for (o, np) in zip(offsets, map(nparams, ops))]
            ranges = [start:stop for (start, stop) in zip(starts, stops)]
        end
        N = length(ops)
        slots = Set([parametricity(ops[i]) == Parametric ? i : -1 for i ∈ 1:N])
        return new{D_out,R_out,L_out,P_out,typeof(ops)}(ops, Range(ops[1]), Domain(ops[end]), ranges, slots, uuid4(GLOBAL_RNG))
    end
end

∘(ops::ParOperator...) = ParCompose(ops...)
*(ops::ParLinearOperator...) = ParCompose(ops...)

Domain(A::ParCompose) = A.n
Range(A::ParCompose) = A.m
children(A::ParCompose) = A.ops
id(A::ParCompose) = A.id
adjoint(A::ParCompose{D,R,Linear,P,F}) where {D,R,P,F} = ParCompose(map(adjoint, reverse(A.ops))...; ranges=reverse(A.ranges))

function (A::ParCompose{D,R,L,Parametric,F})(θ::AbstractVector{<:Number}) where {D,R,L,F}
    N = length(A.ranges)
    ops_out = [i ∈ A.slots ? A.ops[i](θ[A.ranges[i]]) : A.ops[i] for i in 1:N]
    ParCompose(ops_out...)
end

function (A::ParCompose{D,R,L,P,F})(x::X) where {D,R,L,P<:Applicable,F,X<:AbstractVector{D}}
    y = x
    for op in reverse(A.ops)
        y = op(y)
    end
    return y
end