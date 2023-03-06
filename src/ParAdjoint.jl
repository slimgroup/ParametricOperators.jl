export ParAdjoint

"""
Lazy container for adjoint.
"""
struct ParAdjoint{D,R,P,F<:ParLinearOperator} <: ParLinearOperator{R,D,P,Internal}
    op::F
    ParAdjoint(op) = new{DDT(op),RDT(op),parametricity(op),typeof(op)}(op)
end

adjoint(A::ParLinearOperator) = ParAdjoint(A)
adjoint(A::ParAdjoint) = A.op

Domain(A::ParAdjoint) = Range(A.op)
Range(A::ParAdjoint) = Domain(A.op)
children(A::ParAdjoint) = [A.op]
rebuild(::ParAdjoint, cs) = ParAdjoint(cs[1])

(A::ParAdjoint{D,R,Parametric,F})(params) where {D,R,F} = ParParameterized(adjoint(A.op), params)

latex_equation(A::ParAdjoint) = "$(latex_equation(A.op))^\\top"

to_Dict(A::ParAdjoint) = Dict{String, Any}("type" => "ParAdjoint", "of" => to_Dict(A.op))

function from_Dict(::Type{ParAdjoint}, d)
    op = from_Dict(d["of"])
    ParAdjoint(op)
end
