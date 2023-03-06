export ParParameterized

"""
Lazy container for parameterization.
"""
struct ParParameterized{D,R,L,F<:ParParametricOperator,V} <: ParOperator{D,R,L,Parameterized,Internal}
    op::F
    params::V
    ParParameterized(op, params) = new{DDT(op),RDT(op),linearity(op),typeof(op), typeof(params)}(op, params)
end

Domain(A::ParParameterized) = Domain(A.op)
Range(A::ParParameterized) = Range(A.op)
children(A::ParParameterized) = [A.op]
rebuid(A::ParParameterized, cs) = ParParameterized(cs[1], A.params)
adjoint(A::ParParameterized) = ParParameterized(adjoint(A.op), A.params)
params(A::ParParameterized) = A.params

"""
Parameterize an external operator with a set of params.
"""
(A::ParParametricOperator{D,R,L,External})(params) where {D,R,L} = ParParameterized(A, params[A])