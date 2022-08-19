struct FunctionOperator{D,R,F<:Function} <: AbstractOperator{D,R}
    n::Int64
    m::Int64
    f::F
    id::Any
end

FunctionOperator{D,R}(f::F) where {D,R,F<:Function} = FunctionOperator{D,R,F}(-1, -1, f, uid())
FunctionOperator{D,R}(f::F, n::Int64, m::Int64) where {D,R,F<:Function} = FunctionOperator{D,R,F}(n, m, f, uid())

Domain(F::FunctionOperator) = F.n
Range(F::FunctionOperator) = F.m
params(F::FunctionOperator) = []
nparams(F::FunctionOperator) = 0
id(F::FunctionOperator) = F.id
init(F::FunctionOperator, pc::Optional{ParameterContainer} = nothing) = []

for vectype in [:AbstractVector, :AbstractVecOrMat]
    @eval begin
        function (G::FunctionOperator{D,R,F})(x::V, ::Vararg{<:AbstractArray}) where
            {
                D,
                R,
                F<:Function,
                V<:$vectype{D}
            }
            return G.f(x)
        end

        function (G::FunctionOperator{D,R,F})(x::V, ::ParameterContainer) where
            {
                D,
                R,
                F<:Function,
                V<:$vectype{D}
            }
            return G.f(x)
        end
    end
end