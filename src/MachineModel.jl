export elementwise_addition_cost
export elementwise_multiplication_cost
export complexity

elementwise_multiplication_cost(::Type{Float16})    = 0.5
elementwise_multiplication_cost(::Type{Float32})    = 1.0
elementwise_multiplication_cost(::Type{Float64})    = 2.0
elementwise_multiplication_cost(::Type{ComplexF16}) = 1.0
elementwise_multiplication_cost(::Type{ComplexF32}) = 2.0
elementwise_multiplication_cost(::Type{ComplexF64}) = 4.0

elementwise_addition_cost(::Type{Float16})    = 0.25
elementwise_addition_cost(::Type{Float32})    = 1.5
elementwise_addition_cost(::Type{Float64})    = 1.0
elementwise_addition_cost(::Type{ComplexF16}) = 0.5
elementwise_addition_cost(::Type{ComplexF32}) = 1.0
elementwise_addition_cost(::Type{ComplexF64}) = 2.0

"""
Estimated cost of operator application on a vector.
"""
complexity(::ParOperator{D,R,L,P,External}) where {D,R,L,P} = throw(ParException("Unimplemented"))

"""
Complexity of operator combinations defaults to sum of child complexity.
"""
complexity(A::ParOperator{D,R,L,P,Internal}) where {D,R,L,P} = sum(map(complexity, children(A)))