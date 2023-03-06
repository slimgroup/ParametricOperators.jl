export elementwise_addition_cost
export elementwise_multiplication_cost
export complexity

"""
Estimated cost of performing a*b.
"""
elementwise_multiplication_cost(::Type{Float16})    = 0.5
elementwise_multiplication_cost(::Type{Float32})    = 1.0
elementwise_multiplication_cost(::Type{Float64})    = 2.0
elementwise_multiplication_cost(::Type{ComplexF16}) = 1.0
elementwise_multiplication_cost(::Type{ComplexF32}) = 2.0
elementwise_multiplication_cost(::Type{ComplexF64}) = 4.0

"""
Estimated cost of performing a+b.
"""
elementwise_addition_cost(::Type{Float16})    = 0.25
elementwise_addition_cost(::Type{Float32})    = 1.5
elementwise_addition_cost(::Type{Float64})    = 1.0
elementwise_addition_cost(::Type{ComplexF16}) = 0.5
elementwise_addition_cost(::Type{ComplexF32}) = 1.0
elementwise_addition_cost(::Type{ComplexF64}) = 2.0

"""
Estimated cost of moving a single byte of data from worker src to worker dest.
"""
byte_transfer_cost(::Int, ::Int) = 100.0

"""
Estimated cost of operator application on a vector.
"""
complexity(::ParOperator{D,R,L,P,External}) where {D,R,L,P} = throw(ParException("Unimplemented"))

"""
A nominal cost for syntactic complexity, so that we prefer simpler solutions.
"""
syntactic_complexity_cost = 0.00001

"""
Complexity of operator combinations defaults to sum of child complexity.
"""
complexity(A::ParOperator{D,R,L,P,Internal}) where {D,R,L,P} = sum(map(complexity, children(A))) + syntactic_complexity_cost

"""
Complexity of an operator local to a single worker. Defaults to complexity.
"""
local_complexity(A) = complexity(A)