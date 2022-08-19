"""
Package level exception type.
"""
struct TaoException
    msg::String
end

"""
Optional type holding either a T or nothing.
"""
Optional{T} = Union{T, Nothing}

"""
Alias for array shape.
"""
Shape = Union{Vector{Int64},Tuple{Vararg{Int64}}}

"""
Determines whether a type `T` is subset of another type `U`. E.g.
`Float32` ⊂ `ComplexF32`.
"""
issubsettypeof(t::T, u::U) where {T,U} = false           # default to false
issubsettypeof(::T, ::T) where {T} = true                # identity is true
issubsettypeof(::T, ::Complex{T}) where {T<:Real} = true # T ⊂ Complex{T}