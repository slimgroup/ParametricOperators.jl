struct TaoException <: Exception
    msg::String
end

Optional{T} = Union{T, AbstractArray}