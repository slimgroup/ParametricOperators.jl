using LinearAlgebra
using MLDatasets
using Tao
using Zygote
using Flux

function sigmoid(x::T) where {T}
    return T.(1)/(T.(1) + exp(-x))
end

function perceptron(T, n_in::Int64, n_out::Int64, act::Bool = true)
    W = MatrixOperator{T}(n_out, n_in)
    b = BiasOperator{T}(n_out)
    σ = sigmoid
    return act ? σ.(W+b) : W+b
end

model(T) =
    perceptron(T, 64, 10, false) ∘
    perceptron(T, 128, 64) ∘
    perceptron(T, 784, 128)

f = model(Float32)
pv = ParameterVector{Float32}()
init(f, pv)

function one_hot(T, idx)
    v = zeros(T, 10)
    v[idx] = 1.0
    return v
end

X, Y = MLDatasets.MNIST.traindata(Float32)
x, y = vec(X[:,:,1]), one_hot(Float32, Y[1])

L = Flux.logitbinarycrossentropy
η = 0.0001f32

for i in 1:100
    g = gradient(θ -> begin
        ŷ = f(x, θ)
        l = L(ŷ, y)
        @show ŷ l
        return l
    end, pv)[1]
    @show norm(g.data)
    if isnan(norm(g.data)) || norm(g.data) < 1e-8
        break
    end
    global pv = pv - η*g
end

ŷ = f(x, pv)
@show ŷ