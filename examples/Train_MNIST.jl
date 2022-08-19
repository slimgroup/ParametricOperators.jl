using Base.Iterators
using LinearAlgebra
using Match
using MLDatasets
using Random
using Serialization
using Tao
using Zygote
using Flux

function model(T)
    σ = Flux.NNlib.relu
    L1 = σ ∘ (MatrixOperator{T}(32, 784) + BiasOperator{T}(32))
    L2 = MatrixOperator{T}(10, 32)
    return softmax ∘ L2 ∘ L1
end

function one_hot(T, i::U, n::Int64) where {U}
    v = zeros(T, n)
    v[Int64(i)] = T(1)
    return v
end

function train()

    Random.seed!(123)

    T = Float32
    X, Y = MNIST.traindata(T)
    Y = mapreduce(y -> one_hot(T, y+1, 10), hcat, Y)

    F = model(T)
    L = Flux.crossentropy

    bs = 100
    ne = 200
    nd = size(X)[3]
    
    θ = ParameterContainer()
    init(F, θ)
    @show length(θ.data[T])

    for e in 1:ne
        println("epoch = $(e)")
        batches = shuffle(collect(map(i -> i:i+bs-1, 1:bs:nd)))
        avg_loss = T(0)
        
        η = @match e begin
            1:100 => T(2e-2)
            _     => T(5e-3)
        end

        for (i, b) in enumerate(batches)
            x = reshape(X[:,:,b], 784, bs)
            y = Y[:,b]
            g = gradient(p -> begin
                ŷ = F(x, p)
                l = L(ŷ, y)
                avg_loss += l
                return l
            end, θ)[1]
            
            θ = θ - η*g
        end
        avg_loss /= length(batches)
        println("loss = $(avg_loss)")

        if e % 20 == 0
            open(f -> serialize(f, F), "model_$(e).jls", "w")
            open(f -> serialize(f, θ), "params_$(e).jls", "w")
        end
    end
end

train()