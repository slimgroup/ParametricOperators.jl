using Tao

n_embed = 128
n_context = 64
n_hidden = n_embed * 2

Ic = Tao.IdentityOperator{Float16}(n_context)
WQ = Ic ⊗ Tao.MatrixOperator{Float16}(n_embed, n_embed)
WK = Ic ⊗ Tao.MatrixOperator{Float16}(n_embed, n_embed)
WV = Ic ⊗ Tao.MatrixOperator{Float16}(n_embed, n_embed)
WL = Ic ⊗ Tao.MatrixOperator{Float16}(n_hidden, n_embed)
WP = Ic ⊗ Tao.MatrixOperator{Float16}(n_embed, n_hidden)

function attn(x; θ)
    q = WQ(θ)*x
    k = WK(θ)*x
    v = WV(θ)*x
    score = reshape(q, n_embed, n_context)*reshape(k, n_embed, n_context)'
    A = mapslices(r -> exp.(r)/sum(exp.(r)), score, dims = [1])
    return vec(A*reshape(v, n_embed, n_context))
end

mlp(x; θ) = WP(θ)*tanh.(WL(θ)*x)

θ = Tao.ParameterVector()
init(WQ, θ)
init(WK, θ)
init(WV, θ)
init(WL, θ)
init(WP, θ)

x = rand(ddt(WQ), Domain(WQ))
y = mlp(attn(x; θ); θ)