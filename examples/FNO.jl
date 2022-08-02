using LinearAlgebra
using Tao

# Channels and (x, y, t) dimensions
nc = 20
nx = 16
ny = 32
nt = 10

# (x, y, t) modes
mx = 4
my = 4
mt = 4

# Fourier transform in each dimension
Fx = DFTOperator{ComplexF64}([nx])
Fy = DFTOperator{ComplexF64}([ny])
Ft = DFTOperator{ComplexF64}([nt])

# Restriction operators to select modes in each dimension
Rx = RestrictionOperator{ComplexF64}(Range(Fx), [1:mx, nx-mx+1:nx])
Ry = RestrictionOperator{ComplexF64}(Range(Fy), [1:my, ny-my+1:ny])
Rt = RestrictionOperator{ComplexF64}(Range(Ft), [1:mt])

# Fourier transform (with restriction) along (x, y, t)
Fs = (Rt*Ft) ⊗ (Ry*Fy) ⊗ (Rx*Fx)

# Identity along channels, (x, y, t), and restricted (x, y, t)
Ic = IdentityOperator{ComplexF64}(nc)
Is = IdentityOperator{ComplexF64}(nx*ny*nt)
Ir = IdentityOperator{ComplexF64}(Range(Fs))

# Full Fourier transform with channels excluded
F = Fs ⊗ Ic

# Dense weights along channel dimension
Wc1 = MatrixOperator{ComplexF64}(nc, nc)
Wc2 = MatrixOperator{ComplexF64}(nc, nc)

# Diagonal operator (elementwise mul) along (c, x, y, t)
Ds = DiagonalOperator{ComplexF64}(Range(F))

# Full elementwise operator with weighted channel mixing
D = (Ir ⊗ Wc1) * Ds

# Spectral convolution
S = F'*D*F

# Passthrough weighting
W = Is ⊗ Wc2

# Full FNO block (w/o nonlinearity)
B = S+W

# Create a random vector in the Domain of B
x = rand(ddt(B), Domain(B))

# Initialize parameters
θ = ParameterVector()
init(B, θ)

# Display number of parameters
@show count_params(B(θ))

# Fake data
x = rand(ddt(B), Domain(B))
y = rand(rdt(B), Range(B))

@show size(x) size(B(θ)*x)
