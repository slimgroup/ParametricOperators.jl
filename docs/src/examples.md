## Serial Examples

### Parametric Linear Operator:

```
using ParametricOperators

A = ParMatrix(Float64, 4, 4)

x = rand(4)
θ = init(A)

y = A(θ) * x
x = A'(θ) * y
```

## Distributed Examples

### Parametric Linear Operator

```
using ParametricOperators
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
root = 0

x = nothing
if rank == root
    x = rand(4)
end
x = MPI.bcast(x, root, comm)

A = ParMatrix(Float64, 4, 4)
A = distribute(A, [1, 2])

θ = init(A)
y = A(θ) * x

MPI.Finalize()
```

#### Running on 2 workers

```
mpiexecjl --project=./path/to/your/project -n 2 julia test_file.jl
```