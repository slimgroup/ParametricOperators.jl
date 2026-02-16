export DimensionTree, dimtree
export root, is_leaf, is_root, children, left_child, right_child, parent_node
export leaves, internal_nodes, postorder, preorder, level_order
export node2ind, ind2node, num_nodes, num_leaves, num_internal
export dims_at_node, leaf_dim, leaf_index

"""
    DimensionTree

Binary dimension tree over tensor modes {1, ..., d}.

Each internal node represents a subset of dimensions, split into two disjoint
children subsets. Leaf nodes are singleton dimension sets.

The tree is stored as a flat array of node descriptors:
- `nodes[i]` contains the set of dimensions at node i
- `parent[i]` is the parent index of node i (0 for root)
- `left[i]`, `right[i]` are children indices (0 for leaves)
- Nodes are numbered in level-order (BFS), root = 1
"""
struct DimensionTree
    dims::Vector{Vector{Int}}     # dims[i] = set of modes at node i
    parent::Vector{Int}           # parent[i] = parent node index (0 for root)
    left::Vector{Int}             # left[i] = left child index (0 for leaves)
    right::Vector{Int}            # right[i] = right child index (0 for leaves)
    leaf_indices::Vector{Int}     # indices of leaf nodes
    internal_indices::Vector{Int} # indices of internal (non-leaf) nodes
    d::Int                        # number of dimensions (leaves)
end

const dimtree = DimensionTree

"""
    DimensionTree(d::Int)

Construct a balanced binary dimension tree for `d` modes.
Modes are split as evenly as possible at each level.
"""
function DimensionTree(d::Int)
    @assert d >= 2 "Need at least 2 dimensions"

    # Build tree top-down using BFS
    dims_list = Vector{Vector{Int}}()
    parent_list = Vector{Int}()
    left_list = Vector{Int}()
    right_list = Vector{Int}()

    # Start with root containing all dimensions
    push!(dims_list, collect(1:d))
    push!(parent_list, 0)
    push!(left_list, 0)
    push!(right_list, 0)

    queue = [1]  # BFS queue of node indices to process

    while !isempty(queue)
        node_idx = popfirst!(queue)
        node_dims = dims_list[node_idx]

        if length(node_dims) == 1
            # Leaf node, nothing to split
            continue
        end

        # Split dimensions
        mid = length(node_dims) ÷ 2
        left_dims = node_dims[1:mid]
        right_dims = node_dims[mid+1:end]

        # Create left child
        left_idx = length(dims_list) + 1
        push!(dims_list, left_dims)
        push!(parent_list, node_idx)
        push!(left_list, 0)
        push!(right_list, 0)
        left_list[node_idx] = left_idx
        push!(queue, left_idx)

        # Create right child
        right_idx = length(dims_list) + 1
        push!(dims_list, right_dims)
        push!(parent_list, node_idx)
        push!(left_list, 0)
        push!(right_list, 0)
        right_list[node_idx] = right_idx
        push!(queue, right_idx)
    end

    # Identify leaves and internal nodes
    leaf_idx = Int[]
    internal_idx = Int[]
    for i in 1:length(dims_list)
        if left_list[i] == 0 && right_list[i] == 0
            push!(leaf_idx, i)
        else
            push!(internal_idx, i)
        end
    end

    DimensionTree(dims_list, parent_list, left_list, right_list, leaf_idx, internal_idx, d)
end

"""
    DimensionTree(splits::Vector{Vector{Vector{Int}}})

Construct a dimension tree from explicit splits. Each element of `splits`
is `[left_dims, right_dims]` for an internal node, processed in BFS order.
"""
function DimensionTree(all_dims::Vector{Int}, splits::Vector{Tuple{Vector{Int}, Vector{Int}}})
    dims_list = Vector{Vector{Int}}()
    parent_list = Vector{Int}()
    left_list = Vector{Int}()
    right_list = Vector{Int}()

    push!(dims_list, all_dims)
    push!(parent_list, 0)
    push!(left_list, 0)
    push!(right_list, 0)

    split_idx = 1
    queue = [1]

    while !isempty(queue) && split_idx <= length(splits)
        node_idx = popfirst!(queue)
        node_dims = dims_list[node_idx]

        if length(node_dims) == 1
            continue
        end

        left_dims, right_dims = splits[split_idx]
        split_idx += 1

        @assert sort(vcat(left_dims, right_dims)) == sort(node_dims) "Split must partition node dimensions"

        left_idx = length(dims_list) + 1
        push!(dims_list, left_dims)
        push!(parent_list, node_idx)
        push!(left_list, 0)
        push!(right_list, 0)
        left_list[node_idx] = left_idx
        push!(queue, left_idx)

        right_idx = length(dims_list) + 1
        push!(dims_list, right_dims)
        push!(parent_list, node_idx)
        push!(left_list, 0)
        push!(right_list, 0)
        right_list[node_idx] = right_idx
        push!(queue, right_idx)
    end

    leaf_idx = Int[]
    internal_idx = Int[]
    for i in 1:length(dims_list)
        if left_list[i] == 0 && right_list[i] == 0
            push!(leaf_idx, i)
        else
            push!(internal_idx, i)
        end
    end

    d = length(all_dims)
    DimensionTree(dims_list, parent_list, left_list, right_list, leaf_idx, internal_idx, d)
end

# --- Accessors ---

root(T::DimensionTree) = 1
num_nodes(T::DimensionTree) = length(T.dims)
num_leaves(T::DimensionTree) = length(T.leaf_indices)
num_internal(T::DimensionTree) = length(T.internal_indices)

is_leaf(T::DimensionTree, i::Int) = T.left[i] == 0 && T.right[i] == 0
is_root(T::DimensionTree, i::Int) = T.parent[i] == 0

left_child(T::DimensionTree, i::Int) = T.left[i]
right_child(T::DimensionTree, i::Int) = T.right[i]
parent_node(T::DimensionTree, i::Int) = T.parent[i]
children(T::DimensionTree, i::Int) = (T.left[i], T.right[i])

dims_at_node(T::DimensionTree, i::Int) = T.dims[i]

leaves(T::DimensionTree) = T.leaf_indices
internal_nodes(T::DimensionTree) = T.internal_indices

"""
    node2ind(T::DimensionTree, node::Int)

Map a node index to its position in the internal nodes ordering.
For transfer tensors, which are indexed by internal node order.
"""
function node2ind(T::DimensionTree, node::Int)
    idx = findfirst(==(node), T.internal_indices)
    isnothing(idx) && error("Node $node is not an internal node")
    return idx
end

"""
    ind2node(T::DimensionTree, ind::Int)

Inverse of node2ind: map an internal node position back to node index.
"""
ind2node(T::DimensionTree, ind::Int) = T.internal_indices[ind]

"""
    leaf_index(T::DimensionTree, node::Int)

Map a leaf node index to its position in the leaf ordering.
"""
function leaf_index(T::DimensionTree, node::Int)
    idx = findfirst(==(node), T.leaf_indices)
    isnothing(idx) && error("Node $node is not a leaf node")
    return idx
end

"""
    leaf_dim(T::DimensionTree, node::Int)

Return the single dimension index at a leaf node.
"""
function leaf_dim(T::DimensionTree, node::Int)
    @assert is_leaf(T, node) "Node $node is not a leaf"
    return T.dims[node][1]
end

"""
    postorder(T::DimensionTree)

Return node indices in post-order (children before parents).
"""
function postorder(T::DimensionTree)
    result = Int[]
    _postorder!(T, root(T), result)
    return result
end

function _postorder!(T::DimensionTree, node::Int, result::Vector{Int})
    if !is_leaf(T, node)
        _postorder!(T, left_child(T, node), result)
        _postorder!(T, right_child(T, node), result)
    end
    push!(result, node)
end

"""
    preorder(T::DimensionTree)

Return node indices in pre-order (parents before children).
"""
function preorder(T::DimensionTree)
    result = Int[]
    _preorder!(T, root(T), result)
    return result
end

function _preorder!(T::DimensionTree, node::Int, result::Vector{Int})
    push!(result, node)
    if !is_leaf(T, node)
        _preorder!(T, left_child(T, node), result)
        _preorder!(T, right_child(T, node), result)
    end
end

"""
    level_order(T::DimensionTree)

Return node indices in level-order (BFS from root).
"""
function level_order(T::DimensionTree)
    result = Int[]
    queue = [root(T)]
    while !isempty(queue)
        node = popfirst!(queue)
        push!(result, node)
        if !is_leaf(T, node)
            push!(queue, left_child(T, node))
            push!(queue, right_child(T, node))
        end
    end
    return result
end

function Base.show(io::IO, T::DimensionTree)
    print(io, "DimensionTree(d=$(T.d), nodes=$(num_nodes(T)), leaves=$(num_leaves(T)))")
end
