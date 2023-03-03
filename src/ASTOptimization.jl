export merkle_hash, transforms, BFS, Optimizer, step_optimizer

"""
Computes the Merkle hash of a given subtree for use with AST optimization.
"""
function merkle_hash(A::ParOperator{D,R,L,P,External}) where {D,R,L,P}
    # Name of A without types
    op_str = "$(Base.typename(typeof(A)).wrapper)"

    # Domain and range types
    op_str *= "_DDT=$(D)_RDT=$(R)"

    # Domain and range values
    op_str *= "_Domain=$(Domain(A))_Range=$(Range(A))"

    return hash(op_str)
end

function merkle_hash(A::ParOperator{D,R,L,P,Internal}) where {D,R,L,P}
    # Combine hashes of children
    hash_str = foldl(*, map(c -> "$(merkle_hash(c))", children(A)))
    return hash(hash_str)
end

transforms(A::ParOperator) = [A]

mutable struct Optimizer
    # params
    orig::ParOperator
    population::Int
    bfsrounds::Int
    bfsthreshold::Int
    orig_score::Float64

    # state
    best_score::Float64
    best::Vector{Tuple{Float64, UInt, ParOperator}} # up to $population tuples of (score, hash, operator), ordered from best to worst
    seen::Set{UInt} # hash
    rounds_since_last_improvement::Int

    # stats
    num_dups::Int
    num_total_seen::Int
    num_rounds::Int
    num_bfs::Int
end

function Optimizer(orig::ParOperator, populationsize::Int=10, bfsrounds::Int=5, bfsthreshold::Int=5)
    # Keep up to $populationsize ASTs alive at once.
    # Keep track of which ASTs we've seen before.
    # For each "round":
    #     Look at transformations of the current population
    #     Hash them first and discard any we've seen before
    #     If any are equal or better than the current best, add them to the population
    #     When the $population limit would be exceeded, discard the worst scores first.
    # When no improvement has been found within $bfsthreshold rounds, try a BFS of the current best.
    #     The BFS will do $bfsrounds iterations of a breadth first search.
    # If the BFS doesn't find anything better, we're done.

    orig_score = complexity(orig)
    orig_hash = merkle_hash(orig)
    Optimizer(
        orig,                              # orig
        populationsize,                    # population
        bfsrounds,                         # bfsrounds
        bfsthreshold,                      # bfsthreshold
        orig_score,                        # orig_score

        orig_score,                        # best_score
        ([(orig_score, orig_hash, orig)]), # best
        Set{UInt}(orig_hash),              # seen
        0,                                 # rounds_since_last_improvement

        0,                                 # num_dups
        1,                                 # num_total_seen
        0,                                 # num_rounds
        0,                                 # num_bfs
    )
end

function run_optimizer(opt::Optimizer)
    # keep running until it returns false.
    while step_optimizer(opt::Optimizer)
    end
end

# returns true while the optimizer is still making progress; returns false when it runs out of things to try.
function step_optimizer(opt::Optimizer)
    opt.num_rounds += 1
    if opt.rounds_since_last_improvement < opt.bfsthreshold
        opt.rounds_since_last_improvement += 1
        best_N(opt)
        return true
    else
        # direct movement has ceased to find any improvements.
        # try a BFS.
        println("unable to make any further improvements through direct transforms; trying BFS")
        return BFS(opt)
    end
end

function best_N(opt::Optimizer)
    # we have N things; add in any (new) neighbors they have, then keep the best N things
    new_things = Vector{Tuple{Float64, UInt, ParOperator}}()
    # gather new transforms of the current working set
    for (_, _, thing) in opt.best
        for new_thing in transforms(thing)
            opt.num_total_seen += 1
            new_hash = merkle_hash(new_thing)
            if in(new_hash, opt.seen)
                opt.num_dups += 1
                continue
            end
            new_score = complexity(new_thing)
            push!(new_things, (new_score, new_hash, new_thing))
            push!(opt.seen, new_hash)
        end
    end
    # add them to the working set
    working_set = [opt.best; new_things]
    # sort by score, discard the losers
    sort!(working_set; by=t -> t[1])
    if length(working_set) > opt.population
        working_set = working_set[1:opt.population]
    end
    opt.best = working_set
    # if the new "best" is better, reset `rounds_since_last_improvement`
    if working_set[1][1] < opt.best_score
        opt.best_score = working_set[1][1]
        opt.rounds_since_last_improvement = 0
    end
end

function BFS(opt::Optimizer)
    # we have N things; move outwards in all directions for M rounds, then keep the best N things

    wavefront = copy(opt.best)

    (bestscore, besthash, bestthing) = opt.best[1]
    initial_best = bestscore

    rounds = opt.bfsrounds

    while length(wavefront) > 0
        println("BFS: $rounds rounds remaining")
        begin_count = length(opt.seen)
        next_wavefront = Vector{Tuple{Float64, UInt, ParOperator}}([])
        for (thingscore, thinghash, thing) in wavefront
            ts = transforms(thing)
            for neighbor in ts
                neighborhash = merkle_hash(neighbor)
                neighborscore = complexity(neighbor)
                push!(next_wavefront, (neighborscore, neighborhash, neighbor))
                if !(neighborhash in opt.seen)
                    if bestscore > neighborscore
                        bestscore = neighborscore
                        besthash = neighborhash
                        bestthing = neighbor
                    end

                    push!(opt.seen, neighborhash)
                    push!(opt.best, (neighborscore, neighborhash, neighbor))
                end
            end
        end

        # print some stats as we go
        end_count = length(opt.seen)
        round_count = end_count - begin_count
        println("processed $round_count new nodes this round.")
        wavefront = next_wavefront

        # decide whether to stop
        rounds -= 1
        if rounds == 0
            break
        end
    end

    sort!(opt.best; by=t -> t[1])
    if length(opt.best) > opt.population
        opt.best = opt.best[1:opt.population]
    end

    if bestscore < initial_best
        # bfs found its way out of a local minimum
        return true
    end
    # bfs found nothing better
    return false
end
