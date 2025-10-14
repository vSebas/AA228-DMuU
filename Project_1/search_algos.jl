include("score_functions.jl")

struct K2Ordering
    ordering::Vector{Int} # variable ordering
end

# struct LDGS
#     # G::SimpleDiGraph  # initial directed graph
#     k_max::Int       # maximum number of iterations
# end

"""
    K2_search(method::K2Ordering, vars, D)
Learn a Bayesian network structure using the K2 search algorithm given a variable ordering.
    method: K2Ordering instance with variable ordering
    vars: Vector of Variable structs
    D: Data matrix (rows: variables, columns: observations)
    returns: Tuple of (Bayesian score, learned directed graph)
"""
function K2_search(method::K2Ordering, vars, D)
    n = length(vars)
    G = SimpleDiGraph(n) # initialize directed graph, no edges
    y = 0.0

    for (k,i) in enumerate(method.ordering[2:end]) # skip the first element
        y = bayesian_score(vars, G, D)
        
        while true
            y_best, j_best = -Inf, 0
            for j in method.ordering[1:k] # only consider predecessors in the ordering
                if !has_edge(G, j, i) # only consider adding edge if it doesn't already exist
                    add_edge!(G, j, i)
                    y_new = bayesian_score(vars, G, D)
                    if y_new > y_best
                        y_best, j_best = y_new, j
                    end
                    rem_edge!(G, j, i) # remove the edge after evaluation
                end
            end

            if y_best > y
                add_edge!(G, j_best, i)
                y = y_best
            else
                break # no improvement found, exit the loop
            end
        end
        # method.ordering[k] = i # ensure ordering is a permutation of 1:n
    end

    return (y,G)
end

function rand_graph_neighbor(G)
    n = nv(G)   # number of nodes
    i = rand(1:n) # select a random node
    j = mod1(i + rand(2:n)-1, n) # select random node starting from i+1 to n, wrapping around using mod1 to avoid self-loop
    G_new = deepcopy(G)

    has_edge(G, i, j) ? rem_edge!(G_new, i, j) : add_edge!(G_new, i, j) # if edge exists, remove it; otherwise, add it

    return G_new
end

function local_directed_graph_search(k_max, vars, D)
    G = SimpleDiGraph(length(vars)) # initialize directed graph, no edges
    y = bayesian_score(vars, G, D)

    for k in 1:k_max
        G_new = rand_graph_neighbor(G)
        y_new = is_cyclic(G_new) ? -Inf : bayesian_score(vars, G_new, D)

        if y_new > y
            G, y = G_new, y_new
        end
    end

    return (y, G)
end

function are_markov_equivalent(G, H)
    if nv(G) != nv(H) || ne(G) != ne(H) || # if number of nodes or edges differ
        !all(has_edge(H, e) || has_edge(H, reverse(e)) for e in edges(G)) # if they don't have the same skeleton
        return false
    end

    for (I, J) in [(G, H), (H, G)]
        for c in 1:nv(I)
            parents = inneighbors(I, c)
            for (a,b) in subsets(parents, 2)
                if !has_edge(I, a, b) && !has_edge(I, b, a) # if a and b are not connected
                    if has_edge(J, a, c) && has_edge(J, b, c) # if both a and b are parents of c in J
                        return false
                    end
                end
            end
        end
    end

    return true
end
