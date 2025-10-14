include("score_functions.jl")

struct K2Search
    ordering::Vector{Int} # variable ordering
end

"""
    fit(method::K2Search, vars, D)
Learn a Bayesian network structure using the K2 search algorithm given a variable ordering.
    method: K2Search instance with variable ordering
    vars: Vector of Variable structs
    D: Data matrix (rows: variables, columns: observations)
    returns: Tuple of (Bayesian score, learned directed graph)
"""
function fit(method::K2Search, vars, D)
    n = length(vars)
    G = SimpleDiGraph(n) # initialize empty graph
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