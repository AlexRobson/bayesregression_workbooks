using Pkg
Pkg.activate("./")

using Distributions
using LinearAlgebra
using Plots
using IterTools

# Create some data

function generate_data(N; w = randn(2,))

    
    x = 2 * rand(N,) .- 1
    Φ = hcat(ones(N,), x)
    y = w[1] .+ w[2] * x + 0.2 * randn(N,);
    
    return (x=x, y=y, w=w)

end

function prior(; α)
    m₀ = zeros(2);
    S₀ = (1/α) * I(2);
    prior = MvNormal(m₀, S₀)
    return prior
end

# function sample(D::Sampleable)
#     a_0, a_1 = rand(prior)
# end

function posterior(Φ, t; α, β)
    D = 2
    m₀ = zeros(D);
    S₀ = (1/α) * I(D);
    S₀inv = α * I(D) 
    
    Sₙinv = S₀inv + β * transpose(Φ) * Φ
    #Sₙ = cholesky(Sₙinv) \ I
    Sₙ = inv(Sₙinv)
    mₙ(t) = Sₙ * β * transpose(Φ) * t
    posterior = MvNormal(mₙ(t), Symmetric(Sₙ))
    return posterior
end

function loglikelihood(w, Φ, t; β)
    N = size(Φ, 2)
    E_d = 0.5 * sum([(_t - transpose(w) * _x)^2 for (_t, _x) in zip(t, eachrow(Φ))])
    ll = (N / 2) * (log(β) - log(2*π)) - β * E_d
    return ll
end

function Φ(x)
    N = size(x, 1)
    hcat(ones(N, ), x)
end

function create_empty_plot()

    empty_plot = plot(
        grid = false,
        axis = nothing,
        legend = false,
        framestyle = :none,
        showaxis = false,
        ticks = nothing,
        border = :none
    )
    
end



function plot_data_space(d::Sampleable, data=[])
    x_grid = -1:0.01:1
    p = scatter()
    for _ in 1:5
        let (a0, a1) = rand(d)
            plot!(p, x_grid, a0 .+ x_grid .* a1, color = :red, legend = :none)
        end
    end
    if !isempty(data)
        x, y = data
        scatter!(x, y)
    end
    return p
end

function plot_distribution(W::AbstractVector, d::AbstractMatrix; w = [])
    p = heatmap(W, W, d, colorbar = :none)
    if !isempty(w)
        scatter!(p, [w[1]], [w[2]])
    end
    return p
end

# Creates the plot Figure 3.7
function plot_bayesian_linear_regression(x, t, w_star; α = 1.0, β = 1.0)
       
    W = -1:0.01:1
    W_grid = Iterators.product(W, W)


    p_prior = [pdf(prior(;α = α), [w_0, w_1]) for (w_0, w_1) in W_grid];
    
    plot_rows = []
    for ndata in [1, 2, 5, size(x, 1)]
        let (_x, _y) = ((x[1:ndata]), y[1:ndata])
            p_likelihood = exp.([loglikelihood([w_0, w_1], Φ(_x), _y; β = β) for (w_0, w_1) in W_grid]);
            p_posterior = [pdf(posterior(Φ(_x), _y; α = α, β = β), [w_0, w_1]) for (w_0, w_1) in W_grid];
            row = [
                plot_distribution(W, p_likelihood'; w = w_star),
                plot_distribution(W, p_posterior'; w = w_star),
                plot_data_space(posterior(Φ(_x), _y; α = α, β = β), (_x, _y)),
            ]
            push!(plot_rows, row)
        end
    end
    
    first_row = [create_empty_plot(); heatmap(W, W, p_prior', colorbar = :none); plot_data_space(prior(;α = α))]
    return hcat(first_row, hcat(plot_rows...))
end


(x,y, w_star) = generate_data(20; w = [-0.3, 0.5])
p = plot_bayesian_linear_regression(x, y, w_star; α = 2, β = (1/0.2)^2)

plot(p..., layout = (5, 3), size = (1200, 600))


