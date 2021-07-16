using DataFrames
using GLMakie
using Statistics:mean

using NBeats
using Flux
using Flux: train!, @functor

step_size = 0.01
X = -5:step_size:5;

function target_gen(X)
    return sin.(10 .* X) .* 0.2 .+ sin.(5 .* X) .* 0.1
end
Y = target_gen(X)

scatter(X, Y)

config = Dict(
    :back_size => 20, 
    :forw_size => 5, 
    :num_hidden => 64, 
    :num_layers_core => 2,
    :num_blocks => 4,
    :batch_size => 32)

function build_train_data(TS, back_size, forw_size)
    window = back_size + forw_size
    num_batches = length(TS) - window + 1
    
    X = zeros(Float32, back_size, num_batches)
    Y = zeros(Float32, forw_size, num_batches)
    for i in 1:num_batches
        X[:, i] .= TS[i:i + back_size - 1]
        Y[:, i] .= TS[i + back_size:i + window - 1]
    end
    return X, Y
end

function build_infer_data(TS, back_size, forw_size)
    window = back_size + forw_size
    num_batches = length(TS) - window + 1
    
    X = zeros(back_size, num_batches)
    for i in 1:num_batches
        X[:, 1] .= TS[i:i + back_size - 1]
    end
    return X
end

X_train, Y_train = build_train_data(Y, config[:back_size], config[:forw_size])
dtrain = Flux.Data.DataLoader((X_train, Y_train), batchsize=config[:batch_size], shuffle=true)

m = NBeatsStack(back_size=config[:back_size], forw_size=config[:forw_size], num_hidden=config[:num_hidden], num_layers_core=config[:num_layers_core], num_blocks=config[:num_blocks])
loss(x,y) = mean((y .- m(x)).^2)

ps = params(m)
opt = ADAM(1e-3)

Flux.@epochs 1 Flux.train!(loss, ps, dtrain, opt)

scatter(X, Y)
for i in 0:9
    infer_pt = 100 * i + 1
    X_infer = Y[infer_pt:infer_pt + config[:back_size] - 1]
    Y_infer = Y[infer_pt + config[:back_size]:infer_pt + config[:back_size] + config[:forw_size] - 1]
    p_infer = m(X_infer)
    infer_x = X[infer_pt + config[:back_size]:infer_pt + config[:back_size] + config[:forw_size] - 1]
    loss(X_infer, Y_infer)
    scatter!(infer_x, m(X_infer), color=:red)
end

# save("sinus-fit.png", GLMakie.current_figure())