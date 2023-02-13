using Flux
using MAT
using Plots
using Printf
using ProgressBars
using Random
using LinearAlgebra

include("DeepONet.jl")

# Load data
U = matread("/home/tgrady6/data/NavierStokes_V1e-5_N1200_T20.mat")["u"];
U = permutedims(U, (2, 3, 4, 1));
@show size(U);

# Split data
n_train = 1000;
n_test  = 200;
U_train = view(U, 1:2:64, 1:2:64, :, 1:n_train);
U_test  = view(U, 1:2:64, 1:2:64, :, 1:n_test);

# Setup fixed sensor locations
n_sensor_locations = 256;
sensor_locations = mapreduce(_ -> vcat([rand(1:s) for s in size(U_train)[1:end-2]], [1]), hcat, 1:n_sensor_locations)

Random.seed!(42)

# Setup DeepONet model
model = DeepONet(
    Chain(
        Dense(n_sensor_locations   => n_sensor_locations*4, gelu),
        Dense(n_sensor_locations*4 => n_sensor_locations*4, gelu),
        Dense(n_sensor_locations*4 => n_sensor_locations*4, gelu),
        Dense(n_sensor_locations*4 => n_sensor_locations,   gelu)
    ),
    Chain(
        Dense(3 => n_sensor_locations*2, gelu),
        Dense(n_sensor_locations*2 => n_sensor_locations*2, gelu),
        Dense(n_sensor_locations*2 => n_sensor_locations*2, gelu),
        Dense(n_sensor_locations*2 => n_sensor_locations)
    ),
    Bias(eltype(U), 1)
) |> gpu

# Setup optimiser
optim = Flux.setup(Flux.AdamW(1e-3, (0.9, 0.999), 1e-4), model) |> gpu;

# Number of output sample locations
n_output_locations = 1024;

# Main loop
n_epochs = 100;
batch_size = 20;

for e in 1:n_epochs

    starts = collect(1:batch_size:n_train)
    schedule = [s:s+batch_size-1 for s in starts]
    shuffle!(schedule)

    iter = ProgressBar(enumerate(schedule))
    for (i, r) in iter

        u = U_train[:,:,:,r]
        output_locations = mapreduce(_ -> [rand(1:s) for s in size(U_train)[1:end-1]], hcat, 1:n_output_locations*batch_size);
        output_locations = reshape(output_locations, 3, n_output_locations, batch_size)

        v_true   = sample(u, output_locations) |> gpu
        v_sample = sample(u, sensor_locations)
        
        (loss, grads) = Flux.withgradient(model) do m
            v_pred = m((v_sample |> gpu, eltype(U).(output_locations./maximum(output_locations)) |> gpu))
            return norm(v_pred .- v_true)
        end

        Flux.update!(optim, model, grads[1])

        set_description(iter, string(@sprintf("Epoch: %04d, Batch: %04d, Loss: %1.4f", e, i, loss)))
    end

    starts = collect(1:batch_size:n_test)
    schedule = [s:s+batch_size-1 for s in starts]
    shuffle!(schedule)

    iter = ProgressBar(enumerate(schedule))
    n_batches = length(iter)
    avg_test_loss = 0
    for (i, r) in iter
        u = U_test[:,:,:,r]
        output_locations = mapreduce(_ -> [rand(1:s) for s in size(U_test)[1:end-1]], hcat, 1:n_output_locations*batch_size);
        output_locations = reshape(output_locations, 3, n_output_locations, batch_size)

        v_true   = sample(u, output_locations) |> gpu
        v_sample = sample(u, sensor_locations)
        
        loss = norm(model((v_sample |> gpu, eltype(U).(output_locations./maximum(output_locations)) |> gpu)) .- v_true)
        avg_test_loss += loss/n_batches
    end

    @info e avg_test_loss

end

u = U_test[:,:,:,1:1];
output_locations = collect(Iterators.flatten(Iterators.product(1:32, 1:32, 1:20)));
output_locations = reshape(output_locations, 3, 32*32*20, 1);
v_true   = sample(u, output_locations);
v_sample = sample(u, sensor_locations);
v_pred = model((v_sample |> gpu, (eltype(U).(output_locations./maximum(output_locations))) |> gpu)) |> cpu;
v_pred = reshape(v_pred, 32, 32, 20);
v_true = reshape(v_true, 32, 32, 20);

anim = @animate for i in 1:20
    p1 = heatmap(transpose(v_true[:,:,i]), clim=(-2, 2))
    p2 = heatmap(transpose(v_pred[:,:,i]), clim=(-2, 2))
    plot(p1, p2, layout=(1,2), size=(1000, 400))
end

gif(anim, "test.gif"; fps=5);