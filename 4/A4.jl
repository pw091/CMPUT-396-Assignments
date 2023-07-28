### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ fec77c57-c21a-42c4-8e39-3c21c804504c
begin
	using StatsPlots, PlutoUI, Random, Statistics
	using Flux
	using MLDatasets
	import MLDataUtils: MLDataUtils, shuffleobs, eachbatch, getobs
end

# ╔═╡ 5b40470e-6195-4307-8c78-993da5c79699
begin
	_check_complete(complete) = complete ? "✅" : "❌"
	
	md"""
	# Setup

	This section loads and installs all the packages. You should be setup already from assignment 1, but if not please read and follow the `instructions.md` for further details.
	"""
end

# ╔═╡ 6d0284b7-1b4e-4241-a796-034c53b06fef
gr() # In this notebook we use the plotly backend for Plots.

# ╔═╡ 72fef447-cdf6-443a-ae6b-53c217602d98
md"""
### !!!IMPORTANT!!!

Insert your details below. You should see a green checkmark.
"""

# ╔═╡ 54ff2a52-13c5-4e55-ad57-7acf5474cae4
student = (name="Patrick Wyrod", email="pwyrod@ualberta.ca", ccid="pwyrod", idnumber=1584645)

# ╔═╡ 5897039a-9571-4648-b1f6-a74efb6aef3b
let
	def_student = (name="NAME as in eclass", email="UofA Email", ccid="CCID", idnumber=0)
	if length(keys(def_student) ∩ keys(student)) != length(keys(def_student))
		md"You don't have all the right entries! Make sure you have `name`, `email`, `ccid`, `idnumber`. ❌"
	elseif any(getfield(def_student, k) == getfield(student, k) for k in keys(def_student))
		md"You haven't filled in all your details! ❌"
	elseif !all(typeof(getfield(def_student, k)) === typeof(getfield(student, k)) for k in keys(def_student))
		md"Your types seem to be off: `name::String`, `email::String`, `ccid::String`, `idnumber::Int`"
	else
		md"Welcome $(student.name)! ✅"
	end
end

# ╔═╡ c69ce5c8-cd42-4ae2-aaa9-74d0daa5875f
md"""
Important Note: You should only write code in the cells that have: """

# ╔═╡ dfb12ef6-2968-40bc-a7eb-5284dec0c359
#### BEGIN SOLUTION

#### END SOLUTION

# ╔═╡ 5b6b85d6-25f5-4ac9-9fe3-5e85c432f0f0
md"""
# Q1: How do ANNs represent and transform data?

While neural networks are large complex machines, this doesn't mean we can't work to understand the algorithms they encode/learn. In this section, we are going to go through an example based on [this blog](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/). Chris Olah is at the forefront of this work, and has several follow up works on understanding the representations and algorithms encoded by trained networks: 

- [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/),
- [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/), 
- and [Exploring Neural Networks with Activation Atlases](https://distill.pub/2019/activation-atlas/).

You should take a look at these in your own time. We are going to go through buidling the example presented in the first linked blog, and extend these ideas slightly.
"""

# ╔═╡ 6b7db8c9-6a3a-49ae-926a-35a8f8bfb8e8
md"""
### Generate Data

First we need to generate data. We use two sinusoidal functions to generate data for two classes that can't be separated via a linear decision plane. The problem is to classify between the first and second wave, and we will use this data to train a neural network which maximizes the log-liklihood of classifying class 1.

Below is a graphical visualization of the data we are going to use throughout the section.

"""

# ╔═╡ 31543a74-a116-4861-b153-30e6bcaecce8
function class_1(x)
	1//2 * sin( (x-1//2) * π) + 2//5
end

# ╔═╡ b102d8ee-1191-4580-aa53-09740cf608a4
function class_2(x)
	1//2 * sin( (x-1//2) * π) - 2//5
end

# ╔═╡ f9596fd6-b1fa-43d4-b84c-9a1151db372d
function dataset_sin(n)
	X_c1 = reduce(hcat, 
		begin
			x=2*rand(Float32)-1
			y=class_1(x)
			[x, y]
		end for i in 1:n)
	
	
	X_c2 = reduce(hcat, 
		begin
			x=2*rand(Float32)-1
			y=class_2(x)
			[x, y]
		end for i in 1:n)
	
	Y = fill(false, 2*n)
	Y[1:n] .= true
	
	X = hcat(X_c1, X_c2)
	(X, reshape(Y, 1, :))
	
end

# ╔═╡ b37d3b82-fcc2-4ac9-94cc-86174481ba86
let
	plot(-1:0.01:1, class_1)
	plot!(-1:0.01:1, class_2, legend=nothing)
	X, Y = dataset_sin(30)
	scatter!(X[1, :], X[2, :], markercolor=reshape([y==1 ? :blue : :red  for y in Y'], :))
end

# ╔═╡ 226149d2-a4ce-4713-8ff4-420c82206b07
function train_classifier!(model; N_2=1024, batch_size=32, epochs=200, η=1e-2)
	
	D = dataset_sin(N_2)
	ps = Flux.params(model)
	opt = RMSProp(η)
	
	#= 
	Use Flux.Losses.binarycrossentropy to define the 
	loss function for the classifier.
	=#
	loss(x, y) = Flux.Losses.binarycrossentropy(model(x), y)

	#= 
	The training procedure using `Flux.train!`.
	=#
	for epoch in 1:epochs
		Ds = shuffleobs(D)
		Flux.train!(loss, ps, eachbatch(Ds, size=batch_size), opt)
	end

	model
end

# ╔═╡ 0371d19e-9684-4b81-b1cb-91409fcab2d7
begin

	__train_simple_network = let
		Random.seed!(10)
		act = σ
		_layer_cnt = 2
		_hu_cnt = 2
		model = if _layer_cnt == 0
			Flux.Chain(Dense(2, 1, σ))
		else
			Flux.Chain([Dense(i == 1 ? 2 : _hu_cnt, _hu_cnt, act) for i in 1:_layer_cnt]..., 
					   Dense(_hu_cnt, 1, σ))
		end
		train_classifier!(model, epochs=200)

		t1 = all(model[1].W .≈ Float32[12.697066 12.893544; -10.022784 11.295809])
		t2 = all(model[2].W .≈ Float32[-7.322396 -7.376651; -7.669924 -7.7287607])
		t1 && t2
	end
	
md"""
### Train a Simple Neural Network $(_check_complete(__train_simple_network))

In this example, we are going to take advantage of [Flux.jl](https://github.com/FluxML/Flux.jl), which is a neural network package written entirely in Julia. If you are interested in how to write a complex julia package, this is a great example package to learn and read. Flux is hackable and easily extensible.

In the following, we use [`Flux.train!`](https://fluxml.ai/Flux.jl/stable/training/training/) from Flux. We also use `eachbatch` and `shuffleobs` from `MLDataUtils` to create batches and appropriately shuffle the training data on each epoch. You will need these functions for the VAE section.
"""
	
end

# ╔═╡ 49288405-b370-4cfb-a0e3-01d3b9126470
md"""
##### Checkpoint 1

Spend some time with the following visualization. You can use the below interface to modify the activation functions, number of hidden layers, and number of hidden units in each layer. The input space of our problem is $x\in[-1.0,1.0]^2$. We train a neural network using the binary cross entropy loss function to discriminate between class 1 and class 2. The plot below is the resulting decision plane the neural network learns. The heatmap is the activation of the final layer given the input as the (x,y)-coordinate. Higher activation means the network believes it is class 1 and lower activation means the network believes this point is class 2.

You will need this visualization to answer question 1b, and you will need to save plots for the written protion as well. You can save the plot with the cell directly following the heatmap, or drag and drop the plot to your desktop (converting from `svg` to `pdf` after the fact). 

- Hidden Activation Function: $(@bind _hidden_act PlutoUI.Select(["sigmoid", "relu", "identity"]))
- Number of layers: $(@bind _layer_cnt_str PlutoUI.Select(string.(0:2), default="1"))
- Number of hidden units: $(@bind _hu_cnt_str PlutoUI.Select(string.([1, 2, 4, 8, 16]), default="2"))
"""

# ╔═╡ ae2b42f4-5a0c-4e0f-b14c-d6b47004cc4f
_model = let
	Random.seed!(10)
	act = getfield(Flux, Symbol(_hidden_act))
	_layer_cnt = parse(Int, _layer_cnt_str)
	_hu_cnt = parse(Int, _hu_cnt_str)
	model = if _layer_cnt == 0
		Flux.Chain(Dense(2, 1, σ))
	else
		Flux.Chain([Dense(i == 1 ? 2 : _hu_cnt, _hu_cnt, act) for i in 1:_layer_cnt]..., 
				   Dense(_hu_cnt, 1, σ))
	end
	train_classifier!(model, epochs=200)

	model
end;

# ╔═╡ 8b85f477-674e-403e-a834-36eb53ab4c30
let
	X_x = -1.0:0.01:1.0
	X_y = -1.0:0.01:1.0
	plt = heatmap(X_x, X_y, getindex.(_model.(collect.(Iterators.product(X_x, X_y))), 1)', title="Activation: $(_hidden_act), Layers: $(_layer_cnt_str), HUs: $(_hu_cnt_str)")
	plot!(plt, -1.0:0.01:1.0, class_1, legend=nothing)
	plot!(plt, -1.0:0.01:1.0, class_2, legend=nothing)
end

# ╔═╡ df07dbe6-3983-48a8-b92a-40ccd4f08a87
# Manually uncomment and run this cell, you can also drag and drop:
# savefig("location.pdf")

# ╔═╡ 54f27f56-945a-48ae-96cf-51e15ca27cbe
md"""
### Visualizing Hidden Unit Activations

In this section we will go through and visualize the internal representation of our networks. We focus on a subset of networks we analyzed above, but will hopefully discover some interesting properties that elucidate what might be going on internally.

**Note:** A word of caution about analyzing single models. In this assignment we have moved away from looking at a distribution of performance or a distribution of models, as visualizing representations is hard to do over many runs. Looking at individual models can still be meaningful, as long as we look for edge cases and ensure the conclusions we draw generalize to several seeds. In this notebook, we do some work to find these edge cases and explore different random seeds.
"""

# ╔═╡ f8f90dbf-3b35-4ec3-8b65-1652aab3f954
md"""
First, we need to build up the machinery to capture the representation of the internal hidden units of the network. Flux comes to the rescue with the function `Flux.activations`. This returns all the activations of the network given an input. 
"""

# ╔═╡ c4df8697-e133-4e92-b271-4ba5371566fa
#=
This cell creates a new model `test_model` which we will use for some visualizations and testing below.
=#
test_model = let
	Random.seed!(10)
	act = relu
	_layer_cnt = 2
	_hu_cnt = 4
	model = if _layer_cnt == 0
		Flux.Chain(Dense(2, 1, σ))
	else
		Flux.Chain([Dense(i == 1 ? 2 : _hu_cnt, _hu_cnt, act) for i in 1:_layer_cnt]..., 
				   Dense(_hu_cnt, 1, σ))
	end
	train_classifier!(model, epochs=200)

	model
end;

# ╔═╡ 697d1728-1ec3-4a76-9b9d-de2831e39214
Flux.activations(test_model, [-1.0, 0.44])

# ╔═╡ bd1e57af-6373-4815-a305-86e03f57c64e
function get_activation_index(acts, layer, hidden_unit)
	# This should be a single line
	acts[layer][hidden_unit]
end

# ╔═╡ 3d49ff71-63cf-495e-b87c-e0f3ba7b925d
begin
	__get_act_index_check = let
		acts = [[1, 2, 3], [4, 5, 6], [7, 8]]
		c1 = get_activation_index(acts, 1, 2) == 2
		c2 = get_activation_index(acts, 2, 3) == 6
		c3 = get_activation_index(acts, 3, 1) == 7
		c1 && c2 && c3
	end
	
md"""
The `get_activation_index` function takes the return of `Flux.activations`, a layer index, and a hidden_unit index. This should return the activation for that layer and hidden unit.
"""
end

# ╔═╡ 7b6fc4a5-a630-42e7-9924-c2862bebfdd1
md"""
`plot_activations` below is the function we can use to plot the activations of a hidden unit over the input distribution. We plot a heatmap to represent this. The lines overlayed on the heatmap represent the ground truth data we use to generate the training data.
"""

# ╔═╡ 356d6228-0b23-441f-826a-ac8b1cf495e8
function plot_activations(model, layer, hidden_unit)
	X_x = -1.0:0.01:1.0
	X_y = -1.0:0.01:1.0
	
	acts = Flux.activations.((model,), 
		collect.(Iterators.product(X_x, X_y)))
	heatmap(X_x, X_y, get_activation_index.(acts, layer, hidden_unit)')
	plot!(-1.0:0.01:1.0, class_1, legend=nothing)
	plot!(-1.0:0.01:1.0, class_2, legend=nothing)
end

# ╔═╡ f64a54e5-930d-46fd-9741-6cd87696b2e7
md"""
Below is an example of visualizing each hidden unit's output given the input data $x\in[-1.0, 1.0]^2$. $x$ is passed all the way through the network and we capture the activation of the layer and hidden unit in that layer from the below interface. The lines again are the data we use to generate the training data. 

- Layer: $(@bind _lay_str PlutoUI.Select(string.([1, 2, 3]), default="1"))
- Hidden Unit: $(@bind _hu_str PlutoUI.Select(string.(1:4), default="1"))
"""

# ╔═╡ 36fd8579-69bb-4e6d-98a3-5e80e2a2f3b9
plot_activations(test_model, parse(Int, _lay_str), parse(Int, _hu_str))

# ╔═╡ 5d5019a1-42fa-4076-9de1-71c575133f50
md"""
### Data Transformations

Instead of focusing on just the input distribution, we can also look at how different parts of the network transform the input data and how that relates to the final decision plane. The goal of the network is to create a representation such that the two classes are linearly seperable. Below we show some example models, you will have to answer questions in the written portion using these examples.

Below is the function `plot_transform`. This takes a model and a layer `l::Int`. The model can be of any depth, but each layer must have no more than 2 hidden units and we assume the activations of these layers to be `sigmoid`. The plot generated by this function has two parts (much like the previous plots).

- **Heatmap**: The heatmap is the output of the final layer with respect to the output of the layer passed into the function.
- **Lines**: The lines are from the data generating functions we defined above, but now transformed by the first `l` layers.

The xaxis and yaxis of these plots represent the activations of the two hidden units in layer `l`. For the line, this means this is what the hidden units returned when the training data was passed through the front model. For the heatmap, this is the activation of the final layer given the two hidden units in layer `l` output $(x, y) \in [0.0, 1.0]^2$.

"""

# ╔═╡ b5cbed07-f2f0-4360-9714-5a7218831e55
function plot_transform(model, layer; kwargs...)
	
	i = layer
	front_model = model[1:i]
	rear_model = model[(i+1):end]
	
	X_x = 0.0:0.01:1.0
	X_y = 0.0:0.01:1.0
	
	heatmap(X_x, X_y, getindex.(rear_model.(collect.(Iterators.product(X_x, X_y))), 1)'; kwargs...)
	
	X_x = -1.0:0.01:1.0
	X_y = class_1.(X_x)
	
	plot!(getindex.(front_model.(collect.(zip(X_x, X_y))), 1), getindex.(front_model.(collect.(zip(X_x, X_y))), 2), legend=nothing, lw=3)
	
	X_x = -1.0:0.01:1.0
	X_y = class_2.(X_x)
	plot!(getindex.(front_model.(collect.(zip(X_x, X_y))), 1), getindex.(front_model.(collect.(zip(X_x, X_y))), 2), legend=nothing, lw=3)

end

# ╔═╡ a534dcbd-33f4-4920-950b-ae3f34142bd8
md"""
#### What happens when a model fails?

In the following section, we plot a successful run of the model and a failed run of the model. You will have to answer questions about these in the written portion of the assignment.
"""

# ╔═╡ c7f2fc33-4c0d-402a-a8bf-d686458d1b67
md"""
Below is a succesful run of the model.
"""

# ╔═╡ 0390b28b-0b50-424c-8f56-4c2dd01afdc5
successful_model = let
	Random.seed!(10)
	act = sigmoid
	model = Flux.Chain(
		Dense(2, 2, act),
		Dense(2, 2, act),
		Dense(2, 1, σ))
	train_classifier!(model)
	
	model
end

# ╔═╡ 46d0b27d-4cb2-4916-a226-9ab34e561cd9
plot_transform(successful_model, 2)

# ╔═╡ eb82338f-bfc3-4ab8-a30a-abf2ec333593
md"""

Below are two failed runs of the model.
"""

# ╔═╡ 299c202e-b1a9-4e90-8efc-b15fb44c8e5e
failed_2layer_model = let
	Random.seed!(11)
	act = sigmoid
	model = Flux.Chain(
		Dense(2, 2, act),
		Dense(2, 2, act),
		Dense(2, 1, σ))
	train_classifier!(model)
	
	model
end

# ╔═╡ 6b366eac-6b94-44a8-969a-1148577b251d
plot_transform(failed_2layer_model, 2)

# ╔═╡ 7bd1b70a-db39-497d-83ef-1dbd587503de
failed_1layer_model = let
	Random.seed!(15)
	act = sigmoid
	model = Flux.Chain(
		Dense(2, 2, act),
		# Dense(2, 2, act),
		Dense(2, 1, σ))
	train_classifier!(model)
	
	model
end

# ╔═╡ b4246952-2272-4e93-957d-5f68c511752c
plot_transform(failed_1layer_model, 1)

# ╔═╡ 49f77c7b-6e6d-4cf9-a0ea-b90c99891570
md"""
#### Epoch by Epoch

Below we look at the data transformations after each epoch of training for a successful run. Again, you will use these plots to answer questions in the written part of the assignment.

**Note**: On first load, your browser may not show the entire heatmap. You can sweep over all 30 images and have them all load, and they should fully load in a few seconds. You can also try resizing your window to force the browser to redraw the screen.


Epoch: $(@bind plt_idx PlutoUI.NumberField(0:30))
"""

# ╔═╡ 390fc91e-0fb9-4f32-9a3a-54c4df1954bf
rep_plots = let
	Random.seed!(10)
	model = Flux.Chain(Dense(2, 2, sigmoid), Dense(2, 2, sigmoid), Dense(2, 1, σ))
	D = dataset_sin(1024)
	ps = Flux.params(model)
	opt = RMSProp(1e-2)
	title_plt = plot(title = "Epoch-0", grid = false, showaxis = false, bottom_margin = -20Plots.px, ticks=false)
	plts = [plot!(title_plt, plot_transform(model, 1, title="Layer-1"), plot_transform(model, 2, title="Layer-2",), layout=@layout([A{0.01h}; [B C]]))]
	for e in 1:30
		Ds = shuffleobs(D)
		Flux.train!((x, y) -> Flux.Losses.binarycrossentropy(model(x), y), ps, eachbatch(Ds, size=16), opt)
		title_plt = plot(title = "Epoch-$(e)", grid = false, showaxis = false, bottom_margin = -20Plots.px, ticks=false)
		push!(plts, plot(title_plt, plot_transform(model, 1, title="Layer-1"), plot_transform(model, 2, title="Layer-2"), layout=@layout([A{0.01h}; [B C]])))
	end
	plts
end;

# ╔═╡ 09d8ef7a-b96c-4f50-a470-acbbc2d5b895
rep_plots[plt_idx+1]

# ╔═╡ 5b6519a7-00dd-4e18-a8e9-a9eaed60d60e
md"""
# Variational Autoencoders

In the following section, we will be implementing the details of a variational autoencoder. Below you will find two structs. One for the `Encoder` and another for the full `VAE`. We will be using Flux's built in untilites for the `Decoder` network.

**IMPORTANT:** Unlike in previous assignments, we are breaking with some notational assumptions we made in the notes. The matrix of instances `X` will be constructed where each column is an instance instead of each row. This means $X \in \mathbb{R}^{\text{\# Features} \times \text{\# Samples}}$.
"""

# ╔═╡ 38f9f22f-fabf-4278-9a5a-2f77eac7c2db
function encode_input(encoder, x)
	f = encoder.encode_feats(x)
	encoder.μ(f), encoder.logvar(f)
end

# ╔═╡ 8c41e1cf-8139-4e61-b92b-b7d201e91d2d
begin
	struct Encoder{F, M}
		encode_feats::F
		μ::M
		logvar::M
		function Encoder(ef_model::F, ef_out, latent) where F
			μ, logvar = Dense(ef_out, latent), Dense(ef_out, latent)
			new{F, typeof(μ)}(ef_model, μ, logvar)
		end
	end
	Flux.@functor Encoder
	function (encoder::Encoder)(x)
		encode_input(encoder, x)
	end
end

# ╔═╡ d5fa8fc7-ddd0-4739-8ed2-d3c45995e179
begin
	struct VAE
		encoder::Encoder
		decoder::Flux.Chain
	end
	Flux.@functor VAE # to make the struct work w/ Flux's autodiff
end

# ╔═╡ 32bfe0e2-21f0-4d5e-9c60-2a2ed5ca31c3
function reconstruct(vae::VAE, x)
	#### BEGIN SOLUTION
	
	μ, logvar = encode_input(vae.encoder, x)
	
	z = ((exp.(logvar)).^(1//2)) .* randn(Float32, size(μ)) .+ μ
	
	x̂ = vae.decoder(z)
	
	return μ, logvar, x̂
	
	#zeros(Float32, size(vae.encoder.μ.W, 1), size(x, 2)), zeros(Float32, size(vae.encoder.logvar.W, 1), size(x, 2)), zero(x)

	#### END SOLUTION
end

# ╔═╡ 131a7831-250a-47e0-90f7-9cb14db0a53b
md"""
### Elbo Loss

As you've read in your notes the elbo loss has two parts. You will need to implement both the `kl_divergence` and the `binary_cross_entropy` (which is used to estimate the log-likelihood of our distribution). Then we will use these helper functions to define the `elbo` loss used to train our VAE.

**Warning:** Like above, you should be careful about how you implement these losses. You want to ensure you are not doing any implicit promotions to double precision floating point numbers (i.e. `Float64`). This is important for performance when training your network. A promotion will happen whenever two numbers of different types interact, leading to the number with higher precision to be chosen. For example if you type in `1 + 1.0f0` you will get `2.0f0` which is a `Float32`.
"""

# ╔═╡ 4734248c-4542-4531-9b43-ee14aa10f306
function kl_divergence(μ, logvar)
	
	#### BEGIN SOLUTION
	n = size(μ,2)
	σ² = exp.(logvar)
	
	sum((1//2)*(μ.^2 .+ σ² .- logvar .- 1))/n
	
	#0.0f0

	#### END SOLUTION
	
end

# ╔═╡ e6283131-2e8b-453f-867c-56cd91a48527
begin
	
	__check_kl_div = let 
		Random.seed!(1)
		v = kl_divergence(randn(Float32, 2, 10), randn(Float32, 2, 10))
		v isa Float32 && v ≈ 1.3406578f0
	end

md"""
#### KL Divergence $(_check_complete(__check_kl_div))

We need to estimate the KL divergence between our variational distribution $q(\mathbf{h} | \mathbf{x})$ and the true distribution $p(\mathbf{h} | \mathbf{x}, \mathbf{W})$. As derived in the notes, we know this simplifies to:

```math
D_{KL}(q(\cdot | \mathbf{x}) || p(\cdot || \mathbf{x}, \mathbf{W}) = \mathbb{E}_{\mathbf{h} \sim q(\cdot | \mathbf{x})} \left[ \ln q(\mathbf{h} | \mathbf{x}) - \ln p(\mathbf{h})\right].
```

Now the question is how do we simplify this further. If we make the assumption that our distributions are gaussian, we can simplify even further:

```math
D_{KL}(q(\cdot | \mathbf{x}_i) || p(\cdot || \mathbf{x}_i, \mathbf{W}) = \frac{1}{2} \sum_{j} \mu_j^2 + \sigma^2_j - \ln(\sigma^2_j) - 1
```

While we don't have you prove this in the assignment, it would be a good excercise to go through. The above equation is for a single instance. In the below function, we will take the average over the batch of instances.

"""
end

# ╔═╡ e417b47c-9f48-4d3c-82be-0a890c306266
function binary_cross_entropy(x̂, x)
	#### BEGIN SOLUTION
	
	sum(x .* log.(x̂) + (1 .- x) .* log.(1 .- x̂)) / size(x,2)
	
	#0.0f0

	#### END SOLUTION
end

# ╔═╡ fc14bedb-e3df-4c2d-9902-1b9621e372be
begin
	
	__check_bce = let
		rng = Random.MersenneTwister(1)
		x, x̂ = Float32.(rand(rng, Bool, 28, 32)), Float32.(rand(rng, Float32, 28, 32))
		binary_cross_entropy(x̂, x) ≈ -28.755898f0
	end


md"""
#### Binary Cross Entropy $(_check_complete(__check_bce))

We want to maximize the log-liklihood of our reconstruction

```math
\mathbb{E}[\ln p(\mathbf{x} | \mathbf{h}, \mathbf{W})] = \frac{1}{L}\sum^L_{i=1} \ln p(\mathbf{x}_i | \mathbf{h}_i, \mathbf{W})
```

where $L$ is the batch size. We are assuming the MNIST data set is (approximately) a multi-dimensional bernoulli distribution. Thus the sum of log-likelihoods of our reconstruction can be estimated as 

```math
\sum_i \ln p(\mathbf{x}_i | \mathbf{h}_i, \mathbf{W}) = \sum_{i, j} (\mathbf{x}_{i,j} \ln(\hat{\mathbf{x}}_{i,j}) + (1-\mathbf{x}_{i,j}) \ln(1 - \hat{\mathbf{x}}_{i,j})
```

where $\hat{\mathbf{x}} = \text{Decoder}(\mathbf{h}_i)$. In the below implementation, be careful how you average the sum of log-likelihoods (i.e. what is `L`?). 

"""
end

# ╔═╡ dab0476f-8da9-4d6e-a418-9c4e0eda9e4a
md"""
### VAE on MNIST 

Below we build up the necessary utilities to train our VAE on the MNIST dataset.

The following two functions get the data from mnist (saving temporary files in a folder called "./tmp"). We then bin the data set so that any pixel $>0.5$ is white and everything else is black. This makes sure the dataset actually follows the above assumption we made when implementing the log-likelihood estimation function.
"""

# ╔═╡ b1a674fa-8893-4b6a-b38e-a71474165786
function get_mnist_data()
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true #necesarry modification
    X, Y = MLDatasets.MNIST.traindata(Float32, dir="./data")
    X = Float32.(reshape(X, 28^2, :) .> 0.5)
	X, Y
end

# ╔═╡ 0616c043-10e3-4d1e-8328-7a694a931dce
function get_mnist_test_data()
    X, Y = MLDatasets.MNIST.testdata(Float32, dir="./data")
    X = Float32.(reshape(X, 28^2, :) .> 0.5)
	X, Y
end

# ╔═╡ b3d2988a-d99b-4b16-bace-31f0134ed36b
md"""
Below are the arguments we use for running the VAE experiments. These are shared for both the VAE and CVAE experiments.
"""

# ╔═╡ e32e185d-bfc9-411a-ade0-c6cb843ef0d6
Base.@kwdef struct MNIST_Args
	seed = 1                # random seed
	batch_size = 128        # batch size
	epochs = 20             # number of epochs

	# optimizer
	η = 1e-3                # learning rate

	# network
	input_dim = 28^2        # image size
	latent_dim = 16         # latent dimension
	hidden_dim = 64         # hidden dimension
end

# ╔═╡ d1fd5e60-bb94-447e-9b35-39ef07d328b9
md"""
Below is the function we will use to create and train a VAE on the MNIST dataset. You need to complete two tasks for this function to work (labeled in the function).

**Create the VAE model:**

Using the VAE struct and Encoder struct you need to construct the VAE we are going to train. The shared portion of the Encoder will be a single layer with `tanh` activations and an output of `args.hidden_dim` (as well as passing in the other required arguments). The decoder will be a `Flux.Chain` with two layers and activations `tanh` and `sigmoid` respectively. The first layer will have an input the size of our latent state with `args.hidden_dim` as the output. The last layer will have an input of size `args.hidden_dim` and an output the size of our input.

**Note:** `tanh` and `sigmoid` are provided and don't need to be implemented. The functions just need to be passed as activations to the appropriate layers.

**Implement the training loop:**

Because our `elbo` loss implementation doesn't take a label, using the training loop defined by flux is not exactly what we need. Instead, we can write our own training loop! You will use two new functions provided by `Flux` to accomplish this:

- `gradient(loss, ps::Flux.Params)`: takes in a set of parameters to train and returns an `IdDict` contiaining the gradients of these parameters wrt the loss function. The loss function is a function which takes no arguments, you should create a lambda _or_ use a do block to pass this in:

- `Flux.Optimise.update!(opt, ps, grads)` this actually updates our model parameters in `ps=Flux.params(model)` with our optimizer `opt` according to the gradients `grads`.

The inner portion of your training loop should look like:
```julia
train_data, opt, model = #Defined somewhere
ps = Flux.params(model)

### Loops
grads = gradient(ps) do
	# batch[1] are the features, batch[2] are the labels.
	loss(model, batch[1], batch[2]) 
end
Flux.Optimise.update!(opt, ps, grads)
### End Loops
```

You will need to loop over `args.epochs` and batches of size `args.batch_size`. If you go look at the code for [`Flux.train!`](https://github.com/FluxML/Flux.jl/blob/ea26f45a1f4e93d91b1e8942c807f8bf229d5775/src/optimise/train.jl#L99) it is remarkebly simple and should give you a good idea what you need to accomplish: 
"""

# ╔═╡ 25215b53-9a20-4cbc-bcee-17ed057950aa
begin
	struct CVAE
		encoder::Encoder
		decoder::Flux.Chain
	end
	Flux.@functor CVAE # to make the struct work w/ Flux's autodiff
end

# ╔═╡ dec6630f-504b-44dd-9e0a-0f9e796691ff
function append_class_to_data(X, Y)
	bit_Y = zeros(Float32, 10, length(Y))
	Flux.Zygote.ignore() do
		for i in 1:length(Y)
			bit_Y[Y[i] + 1, i] = 1
		end
	end
	vcat(X, bit_Y)
end

# ╔═╡ de0ba0ba-c061-4cae-8198-79524d892487
function reconstruct(cvae::CVAE, X, Y)
	#### BEGIN SOLUTION
	
	X_labeled = append_class_to_data(X,Y)

	μ, logvar = encode_input(cvae.encoder, X_labeled)
	
	ZcondY = ((exp.(logvar)).^(1//2)) .* randn(Float32, size(μ)) .+ μ
	
	ZcondY_labeled = append_class_to_data(ZcondY,Y)
	
	X̂condY = cvae.decoder(ZcondY_labeled)
	
	return μ, logvar, X̂condY
	
	#zeros(Float32, size(cvae.encoder.μ.W, 1), size(X, 2)), zeros(Float32, size(cvae.encoder.logvar.W, 1), size(X, 2)), zero(X)

	#### END SOLUTION
end

# ╔═╡ f11e0cd6-ccd9-415e-9a20-a336227945f1
begin
	__check_recon_vae = let
		Random.seed!(1)
		encoder = Encoder(
			Dense(1, 2, tanh), 
			2, 
			2)
		decoder = Flux.Chain(
			Dense(2, 1, tanh), 
			Dense(1, 1, σ))
		vae = VAE(encoder, decoder)
		μ, logvar, x̂ = reconstruct(vae, rand(Float32, 1, 6))
		
		c_type = eltype(μ) <: Float32 && eltype(logvar) <: Float32 && eltype(x̂) <: Float32
		c_val = μ[2, 4] ≈ 0.30752954f0 && logvar[1, 6] ≈ -0.08888522f0 && x̂[1, 5] ≈ 0.66531307f0
		c_type && c_val
	end
	
md"""
### Reconstructing the input $(_check_complete(__check_recon_vae))

Below you will need to reconstruct the output using the VAE. To do this you will need to pass the inputs into the encoder, getting the parameters for the distribution we sample from. You should assume we are using a Normal distribution with

```math
\mathcal{N}\left(\mu, e^{logvar}\right).
```

Recall that if you have a standard normal random variable $X \sim \mathcal{N}(0, 1)$ you can change its mean and variance to $\mu, \sigma^2$ respectively by multiplying by $\sigma$ and adding $\mu$ like this $\sigma X + \mu$.
Also note we are learning $\log(\sigma^2)$ instead of $\sigma^2$. This helps with numerical stability, and means we can use a linear layer for estimating the variance. If we didn't learn the log we would have to ensure the variance output from the network is positive in some other way.
	
Below, we highly recommend you use either `1//2` or `0.5f0` for the exponent, and `randn(Float32, size...)` for sampling random normal numbers. This way we can ensure the types are stable throughout the reconstruction (this matters a lot for performance in Flux).
"""
end

# ╔═╡ 91aad47c-b715-42b2-8f79-074b016356c3
function elbo(vae, x)
	μ, logvar, x̂ = reconstruct(vae, x)
	@assert eltype(μ) <: Float32
	@assert eltype(logvar) <: Float32
	@assert eltype(x̂) <: Float32
	
	len = size(x)[end]
	
    # KL-divergence
    kl_q_p = kl_divergence(μ, logvar)::Float32
	
	# cross entropy
    logp_x_z = binary_cross_entropy(x̂, x)::Float32
	
	# return the final loss of elbo.
	#### BEGIN SOLUTION
	
	kl_q_p - logp_x_z
	
	#0.0f0

	#### END SOLUTION
end

# ╔═╡ 7402bdc2-eb6f-4837-96f9-2d113d826b55
begin
	__check_elbo = let
		Random.seed!(1)
		encoder = Encoder(
			Dense(1, 2, tanh), 
			2, 
			2)
		decoder = Flux.Chain(
			Dense(2, 1, tanh), 
			Dense(1, 1, σ))
		vae = VAE(encoder, decoder)

		elbo(vae, Float32.(rand(Bool, 1, 32))) ≈ 0.78396136f0
	end

md"""
#### elbo $(_check_complete(__check_elbo))

Write the final output of the elbo loss function below.
"""
end

# ╔═╡ 60abff0f-1a7a-47a4-94b3-880a5894d139
function train_mnist_vae(; kwargs...)

	args = MNIST_Args(;kwargs...)
	
	args.seed > 0 && Random.seed!(args.seed)

	train_data = get_mnist_data()

	# Construct the VAE model
	#### BEGIN SOLUTION
	
	encoder = Encoder(Dense(args.input_dim, args.latent_dim, tanh),
					  args.latent_dim, args.latent_dim)
		
	decoder = Flux.Chain(Dense(args.latent_dim, args.hidden_dim, tanh),
						 Dense(args.hidden_dim, args.input_dim, sigmoid))
	
	vae = VAE(encoder, decoder)
	
	#vae = nothing

	#### END SOLUTION
	
	ps = Flux.params(vae)
	
	opt = ADAM(args.η)
	
	# Write the training loops over epochs and batches.
	#### BEGIN SOLUTION
	
	num_batches = size(train_data[1],2) ÷ args.batch_size
	
	for epoch in 1:args.epochs
		for batch_idx in 1:num_batches
			tail_idx = batch_idx*args.batch_size
			batch = train_data[1][:,tail_idx-args.batch_size+1:tail_idx]
				
			∇L = gradient(() -> elbo(vae, batch), ps)
			Flux.Optimise.update!(opt, ps, ∇L)
		end
	end

	#### END SOLUTION
	
	vae
end

# ╔═╡ 529abd48-aa20-445d-ba5d-44b31749de70
begin
	__check_cvae_recon = let
		Random.seed!(1)
		encoder = Encoder(
			Dense(1 + 10, 2, tanh), 
			2,
			2)
		decoder = Flux.Chain(
			Dense(2 + 10, 2, tanh), 
			Dense(2, 1, σ))
		cvae = CVAE(encoder, decoder)
		μ, logvar, x̂ = reconstruct(cvae, rand(Float32, 1, 16), rand(0:9, 1, 16))
		μ[1, 2] ≈ -0.06641408f0 && logvar[2, 10] ≈ 0.07583164f0 && x̂[1, 16] ≈ 0.40480414f0
	end
md"""
### Conditional VAE

Below we implement the necessary components for learning a conditional VAE. The main difference between a conditional VAE (CVAE) and a vanilla VAE is in the input (i.e. what our function is conditioned on) for both the encoder and decoder. The CVAE will take an extra set of inputs (concatenated to the image or latent state), which is a one-hot encoding of the label.

You will need to write the `reconstruct` $(_check_complete(__check_cvae_recon)) function for the CVAE. This should be very similar to the `VAE` version but using the `append_class_to_data` function to append the appropriate label to both the image data `X` and to the latent state `h`. As you can see the CVAE and VAE are almost identical.
"""
end

# ╔═╡ 33eacfac-60bd-4b89-afd3-1e92b61cf55d
function celbo(cvae::CVAE, X, Y)
	μ, logvar, X̂ = reconstruct(cvae, X, Y)
	@assert eltype(μ) <: Float32
	@assert eltype(logvar) <: Float32
	@assert eltype(X̂) <: Float32
	len = size(X)[end]
	
	#### BEGIN SOLUTION
	
	kl_divergence(μ, logvar) - binary_cross_entropy(X̂, X)

	#### END SOLUTION
end

# ╔═╡ 86b19883-6574-4f10-a822-4186c4c0235c
begin
	__check_celbo = let
		Random.seed!(1)
		encoder = Encoder(
			Dense(1 + 10, 2, tanh), 
			2,
			2)
		decoder = Flux.Chain(
			Dense(2 + 10, 2, tanh), 
			Dense(2, 1, σ))
		cvae = CVAE(encoder, decoder)
		celbo(cvae, Float32.(rand(Bool, 1, 16)), rand(0:9, 1, 16)) ≈ 
1.0942628f0
	end

md"""
Below we will be implementing `celbo` $(_check_complete(__check_celbo)) which is the conditional elbo loss you derived in the written portion. This should look very similar to the above `elbo` loss and doesn't require any new helper functions (i.e. you should re-use `kl_divergence` and `binary_cross_entropy`).
"""
end

# ╔═╡ 63d4b4f1-bc17-4a8e-9a5c-76ef129c905c
let
	__viz_str = _check_complete(__train_simple_network)
	__gai_str = _check_complete(__get_act_index_check)
	
	__vae_recon_str = _check_complete(__check_recon_vae)
	__kl_div_str = _check_complete(__check_kl_div)
	__bce_str = _check_complete(__check_bce)
	__elbo_str = _check_complete(__check_elbo)

	__cvae_recon_str = _check_complete(__check_cvae_recon)
	__celbo_str = _check_complete(__check_celbo)
	
md"""
## Outline

- How do ANNs represent and transform data? (see written part)
- VAE
  - Reconstruct the input $(__vae_recon_str)
  - Implement ELBO $(__elbo_str): `kl_divergence` $(__kl_div_str), `binary_cross_entropy` $(__bce_str), `elbo` $(__elbo_str)
  - `train_mnist_cvae`: This is not checked by an automatic test.
- CVAE
  - Reconstruct the input $(__cvae_recon_str)
  - Implement CELBO $(__celbo_str)
  - `train_mnist_cvae`: This is not checked by an automatic test.

"""
end

# ╔═╡ 64fa9dbe-f9f4-407d-8e19-02831a097eb6
md"""
Below you will implement the training function. This should look very similar to `train_mnist_vae` with some minor differences. Specifically,
- the loss function will be different
- the CVAE's network will be almost exactly the same as the VAE above except the input to the encoder and decoder will be slightly different. (hint: think about the `reconstruct` for CVAE you just implemented and remember mnist has 10 classes).
"""

# ╔═╡ e31da691-aad0-496c-addc-6730763ee7bb
function train_mnist_cvae(; kwargs...)

	args = MNIST_Args(;kwargs...)
	
	args.seed > 0 && Random.seed!(args.seed)

	train_data = get_mnist_data()
	
	@assert train_data[1] isa Matrix{Float32}

	# Construct the VAE
	#### BEGIN SOLUTION
	
	encoder = Encoder(Dense(args.input_dim+10, args.latent_dim, tanh),
					  args.latent_dim, args.latent_dim)
		
	decoder = Flux.Chain(Dense(args.latent_dim+10, args.hidden_dim, tanh),
						 Dense(args.hidden_dim, args.input_dim, sigmoid))
	
	cvae = CVAE(encoder, decoder)
	
	#cvae = nothing

	#### END SOLUTION
	
	ps = Flux.params(cvae)
	
	opt = ADAM(args.η)
	
	# Write the training loops over epochs and batches.
	#### BEGIN SOLUTION
	
	num_batches = size(train_data[1],2) ÷ args.batch_size
	
	for epoch in 1:args.epochs
		for batch_idx in 1:num_batches
			tail_idx = batch_idx*args.batch_size
			batch = train_data[1][:,tail_idx-args.batch_size+1:tail_idx]
			label = train_data[2][tail_idx-args.batch_size+1:tail_idx]'
				
			∇L = gradient(() -> celbo(cvae, batch, label), ps)
			Flux.Optimise.update!(opt, ps, ∇L)
		end
	end
	

	#### END SOLUTION
	cvae
end

# ╔═╡ 5497634b-361e-4ad1-b800-215b91426af2
md"""
### Results

Below, we compare the output of a VAE with that of a Conditional VAE. Training each VAE takes approximately 2.5 minutes using a single core.

Train VAE: $(@bind _check_train_vae PlutoUI.CheckBox()) \
Train CVAE: $(@bind _check_train_cvae PlutoUI.CheckBox())
"""

# ╔═╡ acea195c-0075-4217-8235-6e444d2024a8
trained_vae = _check_train_vae ? train_mnist_vae() : nothing

# ╔═╡ c0eff92e-b378-4da1-b44a-02fce5095612
trained_cvae = _check_train_cvae ? train_mnist_cvae() : nothing

# ╔═╡ 08cc03f4-9338-422f-9884-aa2745d67812
md"""
Below we visualize the reconstruction of the two VAEs. If the image is black, this means you haven't trained the VAE that it is labeled as. The left most column is the ground truth from the dataset. The middle column is the vanilla VAE, and the right most column is the Conditional VAE. Both models are trained using 20 epochs, and the default parameters defined in `MNIST_Args`.
"""

# ╔═╡ e6cdb6d4-6c4a-473b-85cc-b45f7de2409c
(mnist_data_X, mnist_data_Y) = get_mnist_test_data();

# ╔═╡ 3b720427-6fb8-46e9-8cbe-93d69c0b268d
let
	indicies = [10, 9392, 232, 1, 132]
	plts = []
	for i ∈ indicies
		
		p̂_vae = if trained_vae isa Nothing
			zero(getobs(mnist_data_X, i))
		else
			reconstruct(trained_vae, getobs(mnist_data_X, i))[end] .> 0.5
		end
		p̂_cvae = if trained_cvae isa Nothing
			zero(getobs(mnist_data_X, i))
		else
			reconstruct(trained_cvae, getobs(mnist_data_X, i), getobs(mnist_data_Y, i))[end] .> 0.5
		end
		p_rs = permutedims(reshape(getobs(mnist_data_X, i), 28, :), [2, 1])
		
		plt_true = plot(
			Gray.(p_rs), 
			title=i==indicies[1] ? "Ground Truth" : "", 
			axis=nothing)

		p̂_vae_rs = permutedims(reshape(p̂_vae, 28, :), [2, 1])
		plt_vae = plot(
			Gray.(p̂_vae_rs), 
			title=i==indicies[1] ? "VAE" : "", 
			axis=nothing)

		p̂_cvae_rs = permutedims(reshape(p̂_cvae, 28, :), [2, 1])
		plt_cvae = plot(
			Gray.(p̂_cvae_rs), 
			title=i==indicies[1] ? "CVAE" : "", 
			axis=nothing)
		
		push!(plts, plot(plt_true, plt_vae, plt_cvae, layout=(1,3)))
	end
	plot(plts..., layout = (length(plts), 1), size=(800, 1200))
end

# ╔═╡ 19852a30-3a3d-4be9-aa52-e30e303059ee
md"""
The CVAE and VAE can also be used to create new data. To do this we sample from our prior (i.e $\mathcal{N}(0, I)$). For the CVAE we append the appropriate class label (one hot encoded). Notice how the CVAE is able to consistently reproduce a single digit, while the VAE will produce a random set of digits!

You will need the CVAE visualization to answer some questions in the written portion of the assignment.
"""

# ╔═╡ 8a4b1faa-e8e6-49fe-9a33-8bb78fb2126e
md"
**CVAE:**
"

# ╔═╡ 39b53c97-608e-41d6-bfea-641b8247dbe9
let
	class_label = 0
	nbs = 16
	p̂_vae = if trained_cvae isa Nothing
		randn(28^2, 16)
	else
		h = randn(Float32, size(trained_cvae.encoder.μ.W, 1), nbs)
		h_n = append_class_to_data(h, fill(class_label, nbs)) # append label to hiddenstate.
		trained_cvae.decoder(h_n)
	end
	p̂_indv = [Gray.(permutedims(reshape(p̂_vae[:, i], 28, :), [2, 1])) for i in 1:nbs]
	
	plt_vae = [plot(
		p̂_indv[i], 
		# title=i==indicies[1] ? "VAE" : "", 
		axis=nothing) for i in 1:nbs]
	plot(plt_vae...)
end

# ╔═╡ 35a70493-00ec-429d-a66a-bdf4a4abd35c
md"
**VAE:**
"

# ╔═╡ 146886ac-5a0f-4775-8984-65a1b043f0b6
let
	nbs = 16
	p̂_vae = if trained_vae isa Nothing
		randn(28^2, 16)
	else
		h = randn(Float32, size(trained_vae.encoder.μ.W, 1), nbs)
		h_n = h
		trained_vae.decoder(h_n)
	end
	p̂_indv = [Gray.(permutedims(reshape(p̂_vae[:, i], 28, :), [2, 1])) for i in 1:nbs]
	
	plt_vae = [plot(
		p̂_indv[i], 
		axis=nothing) for i in 1:nbs]
	plot(plt_vae...)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
MLDataUtils = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Flux = "~0.12.7"
MLDataUtils = "~0.5.4"
MLDatasets = "~0.5.12"
PlutoUI = "~0.7.16"
StatsPlots = "~0.14.28"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "a8101545d6b15ff1ebc927e877e28b0ab4bc4f16"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.36"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "84cf7d0f8fd46ca6f1b3e0305b4b4a37afe50fd6"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.0"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e747dac84f39c62aff6956651ec359686490134e"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.0+0"

[[BufferedStreams]]
deps = ["Compat", "Test"]
git-tree-sha1 = "5d55b9486590fdda5905c275bb21ce1f0754020f"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.0.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "2c8329f16addffd09e6ca84c556e2185a4933c64"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.5.0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "RealDot", "Statistics"]
git-tree-sha1 = "035ef8a5382a614b2d8e3091b6fdbb1c2b050e11"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.12.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "3533f5a691e60601fe60c90d8bc47a27aa2907ec"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.0"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataDeps]]
deps = ["BinaryProvider", "HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "4f0e41ff461d42cfc62ff0de4f1cd44c6e6b3771"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.7"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "7220bc21c33e990c14f4a9a319b1d242ebc5b269"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.1"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "09d9eaef9ef719d2cd5d928a191dc95be2ec8059"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.5"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3fcfb6b34ea303642aee8f85234a0dcd0dc5ce73"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.22"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flux]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "NNlibCUDA", "Pkg", "Printf", "Random", "Reexport", "SHA", "SparseArrays", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "e4ade0790850bb16b5309945658fa4e7626226f1"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.7"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "63777916efbcb0ab6173d09a658fb7f2783de485"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.21"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
git-tree-sha1 = "e4768c3b7f597d5a352afa09874d16e3c3f6ead2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.7"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GPUArrays]]
deps = ["Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "7772508f17f1d482fe0df72cabc5b55bec06bbe0"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.1.2"

[[GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "ba475ea91facd7bde9f0f113f60e975882b4f5ca"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.13.6"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "d189c6d2004f63fd3c91748c458b09f26de0efaa"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.61.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cafe0823979a5c9bff86224b3b8de29ea5a44b2e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.61.0+0"

[[GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "74ef6288d071f58033d54fd6708d4bc23a8b8972"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "698c099c6613d7b7f151832868728f426abe698b"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.7"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fd83fa0bde42e01952757f01149dd968c06c4dba"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.0+1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "5efcf53d798efede8fee5b2c8b09284be359bf24"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.2"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "f0c6489b12d28fb4c2103073ec7452f3423bd308"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.1"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "7d58534ffb62cd947950b3aa9b993e63307a6125"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "46092047ca4edc10720ecab437c42283cd7c44f3"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.6.0"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6a2af408fe809c4f1a54d2b3f188fdd3698549d6"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.11+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LearnBase]]
deps = ["LinearAlgebra", "StatsBase"]
git-tree-sha1 = "47e6f4623c1db88570c7a7fa66c6528b92ba4725"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "6193c3815f13ba1b78a51ce391db8be016ae9214"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.4"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "5c62992f3d46b8dce69bdd234279bb5a369db7d5"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.1"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MLDataPattern]]
deps = ["LearnBase", "MLLabelUtils", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e99514e96e8b8129bb333c69e063a56ab6402b5b"
uuid = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
version = "0.5.4"

[[MLDataUtils]]
deps = ["DataFrames", "DelimitedFiles", "LearnBase", "MLDataPattern", "MLLabelUtils", "Statistics", "StatsBase"]
git-tree-sha1 = "ee54803aea12b9c8ee972e78ece11ac6023715e6"
uuid = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
version = "0.5.4"

[[MLDatasets]]
deps = ["BinDeps", "ColorTypes", "DataDeps", "DelimitedFiles", "FixedPointNumbers", "GZip", "JSON3", "MAT", "PyCall", "Requires"]
git-tree-sha1 = "3ad568c323866280500096860a5e2a76b2e7e12d"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.5.12"

[[MLLabelUtils]]
deps = ["LearnBase", "MappedArrays", "StatsBase"]
git-tree-sha1 = "3211c1fdd1efaefa692c8cf60e021fb007b76a08"
uuid = "66a33bbf-0c2b-5fc8-a008-9da813334f0a"
version = "0.5.6"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "5203a4532ad28c44f82c76634ad621d7c90abcbd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.29"

[[NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "04490d5e7570c038b1cb0f5c3627597181cc15a9"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.1.9"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "d911b6a12ba974dabe2291c6d450094a7226b372"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.1"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "25007065fa36f272661a0e1968761858cc880755"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "4c8a7d080daca18545c56f1cac28710c362478f3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.16"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "4ba3651d33ef76e24fef6a598b63ffd1c5e1cd17"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.5"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Random123]]
deps = ["Libdl", "Random", "RandomNumbers"]
git-tree-sha1 = "0e8b146557ad1c6deb1367655e052276690e71a3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.4.2"

[[RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "eb007bb78d8a46ab98cd14188e3cec139a4476cf"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.28"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7cb456f358e8f9d102a8b25e8dfedf58fa5689bc"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.13"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "0fc9959bcabc4668c403810b4e851f6b8962eac9"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.29"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─5b40470e-6195-4307-8c78-993da5c79699
# ╠═fec77c57-c21a-42c4-8e39-3c21c804504c
# ╠═6d0284b7-1b4e-4241-a796-034c53b06fef
# ╟─72fef447-cdf6-443a-ae6b-53c217602d98
# ╟─5897039a-9571-4648-b1f6-a74efb6aef3b
# ╠═54ff2a52-13c5-4e55-ad57-7acf5474cae4
# ╟─c69ce5c8-cd42-4ae2-aaa9-74d0daa5875f
# ╠═dfb12ef6-2968-40bc-a7eb-5284dec0c359
# ╟─63d4b4f1-bc17-4a8e-9a5c-76ef129c905c
# ╟─5b6b85d6-25f5-4ac9-9fe3-5e85c432f0f0
# ╟─6b7db8c9-6a3a-49ae-926a-35a8f8bfb8e8
# ╠═b37d3b82-fcc2-4ac9-94cc-86174481ba86
# ╠═31543a74-a116-4861-b153-30e6bcaecce8
# ╠═b102d8ee-1191-4580-aa53-09740cf608a4
# ╠═f9596fd6-b1fa-43d4-b84c-9a1151db372d
# ╟─0371d19e-9684-4b81-b1cb-91409fcab2d7
# ╠═226149d2-a4ce-4713-8ff4-420c82206b07
# ╟─49288405-b370-4cfb-a0e3-01d3b9126470
# ╟─ae2b42f4-5a0c-4e0f-b14c-d6b47004cc4f
# ╠═8b85f477-674e-403e-a834-36eb53ab4c30
# ╠═df07dbe6-3983-48a8-b92a-40ccd4f08a87
# ╟─54f27f56-945a-48ae-96cf-51e15ca27cbe
# ╟─f8f90dbf-3b35-4ec3-8b65-1652aab3f954
# ╠═c4df8697-e133-4e92-b271-4ba5371566fa
# ╠═697d1728-1ec3-4a76-9b9d-de2831e39214
# ╟─3d49ff71-63cf-495e-b87c-e0f3ba7b925d
# ╠═bd1e57af-6373-4815-a305-86e03f57c64e
# ╟─7b6fc4a5-a630-42e7-9924-c2862bebfdd1
# ╠═356d6228-0b23-441f-826a-ac8b1cf495e8
# ╟─f64a54e5-930d-46fd-9741-6cd87696b2e7
# ╠═36fd8579-69bb-4e6d-98a3-5e80e2a2f3b9
# ╟─5d5019a1-42fa-4076-9de1-71c575133f50
# ╠═b5cbed07-f2f0-4360-9714-5a7218831e55
# ╟─a534dcbd-33f4-4920-950b-ae3f34142bd8
# ╟─c7f2fc33-4c0d-402a-a8bf-d686458d1b67
# ╟─0390b28b-0b50-424c-8f56-4c2dd01afdc5
# ╟─46d0b27d-4cb2-4916-a226-9ab34e561cd9
# ╟─eb82338f-bfc3-4ab8-a30a-abf2ec333593
# ╟─299c202e-b1a9-4e90-8efc-b15fb44c8e5e
# ╠═6b366eac-6b94-44a8-969a-1148577b251d
# ╟─7bd1b70a-db39-497d-83ef-1dbd587503de
# ╠═b4246952-2272-4e93-957d-5f68c511752c
# ╟─49f77c7b-6e6d-4cf9-a0ea-b90c99891570
# ╠═09d8ef7a-b96c-4f50-a470-acbbc2d5b895
# ╟─390fc91e-0fb9-4f32-9a3a-54c4df1954bf
# ╟─5b6519a7-00dd-4e18-a8e9-a9eaed60d60e
# ╠═8c41e1cf-8139-4e61-b92b-b7d201e91d2d
# ╠═38f9f22f-fabf-4278-9a5a-2f77eac7c2db
# ╠═d5fa8fc7-ddd0-4739-8ed2-d3c45995e179
# ╟─f11e0cd6-ccd9-415e-9a20-a336227945f1
# ╠═32bfe0e2-21f0-4d5e-9c60-2a2ed5ca31c3
# ╟─131a7831-250a-47e0-90f7-9cb14db0a53b
# ╟─e6283131-2e8b-453f-867c-56cd91a48527
# ╠═4734248c-4542-4531-9b43-ee14aa10f306
# ╟─fc14bedb-e3df-4c2d-9902-1b9621e372be
# ╠═e417b47c-9f48-4d3c-82be-0a890c306266
# ╟─7402bdc2-eb6f-4837-96f9-2d113d826b55
# ╠═91aad47c-b715-42b2-8f79-074b016356c3
# ╟─dab0476f-8da9-4d6e-a418-9c4e0eda9e4a
# ╠═b1a674fa-8893-4b6a-b38e-a71474165786
# ╠═0616c043-10e3-4d1e-8328-7a694a931dce
# ╟─b3d2988a-d99b-4b16-bace-31f0134ed36b
# ╠═e32e185d-bfc9-411a-ade0-c6cb843ef0d6
# ╟─d1fd5e60-bb94-447e-9b35-39ef07d328b9
# ╠═60abff0f-1a7a-47a4-94b3-880a5894d139
# ╟─529abd48-aa20-445d-ba5d-44b31749de70
# ╠═25215b53-9a20-4cbc-bcee-17ed057950aa
# ╠═dec6630f-504b-44dd-9e0a-0f9e796691ff
# ╠═de0ba0ba-c061-4cae-8198-79524d892487
# ╟─86b19883-6574-4f10-a822-4186c4c0235c
# ╠═33eacfac-60bd-4b89-afd3-1e92b61cf55d
# ╟─64fa9dbe-f9f4-407d-8e19-02831a097eb6
# ╠═e31da691-aad0-496c-addc-6730763ee7bb
# ╟─5497634b-361e-4ad1-b800-215b91426af2
# ╠═acea195c-0075-4217-8235-6e444d2024a8
# ╠═c0eff92e-b378-4da1-b44a-02fce5095612
# ╟─08cc03f4-9338-422f-9884-aa2745d67812
# ╠═3b720427-6fb8-46e9-8cbe-93d69c0b268d
# ╠═e6cdb6d4-6c4a-473b-85cc-b45f7de2409c
# ╟─19852a30-3a3d-4be9-aa52-e30e303059ee
# ╟─8a4b1faa-e8e6-49fe-9a33-8bb78fb2126e
# ╠═39b53c97-608e-41d6-bfea-641b8247dbe9
# ╟─35a70493-00ec-429d-a66a-bdf4a4abd35c
# ╠═146886ac-5a0f-4775-8984-65a1b043f0b6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
