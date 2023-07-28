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

# ╔═╡ dc004086-db6e-4813-841e-d427520402f7
begin
	using CSV, DataFrames, StatsPlots, PlutoUI, Random, Statistics
	using LinearAlgebra: dot, norm, norm1, norm2, I
	using Distributions: Distributions
	using MultivariateStats: MultivariateStats, PCA
	using StatsBase: StatsBase

end

# ╔═╡ 75441ce6-2137-4fcf-bba2-6ed67b9acb59
begin
	_check_complete(complete) = complete ? "✅" : "❌"
	
	md"""
	# Setup

	this section loads and install all the packages. You should be setup already from assignment 1, but if not please read and follow the `instructions.md` for further details.
	"""
end

# ╔═╡ c9797979-7d16-4ba4-8f82-c7ec33f6e517
plotly() # In this notebook we use the plotly backend for Plots.

# ╔═╡ 693a3933-c1c2-4249-8c03-f5151267222f
md"""
### !!!IMPORTANT!!!

Insert your details below. You should see a green checkmark.
"""

# ╔═╡ def97306-1703-42bc-bc09-da623c545e87
student = (name="Patrick Wyrod", email="pwyrod@ualberta.ca", ccid="pwyrod", idnumber=1584645)

# ╔═╡ bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
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

# ╔═╡ 7b513344-1cad-4eef-9faf-e77ba176323e
md"""
# Models

"""

# ╔═╡ 4f4029a2-c590-4bd3-a0db-d2380d4b4620
md"""
## The model interface

- `AbstractModel`: This is an abstract type which is used to derive all the model types in this assignment
- `predict`: This takes a matrix of samples and returns the prediction doing the proper data transforms.
- `get_features`: This transforms the features according to the non-linear transform of the model (which is the identity for linear).
- `update_transform!`: This "trains" the transform. Using the data provided we update the PCA or Kernel prototypes used according to the strategy.
- `get_linear_model`: All models are based on a linear model with transformed features, and thus have a linear model.
- `copy`: This returns a new copy of the model.
"""

# ╔═╡ dcfecc35-f25c-4856-8429-5c31d94d0a42
"""
	AbstractModel

Used as the root for all models in this notebook. We provide a helper `predict` function for `AbstractVectors` which transposes the features to a row vector. We also provide a default `update_transform!` which does nothing.
"""
abstract type AbstractModel end

# ╔═╡ d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
predict(alm::AbstractModel, x::AbstractVector) = predict(alm, x')[1]

# ╔═╡ 7cd46d84-a74c-44dc-8339-68010924bc39
update_transform!(AbstractModel, args...) = nothing

# ╔═╡ 8745fec1-47c8-428b-9ea4-1e6828618830
md"
#### Linear Model

As before, we define a linear model as a linear map
```math
\hat{y} = \mathbf{w}^\top x
```

or with a data matrix $$X$$ of size `(samples, features)`

```math
\hat{Y} = X \mathbf{w}
```

To make the predict function simpler we provide a convenience predict function for all abstract models which transforms a `Vector` (which in julia is always a column vector), to a row vector (or a 1xn matrix). So you can call `predict(model, rand(10))` without worrying about whether `x` is a column or row vector. You will still need to pay attention to this when implementing future code.

"

# ╔═╡ 2d43a3ba-2a2c-4114-882f-5834d42e302a
begin
	struct LinearModel <: AbstractModel
		W::Matrix{Float64} # Aliased to Array{Float64, 2}
	end
	
	LinearModel(in, out=1) = 
		LinearModel(zeros(in, out)) # feture size × output size
	
	predict(lm::LinearModel, X::AbstractMatrix) = X*lm.W
	Base.copy(lm::LinearModel) = LinearModel(copy(lm.W))
	# get_features(::LinearModel, x) = x
end

# ╔═╡ 401bf2ad-53b9-44e7-be90-41f749d98e66
html"<h4 id=kernel_regression>Using kernels as similarity features</h4>"

# ╔═╡ f4dc2bf9-d1bd-4683-9306-8b77de643266
begin
	"""
		KernelModel
	
	This model transforms the features and then uses a linear model to learn. Notice how the structure has three components. The `kern` is the kernel function used to build the features
	"""
	mutable struct KernelModel{F, PSS} <: AbstractModel # F is a generic type for the kernel fucntion.
		kern::F
		prototype_selection_strategy::PSS
		prototypes::Vector{Vector{Float64}}
		model::LinearModel
	end

	function KernelModel(kern, prototypes::Vector, out::Int=1)

		@assert kern(rand(5), rand(5)) isa Float64
		@assert length(prototypes) > 0
		
		KernelModel(kern, nothing, prototypes, LinearModel(length(prototypes), out))
	end
	
	function KernelModel(kern, css, out::Int=1) # this creates a "blank" kenrel model where we have to find the prototypes.

		@assert kern(rand(5), rand(5)) isa Float64
		
		KernelModel(
			kern,
			css,
			Vector{Float64}[],
			LinearModel(0, out))
	end
	
	get_linear_model(km::KernelModel) = km.model
	Base.copy(m::KernelModel) = KernelModel(m.kern, m.prototype_selection_strategy, deepcopy(m.prototypes), copy(m.model))
end

# ╔═╡ 84d3240a-4350-43a5-98bf-5dd8daf2586b
md"""
The first function you need to implement is `get_features` which transforms a matrix of features `X` with dimensions `(num_samples, features)` according to the kernel function $K$ and the collection of prototypes $C$. The new features $\tilde{x}$ for a sample $x \doteq X_{j, :}$ will be a vector with components

```math
\tilde{x}_i = K(x, C_i)
```

"""

# ╔═╡ cdbe526a-d0c1-45f8-9935-6de78c19304c
function get_features(km::KernelModel, X::AbstractMatrix)::Matrix
	zeros(size(X, 1), length(km.prototypes))
	#### BEGIN SOLUTION
	
	Y = zeros(size(X, 1), length(km.prototypes))
	
	for i in 1:size(X,1)
		Y[i,:] .= [km.kern(X[i,:], c) for c in km.prototypes]
	end
	
	return Y
	
	#### END SOLUTION
end

# ╔═╡ fd8a21a8-3683-4691-af38-48178c805332
md"Notice how `get_features` interacts with `predict`."

# ╔═╡ 157e00c4-9154-4e2d-84cc-0c73ea25549d
predict(km::KernelModel, x::AbstractMatrix) = predict(km.model, get_features(km, x))

# ╔═╡ 8171b686-c64a-4e9f-b6a9-14b08e1e559c
md"""
`update_transform!` updates the prototypes of our model, and is used later on. This function uses a function `select_prototypes`. You will implement this function for a random subselection and a simple prototype selection strategy [here](#prototypes).
"""

# ╔═╡ d5fa8c7e-5baf-4dd3-9d39-98b3ab6fdf7e
"""
	cosine_similairity()

This returns a function which takes two vectors and returns the cosine similarity between them.
"""
function cosine_similarity() # also known as the RBF
	function similarity(x::AbstractVector, c::AbstractVector)
		0.0
		## BEGIN SOLUTION
		
		dot(x,c)/(norm(x)*(norm(c)))
		

		## END SOLUTION
	end
	similarity
end

# ╔═╡ 2ac36388-1789-4133-8dd0-1797fd19899e

"""
	RBF(σ)

This returns a function which takes two vectors and applies the rbf function. Use `σ` in the RBF kernel as if it was defined as a global. This will attach the passed value to the new function.
"""
function RBF(σ::F) where {F<:AbstractFloat} # also known as the RBF
	function similarity(x::AbstractVector, c::AbstractVector)
		0.0
		# Use σ as the standard deciation of the RBF
		## BEGIN SOLUTION
		
		exp((-1/(2*σ^2)) * norm(x-c)^2)

		## END SOLUTION
	end
	similarity
end

# ╔═╡ 8044511f-77e4-4d77-b2a4-a04c24eab53a
begin
	
	__get_kernel_features_done = let n=5
		# 10 prototypes with dimensions 5
		km = KernelModel(cosine_similarity(), [rand(n) for i in 1:10], 1)
		X = rand(7, n)
		gf = (x)->[km.kern(x, c) for c in km.prototypes]
		feats = reduce(
			vcat, 
			permutedims.(gf.([x for x in eachrow(X)])))
		feats_2 = get_features(km, X)
		feats_3 = reduce(
			vcat, 
			get_features.((km,), permutedims.([x for x in eachrow(X)])))
		feats_4 = reduce(
			vcat, 
			get_features.((km,), [X[i:i, :] for i in 1:size(X, 1)]))

		all(feats .≈ feats_2 .≈ feats_3 .≈ feats_4) && !all(feats .== 0.0)
	end
	
	__cos_sim = cosine_similarity()(ones(5), ones(5)) ≈ 1
	__rbf = (RBF(0.1)(ones(5), ones(5)) == 1.0) &&
			(RBF(1.0)(ones(5), zeros(5)) ≈ 0.08208499862389876)
	
	
	__kernel_regression_check_1 = __get_kernel_features_done && 
								  __cos_sim &&
								  __rbf
	
	md"""

	In this section we will be defining the components to use kernel functions as similarity features to transform our input data. As noted in the notes, this is not the same as implementing kernel regression using the kernel trick. To implement the `KernelModel` we need to implement a helper function and some similiarity functions to use with the model!
	
	- `get_features` $(_check_complete(__get_kernel_features_done))
	- `cosine_similarty` $(_check_complete(__cos_sim))
	- `RBF` $(_check_complete(__rbf))
	
	"""
end

# ╔═╡ 15ff9bb7-bff4-4cf4-830c-79197a6c1c21
md"""
##### Similarity functions $(_check_complete(__cos_sim && __rbf))
"""

# ╔═╡ 1ba63749-06cb-4f84-a018-ae3c4aa5992f
md"""
The cosine similarity $(_check_complete(__cos_sim)) measures the angle between two vectors. It is defined as

```math
\frac{\langle x, c \rangle}{\vert\vert x \vert\vert_2 \vert\vert c \vert\vert_2}
```
"""

# ╔═╡ b1aa4dff-b2f2-4151-9e02-46199674a956
md"""
The RBF kernel $(_check_complete(__rbf)) is widely used in kernel regression and has many appealing properties. Check the notes for how to implement this function.
"""

# ╔═╡ 69ee4c2d-b133-4b82-b4d1-48988d26d184
md"""
# Learning

In this section, we will be implementing Lasso regression. We provide implementations of OLS and Ridge regression as from prior assignments as examples for how the interface works!
"""

# ╔═╡ 3e04ec5e-393a-4031-9507-7e3339118ddc
html"<h2 id=ols_ridge>Recap: OLS, Ridge, and Lasso</h2>"

# ╔═╡ d1c5c8c8-1f29-4a60-b362-ae3d300413cd
md"""

Before we get into gradient descent, lets review ordinary least squares and two common regularization techniques. The first, `Ridge`, should be familiar, and an implementation is provided. The second, `Lasso`, is for regularization when `p=1`. You will be filling in this implementation.

"""

# ╔═╡ ae98fbde-b1c4-46b6-aca3-a222acf20bc8
md"""
### Ridge

Remember ridge regression corresponds to L2 regularization with the cost function:

```math
c(X) = (X^\top \mathbf{w} - Y)^\top (X^\top \mathbf{w} - Y) + \lambda \mathbf{w}^\top \mathbf{w}
```

And a MAP solution
```math
\mathbf{w}_{\text{MAP}} = (X^\top X + \lambda I)^{-1} X^\top Y
```

where $$I$$ is the identity matrix. We can represent an OLS solution with `λ=0.0` and we provide a convenience constructor `OLS()`
"""

# ╔═╡ bf05c0a2-de24-4a44-841c-8a7f4b40f8b8
struct Ridge 
	λ::Float64
end

# ╔═╡ 68386c6b-d025-4b84-8afb-6879d74f74a7
Base.copy(ridge::Ridge) = Ridge(ridge.λ)

# ╔═╡ 4c4ad7c7-6e81-49c1-92cd-1d133cd2c905
OLS() = Ridge(0.0)

# ╔═╡ 57931788-0280-4c73-b7e9-8e2c581fca9e
function train!(ridge::Ridge, model::LinearModel, X, Y)
	λ = ridge.λ
	n = size(X, 2)
	model.W .= inv(X'*X + λ * I(n)) * X' * Y
end

# ╔═╡ 350a0c24-fb1e-4b62-b9e9-26bba9ac35f3
struct Lasso
	λ::Float64
	τ::Float64 # Tolerance
end

# ╔═╡ 9aabea45-d91d-475f-93ed-a5c5a4292cf4
Base.copy(lasso::Lasso) = Lasso(lasso.λ, lasso.τ)

# ╔═╡ ec01437e-4df6-495d-a22d-cd04f14534b7
function prox_l1(w, η, λ)
	#### BEGIN SOLUTION

	if abs(w) <= η*λ
		return 0
	end

	return w > η*λ ? w - η*λ : w + η*λ

	#### END SOLUTION
end

# ╔═╡ 3d81c783-e101-441e-922f-b5a1aebd5b6f
function train!(lasso::Lasso, model::LinearModel, X, Y)
	n = size(X, 1) # number of samples
	λ = lasso.λ
	τ = lasso.τ # tolerance
	err = Inf
	c(x) = begin 
		e = (predict(model, x) - Y)
		dot(e, e) + λ*sum(abs.(model.W)) 
	end
	
	#### BEGIN SOLUTION
	
	XX = (1/n)*(X'*X)
	XY = (1/n)*(X'*Y)
	η = 1/(2*norm(XX))

	while abs.(c(X) - err) > τ
		err = c(X)
		model.W .= [prox_l1((model.W .- η*XX*model.W .+ η*XY)[i], η, λ) for i in 1:length(model.W)]
	end
	
	#### END SOLUTION
end

# ╔═╡ 3664982b-3543-418b-9b18-33d16dd25b05
md"""
## What about the KernelModel?

In this notebook, we are going to take a feature perspective on our kernel models to simplify the OLS algorithms. We will use `Lasso` and `Ridge` with a kernel representation as if it is just transforming the features through a non-linear function defined by our kernel representations, instead of deriving new algorithms through the dual representation. See the notes for more details.

When moving to gradient descent, this is how we will also use our kernel model.

"""

# ╔═╡ cab98efd-c83f-4cbf-955d-4fc33d122c90
function train!(ls, model::KernelModel, X, Y)
	# build K matrix
	K = get_features(model, X)
	train!(ls, model.model, K, Y)
end

# ╔═╡ d9935cc8-ec24-47e9-b39a-92c21377a161
struct MiniBatchGD
	n::Int
end

# ╔═╡ 5714c84f-1653-4c4a-a2e4-003d8560484a
md"""

Gradient descent is another strategy for learning weights of a model. Instead of creating a closed form solution (like OLS) we learn iteratively following the gradient of the loss/cost function. When our data needs to be represented in more complex forms, we often will use some variant of gradient descent to learn complex parameterizations. Gradient Descent also doesn't require the `XᵀX` to be invertable to find a solution. 

In this notebook we will be focusing on minibatch gradient descent, and using 3 learning rate adaptation rules `ConstantLR`, `RMSProp`, and `LineSearch`. All of these have their use in various parts of the literature and in various settings. 

Below you need to implement the function `epoch!` which goes through the data set in minibatches of size `mbgd.n`. Remember to randomize how you go through the data **and** that you are using the correct targets for the data passed to the learning update. In this implementation you will use 

```julia
update!(model, lossfunc, opt, X_batch, Y_batch)
```

to update your model. These functions are defined in the section on [optimizers](#opt).

"""

# ╔═╡ 9d96ede3-533e-42f7-ada1-6e71980bc6c2
function epoch!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y)
	epoch!(mbgd, get_linear_model(model), lossfunc, opt, get_features(lp.model, X), Y)
end

# ╔═╡ 6ff92fca-6d66-4f27-8e09-11a3887e66ba
function train!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y, num_epochs)
	train!(mbgd, get_linear_model(model), lossfunc, opt, get_features(model, X), Y, num_epochs)
end

# ╔═╡ 7e777dba-b389-4549-a93a-9b0394646c57
abstract type LossFunction end

# ╔═╡ 4f43373d-42ee-4269-9862-f53695351ea3
struct MSE <: LossFunction end

# ╔═╡ ada800ba-25e2-4544-a297-c42d8b36a9ff
function loss(lm::LinearModel, mse::MSE, X, Y)
	0.0
	#### BEGIN SOLUTION
	
	Y_hat = predict(lm, X)
	n = length(Y)
	
	loss = (1/(2*n)) * sum([(Y_hat[i] - Y[i])^2 for i in 1:n])

	#### END SOLUTION
end

# ╔═╡ 299116ea-66f3-4e52-ab0f-594249b9dd23
function gradient(lm::LinearModel, mse::MSE, X::Matrix, Y::Vector)
	∇W = zero(lm.W) # gradients should be the size of the weights
	
	#### BEGIN SOLUTION
	
	Y_hat = predict(lm, X)
	n = length(Y)
	
	∇W = (1/n) * ((Y_hat - Y)' * X)'
	
	#### END SOLUTION
	
	@assert size(∇W) == size(lm.W)
	∇W
end

# ╔═╡ 36c1f5c8-ac43-41ea-9100-8f85c1ee3708
abstract type Optimizer end

# ╔═╡ 159cecd9-de77-4586-9479-383661bf3397
begin
	struct _LR <: Optimizer end
	struct _LF <: LossFunction end
	function gradient(lm::LinearModel, lf::_LF, X::Matrix, Y::Vector)
		sum(X, dims=1)
	end
	function update!(lm::LinearModel, 
		 			 lf::_LF, 
		 			 opt::_LR, 
		 			 x::Matrix,
		 			 y::Vector)

		ΔW = gradient(lm, lf, x, y)[1, :]
		lm.W .-= ΔW
	end
end;

# ╔═╡ a3387a7e-436c-4724-aa29-92e78ea3a89f
begin
	# __check_mseGrad 
	__check_mseloss = loss(LinearModel(3, 1), MSE(), ones(4, 3), [1,2,3,4]) == 3.75
	__check_msegrad = all(gradient(LinearModel(3, 1), MSE(), ones(4, 3), [1,2,3,4]) .== -2.5)
	
	__check_MSE = __check_mseloss && __check_msegrad
	
md"""
For this notebook we will only be using MSE, but we still introduce the abstract type LossFunction for the future. Below you will need to implement the `loss` $(_check_complete(__check_mseloss)) function and the `gradient` $(_check_complete(__check_msegrad)) function for MSE.
	
Please use this scaled MSE for the loss and gradient:
```math
	MSE(\hat{Y}, Y) = \frac{1}{2n} \sum_{i=1}^n (\hat{Y_i} - Y_i)^2
```
"""
end

# ╔═╡ a17e5acd-d78d-4fab-9ab2-f01bd888339d
HTML("<h3 id=lossfunc> Loss Functions $(_check_complete(__check_MSE)) </h3>")

# ╔═╡ 0f6929b6-d869-4c3c-90f6-c923b265b164
struct ConstantLR <: Optimizer
	η::Float64
end

# ╔═╡ 8b8fd9b8-a41b-4fef-96b7-a146986c6f82
Base.copy(clr::ConstantLR) = ConstantLR(clr.η)

# ╔═╡ 344092df-c60b-4f8d-8992-cae088664632
function update!(lm::LinearModel, 
				 lf::LossFunction, 
				 opt::ConstantLR, 
				 x::Matrix,
				 y::Vector)

	g = gradient(lm, lf, x, y)
	
	#### BEGIN SOLUTION
	
	lm.W .-= opt.η * g

	#### END SOLUTION
end

# ╔═╡ 8ff69e30-811c-482e-833e-82357d258f95
begin
	mutable struct RMSProp <: Optimizer
		η::Float64 # step size
		ρ::Float64 # decay parameter
		v::Matrix{Float64} # exponential decaying average
		ϵ::Float64 #
	end
	
	RMSProp(η, ρ) = RMSProp(η, ρ, zeros(1, 1), 1e-5)
	RMSProp(η, ρ, lm::LinearModel) = RMSProp(η, ρ, zero(lm.W), 1e-5)
	RMSProp(η, ρ, lm::AbstractModel) = RMSProp(η, ρ, get_linear_model(model))
	Base.copy(rms::RMSProp) = RMSProp(rms.η, rms.ρ, zero(rms.v), rms.ϵ)
end

# ╔═╡ 11effb8a-b4f9-45c6-9b59-e8952fe40f3e
function update!(lm::LinearModel, 
				 lf::LossFunction,
				 opt::RMSProp,
				 x::Matrix,
				 y::Vector)

	g = gradient(lm, lf, x, y)
	if size(g) !== size(opt.v) # need to make sure this is of the right shape.
		opt.v = zero(g)
	end
	
	# update opt.v and lm.W
	η, ρ, v, ϵ = opt.η, opt.ρ, opt.v, opt.ϵ
	
	#### BEGIN SOLUTION
	
	opt.v = ρ*opt.v + (1-ρ)*g.^2
	lm.W .-= (η*g) ./ sqrt.(opt.v .+ ϵ)

	#### END SOLUTION
	
end

# ╔═╡ 0c8f46e8-1838-47fd-bc42-28df86ec6eb3
struct LineSearch <: Optimizer
	η_max::Float64
	τ::Float64
	ϵ::Float64
	max_iter::Int
end

# ╔═╡ 05a5b2eb-cc06-4d23-9a23-15bf6db484db
Base.copy(ls::LineSearch) = LineSearch(ls.η_max, ls.τ, ls.ϵ, ls.max_iter)

# ╔═╡ 03060875-d423-4448-bd81-9729161b7ea5
function update!(lm::LinearModel, 
				 lf::LossFunction, 
				 opt::LineSearch, 
				 X::Matrix, 
				 Y::Vector)
	
	g = gradient(lm, lf, X, Y)
	
	#### BEGIN SOLUTION
	
	η = opt.η_max
	init_loss = loss(lm, lf, X, Y)
	new_lm = copy(lm)
	
	improved_flag = false
	for _ in 1:opt.max_iter
		new_lm = copy(lm)
		new_lm.W .-= η*g
		if loss(new_lm, lf, X, Y) < init_loss - opt.ϵ
			improved_flag = true
			break
		end
		η *= opt.τ
		
	end
	
	improved_flag ? lm.W .= new_lm.W : nothing

	#### END SOLUTION
end

# ╔═╡ 69cf84e2-0aba-4595-8cb0-c082dbccdbe2
function epoch!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y)
	
	#### BEGIN SOLUTION
	
	#handles case when batch size larger than data present
	if size(X,1)÷mbgd.n == 0
		return update!(model, lossfunc, opt, X[1:size(X,1), :], Y[1:size(X,1)])
	end
	
	#deals with portion of data divisible by mbgd.n
	div_data_size = size(X,1)÷mbgd.n
	block_indices = Int(ceil(div_data_size))
	
	for index in randperm(block_indices)
		stop = index*mbgd.n
		start = index*mbgd.n - mbgd.n + 1
		
		update!(model, lossfunc, opt, X[start:stop, :], Y[start:stop])
	end
	

	#leftover data
	leftovers = size(X,1)%mbgd.n
	if leftovers != 0
		update!(model, lossfunc, opt, X[div_data_size+1:div_data_size+1 + leftovers, :], Y[div_data_size+1:div_data_size+1 + leftovers])
	end
		
	#### END SOLUTION
end

# ╔═╡ acf1b36c-0412-452c-ab4d-a388f84fd1fb
begin
	__check_MBGD = let

		lm = LinearModel(3, 1)
		opt = _LR()
		lf = _LF()
		X = ones(10, 3)
		Y = collect(0.0:0.1:0.9)
		mbgd = MiniBatchGD(5)
		epoch!(mbgd, lm, lf, opt, X, Y)
		all(lm.W .== -10.0)
	end
	str = "<h2 id=graddescent> Gradient Descent $(_check_complete(__check_MBGD)) </h2>"
	HTML(str)
end

# ╔═╡ 2782903e-1d2e-47de-9109-acff4595de42
function train!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y, num_epochs)
	ℒ = zeros(num_epochs + 1)
	ℒ[1] = loss(model, lossfunc, X, Y)
	for i in 1:num_epochs
		epoch!(mbgd, model, lossfunc, opt, X, Y)
		ℒ[i+1] = loss(model, lossfunc, X, Y)
	end
	ℒ
end

# ╔═╡ 7dcfc207-0e32-42cf-a9cc-fe9bbef7e92a
begin

	__prox_complete = (prox_l1(0.1, 1.0, 0.15) == 0.0) &&
					  (prox_l1(1.0, 0.1, 0.15) == 0.985) &&
					  (prox_l1(-1.3, 0.03, 0.2) == -1.294)
	
	# __lasso_train_complete = false
	__lasso_train_complete = let
		rng = Random.MersenneTwister(2)
		ols = Lasso(0.1, 0.01)
		X = rand(rng, 1000, 6)
		W = rand(rng, 6, 1)
		Y = X*W .+ randn(rng, 1000)*0.1
		m = LinearModel(6, 1)
	
		train!(ols, m, X, Y)
		mean((X*m.W .- Y).^2) ≈ 0.04866648559755483
	end

	__lasso_complete = __prox_complete && __lasso_train_complete
	


	md"""

	Lasso regression corresponds to the l1 regularized problem with cost function

	```math
	c(X) = \lVert X^\top \mathbf{w} - Y \rVert_2^2 + \lambda \lVert \mathbf{w} \rVert_1
	```

	Because this is non-differentiable when $$\mathbf{w} = \mathbf{0}$$ there is no closed form solution. So we solve iteratively, making the lasso regressor. See your notes for the algorithm. You will need to fill in two functions for this portion:
	- `prox_l1` $(__prox_complete ? "✅" : "❌"): the proximal operator taking a weight, a stepsize, and the regularization parameter
	- `train!` $(__lasso_train_complete ? "✅" : "❌"): which does lasso using `prox_l1`
	"""
end

# ╔═╡ 257c33a7-da35-4d88-b6fb-43b14c00bdc5
HTML("<h3 id=lasso> Lasso Regression $(_check_complete(__lasso_complete)) </h3>")

# ╔═╡ eb5d3e74-f156-43a1-9966-4880f80a3d60
begin
	_check_ConstantLR = let
		lm = LinearModel(3, 1)
		opt = ConstantLR(0.1)
		lf = MSE()
		X = ones(4, 3)
		Y = [0.1, 0.2, 0.3, 0.4]
		update!(lm, lf, opt, X, Y)
		all(lm.W .== 0.025)
	end
	md"""
	#### Constant Learning Rate $(_check_complete(_check_ConstantLR))

	`ConstantLR` updates the weights using a constant learning rate `η`
	
	```math
	W = W - η*g
	```
	
	where `g` is the gradient defined by the loss function.
	"""
end

# ╔═╡ e5b9d8ab-6929-4de5-b197-67c1e93cd472
begin
	 __check_RMSProp_v, __check_RMSProp_W = let
		lm = LinearModel(2, 1)
		opt = RMSProp(0.1, 0.9, lm)
		X = [0.1 0.5; 
			 0.5 0.0; 
			 1.0 0.2]
		Y = [1, 2, 3]
		update!(lm, MSE(), opt, X, Y)
		true_v = [0.18677777777777768, 0.013444444444444445]
		true_W = [0.31621930100905377, 0.3161102262149725]
		all(opt.v .≈ true_v), all(lm.W .≈ true_W)
	end
	
	__check_RMSProp = __check_RMSProp_v && __check_RMSProp_W
	
md"""
#### RMSProp $(_check_complete(__check_RMSProp))

RMSProp is a first-order adaptive stepsize optimizer which was proposed by Geoff Hinton [in this lecture](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). You will be implementing the `update!` function, specifically the updates to the exponentially decaying average of the squared gradients (i.e. `v`) $(_check_complete(__check_RMSProp_v)), and the updates to the model weights $(_check_complete(__check_RMSProp_W)). In this implementation you can assume you will always recieve a linear model, and the gradients are provided for you in `∇W`. You can find another useful resource for this (and many other optimizers) [here](https://ruder.io/optimizing-gradient-descent/index.html).
	
The implementation updates the exponentially weighted sum of squared gradients `v` to normalize the gradient. This is very effective in practice and used primarily deep learning settings (we will discuss this in the future). The update equations are as follows
	
```math
\begin{align*}
v_i = \rho * v_i + (1-\rho) g_i^2 \\	
W_i = W_i - \frac{\eta}{\sqrt{v_i + \epsilon}} g_i	
\end{align*}
```
	
"""
end

# ╔═╡ 4479273c-e94e-4630-94e8-881ab18d8642
begin
	
	__check_LineSearch = let
		lm = LinearModel(2, 1)
		opt = LineSearch(1.0, 0.7, 1e-4, 50)
		rng = Random.MersenneTwister(10)
		X = rand(rng, 1000, 2)
		Y = X*rand(rng, 2) + 0.01*randn(rng, 1000)
		for _ in 1:30
			idx = randperm(rng, size(X, 1))[1:30]
			update!(lm, MSE(), opt, X[idx,:], Y[idx])
		end
		true_w =  [0.6412283969853311; 0.05523836697813841]
		all(lm.W .≈ true_w)
	end
	
	
	md"""
	#### Line Search $(_check_complete(__check_LineSearch))
	
	Line search is another algorithm to select learning rates. In this algorithm we iteratively try a sequence of learning rates based on a starting learning `η_max` and an exponential decay paramter `τ`. With starting `η = η_max` we create a new solution
	
	```julia
	new_lm = copy(lm)
	new_lm.W .= new_lm.W - η*g
	```
	
	We then try `max_iter` number of iterations checking if the new solution is better than the previous solution:

	```julia
	loss(new_lm, lf, X, Y) < loss(lm, lf, X, Y) - opt.ϵ
	```	
	
	If it is we use the new learning rate to update the original model. If not we decay the learning rate parameters `η = η*τ` and try again. If we reach the maximum number of parameters we don't update the original model.


	"""
end

# ╔═╡ af8acfdf-32bd-43c1-82d0-99008ee4cb3e
HTML("<h3 id=opt> Optimizers $(_check_complete(_check_ConstantLR && __check_RMSProp &&__check_LineSearch)) </h3>")

# ╔═╡ 3738f45d-38e5-415f-a4e6-f8922df84d09
md"""
Below you will need to implement three optimizers

- Constant learning rate $(_check_complete(_check_ConstantLR))
- RMSProp $(_check_complete(__check_RMSProp))
- LineSearch $(_check_complete(__check_LineSearch)).
"""

# ╔═╡ 7d108fb6-3a4e-4158-b416-d39f3ee43a54
html"<h1 id=prototypes>Prototype Selection for Kernels</h1>"

# ╔═╡ 3a428568-9991-449d-bae8-42395cdaa80e
md"""
Now that we have some learning methods defined, we can take a look at a prototype selection strategy for our kernel models. The first is a naive prototype selection strategy, the second takes advantage of the sparsity induced by Lasso regression to select prototypes. For both methods we will be using the actual data as templates for our prototypes

"""

# ╔═╡ 13c83a66-46c0-4469-bd0c-77e7ec2fa459
abstract type PrototypeSelectionStrategy end

# ╔═╡ 72b573a4-b426-48f0-937c-7bdf841d9687
md"""
### Random prototypes from data

This algorithm selects a random set of `n` prototypes from the data. This is surprisingly effective as you can see in the experiments below.
"""

# ╔═╡ ccbc151e-c24f-4588-9420-7dd527887daf
struct RandomPrototypes <: PrototypeSelectionStrategy 
	n::Int
end

# ╔═╡ f9070a63-d154-47be-afd9-d45f13bbfc17
function select_prototypes(css::RandomPrototypes, kern_func, X, Y)
	prototype_idx = randperm(size(X, 1))[1:css.n]
	[X[i,:] for i in prototype_idx]
end

# ╔═╡ bb39c8ff-dfec-47e0-9508-bec9068ca827
struct L1Prototypes <: PrototypeSelectionStrategy
	λ::Float64
	τ::Float64
	n::Int
	start_n::Int
end

# ╔═╡ 81f5a184-84ab-4868-9ecb-73fd17e4dd6b
function maxk_idx(a, k)
	b = partialsortperm(a, 1:k, rev=true)
	return b
end

# ╔═╡ 1558fb11-49d7-4931-b4fb-c447b03193c9
function select_prototypes(css::L1Prototypes, kern_func, X, Y)
	
	#### BEGIN SOLUTION
	
	prototype_idx = randperm(size(X, 1))[1:css.start_n]
	prototypes = [X[i,:] for i in prototype_idx]
	
	km = KernelModel(kern_func, prototypes)
	lasso = Lasso(css.λ, css.τ)
	train!(lasso, km, X, Y)
	
	prototype_indices = maxk_idx(vec(km.model.W), css.n)
	
	[prototypes[prototype_indices[i]] for i in 1:css.n]
	
	#### END SOLUTION
	
end

# ╔═╡ 69e10195-fca8-47e0-9136-7a80b2d44b76
function update_transform!(km::KernelModel, X::AbstractMatrix, Y::AbstractVector)
	# if nothing just use the prototypes that already exist.
	css = km.prototype_selection_strategy
	if isnothing(css)
		@assert length(km.prototypes) > 0
	else
		prototypes = select_prototypes(css, km.kern, X, Y)
		km.prototypes = prototypes
		km.model = LinearModel(length(prototypes), size(km.model.W, 2))
	end
	
end

# ╔═╡ a253f0f3-5e77-4adc-985e-96aa7891e954
begin
	
	__check_L1_CSS = let
		rng = Random.MersenneTwister(2)
		km = KernelModel(RBF(0.9), L1Prototypes(0.1, 0.001, 4, 10))
		X = rand(rng, 10, 4)
		m = (x)-> sin(π*x[1]) + sin(π*x[2]) + sin(π*(x[3] + x[4])/2)
		Y = [m(x) for x in eachrow(X)]
		try
			update_transform!(km, X, Y)
			length(km.prototypes) == 4
		catch
			false
		end
	end
	
	
	md"""
	### L1 Strategy for selecting prototypes $(_check_complete(__check_L1_CSS))
	
	In this section we will make a novel ceneter selection algorithm based on using L1 regularization. You will need to implement `select_prototypes` which returns an array of prototypes (see `RandomPrototypes` for an example). These prototypes will be slected based on our implementation of `Lasso`. You will use `Lasso` with parameters `L1Prototypes.λ` and `L1Prototypes.τ`, and do a regression on a random set of prototypes `start_n`. Once you do this regression you will use `maxk_idx` to return the top `k=n` prototypes based off the absolute values of the model weights.
	"""
end

# ╔═╡ fa610de0-f8c7-4c48-88d8-f5398ea75ae2
md"""
# Evaluating models

In the following section, we provide a few helper functions and structs to make evaluating methods straightforward. The abstract type `LearningProblem` with children `GDLearningProblem` and `OLSLearningProblem` are used to construct a learning problem. You will notice these structs contain all the information needed to `train!` a model for both gradient descent and for OLS. We also provide the `run` and `run!` functions. These will update the transform according to the provided data and train the model. `run` does this with a copy of the learning problem, while `run!` does this inplace. 

"""

# ╔═╡ d695b118-6d0d-401d-990f-85ba467cc53e
abstract type LearningProblem end

# ╔═╡ 6edc243e-59ac-4c6f-b507-80d3ec13bc21
"""
	GDLearningProblem

This is a struct for keeping a the necessary gradient descent learning setting components together.
"""
struct GDLearningProblem{M<:AbstractModel, O<:Optimizer, LF<:LossFunction} <: LearningProblem
	gd::MiniBatchGD
	model::M
	opt::O
	loss::LF
end

# ╔═╡ 3bdde6cf-3b68-46d3-bf76-d68c20b661e9
Base.copy(lp::GDLearningProblem) = 
	GDLearningProblem(lp.gd, copy(lp.model), copy(lp.opt), lp.loss)

# ╔═╡ 77fc32cf-4b3d-4320-a7d3-8c70df73811c
"""
	OLSLearningProblem

This is a struct for keeping a the necessary OLS learning setting components together.
"""
struct OLSLearningProblem{R<:Union{Ridge, Lasso}, LM} <: LearningProblem
	regression::R
	model::LM
end

# ╔═╡ 6196c8e4-8f80-4b16-bf30-0f0b828c41de
Base.copy(lp::OLSLearningProblem) = 
	OLSLearningProblem(copy(lp.regression), copy(lp.model))

# ╔═╡ 7905f581-1593-4e06-8aaf-faec05c3b306
function run!(lp::GDLearningProblem, X, Y, num_epochs)
	update_transform!(lp.model, X, Y)
	train!(lp.gd, lp.model, lp.loss, lp.opt, X, Y, num_epochs)
end

# ╔═╡ 4cb0e4e1-99b3-41fd-b1d8-d19e7103cafe
function run!(lp::OLSLearningProblem, X, Y, args...)
	update_transform!(lp.model, X, Y)
	train!(lp.regression, lp.model, X, Y)
	sqrt(mean(abs2, predict(lp.model, X) - Y))
end

# ╔═╡ 69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
function run(lp::LearningProblem, args...)
	cp_lp = copy(lp)
	ℒ = run!(cp_lp, args...)
	return cp_lp, ℒ
end

# ╔═╡ db416236-b920-42a0-b5c6-e472d2a3c224
function random_dataset_split(X, Y, n_train)
	#### BEGIN SOLUTION
	
	rand_idx = randperm(length(Y))
	
	X_train = X[rand_idx[1:n_train],:]
	X_test = X[rand_idx[n_train+1:length(Y)],:]
	
	Y_train = Y[rand_idx[1:n_train]]
	Y_test = Y[rand_idx[n_train+1:length(Y)]]
	
	return (X_train, Y_train), (X_test, Y_test)

	#### END SOLUTION
	
	# return two tuples (X_train, Y_train), (X_test, Y_test)
end

# ╔═╡ d339a276-296a-4378-82ae-fe498e9b5181
"""
	cross_validation(lp, X, Y, num_epochs, k; train_size)

Using `train!` do RSS cross validation with k, train_size, and num_epochs. This will create a copy of the learning problem and use this new copy to train. It will return the estimate of the error with confidence bounds.
"""
function cross_validation(lp::LearningProblem, 
						  X, 
						  Y, 
						  num_epochs,
						  k;
						  train_size=Int(floor(size(X, 1)*0.9)))

	cv_err = zeros(k)
	# problems = Vector{typeof(lp)}()
	
	for i in 1:k
		# Use random_dataset_split to split data
		# train
		# test (assume MSE)
		
		(X_train, Y_train), (X_test, Y_test) = random_dataset_split(X,Y,train_size)
		
		lp, L = run(lp, X_train, Y_train, num_epochs)
		
		err = predict(lp.model, X_test) - Y_test
		MSE_test = dot(err,err)
		cv_err[i] = MSE_test

		#### END SOLUTION
	end

	cv_err
end

# ╔═╡ fd75ff49-b5de-48dc-ae89-06bf855d81b2
begin
	
	__check_random_data_split = let
		rng = Random.MersenneTwister(10)
		X, Y = rand(rng, 10, 3), rand(rng, 10)
		data = random_dataset_split(X, Y, 8)
		if isnothing(data)
			false
		else
			d_train = data[1]
			d_test = data[2]
			(size(d_train[1]) == (8, 3)) && 
			(size(d_train[2]) == (8,)) &&
			(size(d_test[1]) == (2, 3)) && 
			(size(d_test[2]) == (2,))
		end
	end
	__check_cross_validation = let
		Random.seed!(2)
		num_epochs = 100
		X = rand(5000, 10)
		X[:, 2] .= 0.1*X[:, 3]
		X[:, 5] .= 0.5*(X[:, 6] + X[:, 7])
		w = rand(10)
		Y = (X*w) + randn(5000)*0.001

		lp = GDLearningProblem(
			MiniBatchGD(30),
			LinearModel(size(X, 2), 1),
			ConstantLR(0.01),
			MSE())

		cv_err = cross_validation(lp, X, Y, num_epochs, 10)
		all(cv_err .!= 0.0)

	end
	
	__check_CV = __check_random_data_split && __check_cross_validation
	
	md"""

	In this section you will implement two functions to implement cross validation with random data splits. 
	
	`random_dataset_split` $(_check_complete(__check_random_data_split)): randomly splits the data `X` and `Y` into a training set and a validation set. This will also be used to split out data into a training set and test set. This function returns two tuples `(X_train, Y_train), (X_test, Y_test)`.
	
	`cross_validation` $(_check_complete(__check_cross_validation)): This does `k` independent experiments of the passed LearningProblem. The function trains the model and then stores the error according to the Root Mean Squared Error (no matter what the loss is in the learning problem), storing this in `cv_err`. `cv_err` is then returned.

	Because these functions require randomness, the check marks which have been used to check correctness before only check for returning the correct datatypes (for `random_dataset_split`) or that the returned vector of numbers `cv_err` is non-zero. So getting checks here **does not mean** you will get full points.
	
	"""
end

# ╔═╡ a9d3c6c3-1cb7-4417-ba6a-54998841c87d
let
	q1_a_check = _check_complete(__kernel_regression_check_1)
	q1_b_check = _check_complete(__check_L1_CSS)
	q2_check = _check_complete(__lasso_complete)
	q3_a_check = _check_complete(__check_RMSProp)
	q3_b_check = _check_complete(__check_LineSearch)
	q3_c_check = _check_complete(_check_ConstantLR)
	q4_check = _check_complete(__check_CV)
	
	q5_check = _check_complete(__check_MBGD)
	q5_b_check = _check_complete(__check_MSE)

md"""
# Preamble 

In this assignment, we will be:
- implementing a model which uses kernels as similarity features [KernelModel](#kernel_regression) $(q1_a_check) and two [prototype selection strategies](#prototypes) $(q1_b_check),
- reviewing [OLS and Ridge Regression](#ols_ridge),
- implementing [Lasso regression](#lasso) $(q2_check),
- implementing [minibatch gradient descent](#graddescent) $(q5_check) and [MSE](#lossfunc) $(q5_b_check)
- implementing several [learning rate strategies](#opt): ConstantLR $(q3_c_check), RMSProp $(q3_a_check), and LineSearch $(q3_b_check)
- implementing [cross validation](#cv) with random data splitting $(q4_check)

"""
end

# ╔═╡ eef918a9-b8af-4d41-85b1-bebf1c7889cc
HTML("<h2 id=cv> Cross Validation $(_check_complete(__check_CV)) </h2>")

# ╔═╡ 58e626f1-32fb-465a-839e-1f413411c6f3
md"
# Experiments

In this section, we will run three experiments on the different algorithms we implemented above. We provide the data in the `Data` section, and then follow with the three experiments and their descriptions. You will need to analyze and understand the three experiments for the written portion of this assignment.
"

# ╔═╡ 5ec88a5a-71e2-40c1-9913-98ced174341a
md"""
## Data

This section creates the datasets we will use in our comparisons. Feel free to play with them in `let` blocks.
"""

# ╔═╡ 12d0a092-e5d5-4aa1-ac82-5d262f54aa6d
"""
	splitdataframe(split_to_X_Y::Function, df::DataFrame, test_perc; shuffle = false)
	splitdataframe(df::DataFrame, test_perc; shuffle = false)

Splits a dataframe into test and train sets. Optionally takes a function as the first parameter to split the dataframe into X and Y components for training. This defaults to the `identity` function.
"""
function splitdataframe(split_to_X_Y::Function, df::DataFrame, test_perc; 
		                shuffle = false)
	#= shuffle dataframe. 
	This is innefficient as it makes an entire new dataframe, 
	but fine for the small dataset we have in this notebook. 
	Consider shuffling inplace before calling this function.
	=#
	
	df_shuffle = if shuffle == true
		df[randperm(nrow(df)), :]
	else
		df
	end
	
	# Get train size with percentage of test data.
	train_size = Int(round(size(df,1) * (1 - test_perc)))
	
	dftrain = df_shuffle[1:train_size, :]
	dftest = df_shuffle[(train_size+1):end, :]
	
	split_to_X_Y(dftrain), split_to_X_Y(dftest)
end

# ╔═╡ d2c516c0-f5e5-4476-b7d6-89862f6f2472
function unit_normalize_columns!(df::DataFrame)
	for name in names(df)
		mn, mx = minimum(df[!, name]), maximum(df[!, name])
		df[!, name] .= (df[!, name] .- mn) ./ (mx - mn)
	end
	df
end

# ╔═╡ 72641129-5274-47b6-9967-fa37c8036552
md"""
### [Boston Housing dataset](https://www.kaggle.com/vikrishnan/boston-house-prices)
"""

# ╔═╡ 90f34d85-3fdc-4e2a-ada4-085154103c6b
housing_data = let
	names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
	data = CSV.read("data/housing.csv", DataFrame, header=names, delim=' ', ignorerepeated=true)
	unit_normalize_columns!(data)
end;

# ╔═╡ 00c94b56-d39d-4289-9a61-1f07dee2cd57
md"
### [Susy dataset](https://archive.ics.uci.edu/ml/datasets/SUSY)
"

# ╔═╡ 10cc73dc-f07d-4674-8aff-e7025be20ccb
susy_dataset = let
	data = CSV.read("data/susysubset.csv", DataFrame, header=false)
	unit_normalize_columns!(data)
end;

# ╔═╡ 788e10bf-fb30-43f3-bbd3-aafd59467636
md"### Highly Correlated dataset"

# ╔═╡ 81e13ad0-53c6-42e1-8e0f-8345d6f42cc6
md"""
This dataset is similar to the examples found in the original 
[lasso paper](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.2517-6161.1996.tb02080.x). This dataset simulates heavily correlated data. Below we will use this to test `OLS`, `Ridge`, and `Lasso`.
"""

# ╔═╡ 3218558d-01e2-4dba-a90a-cc64f2952869
function simulate_dataset(ρ, N)
	s = 8
	Σ = reshape([ρ^abs(i - j) for i in 1:s for j in 1:s], s, :)
	mv_g = Distributions.MultivariateNormal(zeros(s), Σ)
	β = [1, 0, 0, 1.5, 1, 0, 0, 1]
	σ = 3
	X = collect(transpose(rand(mv_g, N)))
	Y = X*β + σ*randn(N)
	X, Y
end

# ╔═╡ 14b329fb-8053-4148-8d24-4458e592e7e3
md"""
## Plotting our data

The `plot_data` function produces two plots that can be displayed horizontally or vertically. The left or top plot is a box plot over the cv errors, the right or bottom plot is a bar graph displaying average cv errors with standard error bars. This function will be used for all the experiments, and you should use this to finish your written experiments.

"""


# ╔═╡ eebf5285-2336-4c07-a4fd-b1fd841dee52
function plot_data(algs, errs; vert=false)
	stderr(x) = sqrt(var(x)/length(x))
	
	plt1 = boxplot(reshape(algs, 1, :),
				   errs,
				   legend=false, ylabel="MSE",
				   pallette=:seaborn_colorblind)
	
	plt2 = bar(reshape(algs, 1, :),
			   reshape(mean.(errs), 1, :),
			   yerr=reshape(stderr.(errs), 1, :),
			   legend=false,
			   pallette=:seaborn_colorblind,
			   ylabel=vert ? "MSE" : "")
	
	if vert
		plot(plt1, plt2, layout=(2, 1), size=(600, 600))
	else
		plot(plt1, plt2)
	end
end

# ╔═╡ 9c345c36-6519-4bc6-8947-df88885bceb0
md"""
## OLS, Ridge, and Lasso

We will compare OLS, Ridge, and Lasso on simulated data mentioned above.

To run the experiment turn on the checkbox $(@bind __run_ols PlutoUI.CheckBox())
"""

# ╔═╡ 0187a663-34aa-4464-8504-c969d2d9ec11
md"""
Below we use two plot types to compare the cross validation error with `k=3000` and the number of training samples `N=20`.
"""

# ╔═╡ c8577196-e64f-4cfd-9c9d-bc0604fd1f8c
ols_settings = Dict(
	"ρ"=>0.7,
	"k"=>3000, 
	"N"=>20)

# ╔═╡ aa012d54-38b7-411f-ab89-35fb4a074d39
let
	if __run_ols
		ρ, N, k = ols_settings["ρ"], ols_settings["N"], ols_settings["k"]
		algs_ols = ["OLS", "Ridge", "Lasso"]
		ols_problems = [OLSLearningProblem(
			OLS(),
				LinearModel(8, 1)),
			OLSLearningProblem(
				Ridge(0.5),
				LinearModel(8, 1)),
			OLSLearningProblem(
				Lasso(0.5, 0.1),
				LinearModel(8, 1))
		]
		ols_errs = let
			Random.seed!(5)
			errs = Vector{Float64}[]
			# for r in 1:100
				sim_X, sim_Y = simulate_dataset(ρ, 30)
				err = zeros(length(ols_problems))
				for (idx, prblms) in enumerate(ols_problems)
					cv_err = cross_validation(prblms, sim_X, sim_Y, 1, k; train_size=N)
					push!(errs, cv_err)
				end
			errs
		end
		plot_data(algs_ols, ols_errs)
	end
end

# ╔═╡ b689d666-37da-40f7-adb8-44aa2b9f5139
md"""
## Non-linear feature transforms

We will compare the linear to non-linear models using the [boston housing dataset](https://www.kaggle.com/vikrishnan/boston-house-prices) and a simulated data set.

To run these experiments use $(@bind __run_nonlinear PlutoUI.CheckBox())
"""

# ╔═╡ 55ce32ff-dec3-4bd4-b6a2-95483e7637e9
md"""
This first expereiment uses a simulated training set which aims to predict this function

```julia
f(x) = sin(π*x[1]*x[2]^2) + cos(π*x[3]^3) + x[5]*sin(π*x[4]^4) + 0.001*randn()
```

from inputs $\mathbf{x} \in [0.0, 1.0]^5$. We compare a linear representation, cosine similarity with random prototypes, RBF kernels with random prototypes, and RBF kernels with `L1CSS`. We use cross validation with `k=50` and `N=400`.
"""

# ╔═╡ d381d944-5069-4f16-8194-bd49eb2fe1cd
let
	if __run_nonlinear
		algs = ["Linear", "Cos-RandPrototypes", "RBF-RandPrototypes", "RBF-L1Prototypes"]
		non_linear_problems_sin = [
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(5, 1),
				ConstantLR(0.01),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				KernelModel(cosine_similarity(), RandomPrototypes(40)),
				ConstantLR(0.01),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				KernelModel(RBF(1.0), RandomPrototypes(40)),
				ConstantLR(0.01),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				KernelModel(RBF(1.0), L1Prototypes(0.1, 1e-1, 40, 250)),
				ConstantLR(0.01),
				MSE())
			];
		nonlinear_errs_sin = let
			X = rand(500, 5)
			f(x) = sin(π*x[1]*x[2]^2) + cos(π*x[3]^3) + x[5]*sin(π*x[4]^4) + 0.001*randn()
			Y = [f(x) for x in eachrow(X)]
			Y .= (Y.-minimum(Y))/(maximum(Y) - minimum(Y))
			plot(Y)
			errs = Vector{Float64}[]
			for (idx, prblms) in enumerate(non_linear_problems_sin)
				cv_err = cross_validation(prblms, X, Y, 1, 50; train_size=400)
				push!(errs, cv_err)
			end
			errs
		end
		plot_data(algs, nonlinear_errs_sin, vert=true)
	end
end

# ╔═╡ 80406819-83d2-4625-8ed3-959c127e3e2c
md"""
The following experiment uses the [boston housing dataset](https://www.kaggle.com/vikrishnan/boston-house-prices) and aims to predict housing prices from a set of features. Here we use `k=50` and `N=450`.
"""

# ╔═╡ 5a4ac4ed-6ed5-4d28-981e-8bf7b6b8889d
let
	if __run_nonlinear
		algs = ["Linear", "Cos-Rand", "RBF-Rand", "RBF-L1"]
		non_linear_problems = [
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(13, 1),
				ConstantLR(0.01),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				KernelModel(cosine_similarity(), RandomPrototypes(10)),
				ConstantLR(0.01),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				KernelModel(RBF(3.0), RandomPrototypes(10)),
				ConstantLR(0.01),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				KernelModel(RBF(3.0), L1Prototypes(0.1, 1e-3, 10, 250)),
				ConstantLR(0.01),
				MSE())
		]
		nonlinear_errs = let

			Random.seed!(2)
			data = (X=Matrix(housing_data[:, 1:end-1]), Y=housing_data[:, end])
			errs = Vector{Float64}[]
			for (idx, prblms) in enumerate(non_linear_problems)
				cv_err = cross_validation(
					prblms, data.X, data.Y, 1, 50; train_size=450)
				push!(errs, cv_err)
			end
			errs
		end
		
		plot_data(algs, nonlinear_errs; vert=true)
	end
end

# ╔═╡ 0903dd95-5525-44e5-891d-acbe2fb2190f
md"""
## Learning Rate adapatation

We will compare the different learning rate algorithms on a subset of the [Susy dataset](https://archive.ics.uci.edu/ml/datasets/SUSY). We will be predicting the first component.

To run this experiment click $(@bind __run_lra PlutoUI.CheckBox())
"""

# ╔═╡ c6f5e697-b72a-441e-9a05-e47a09bee2f7
md"""
In this experiment we compare constant learning rates, RMSProp, and LineSearch with cross validation `k=10` and `N=95000`.
"""

# ╔═╡ c01ff616-e570-4013-a0b2-d97fcda6f279
let
	if __run_lra
		algs_lr = ["ConstantLR", "RMSProp", "LineSearch"]
		lr_adapt_problems = [
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(8, 1),
				ConstantLR(0.01),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(8, 1),
				RMSProp(1e-3, 0.9),
				MSE()),
			GDLearningProblem(
				MiniBatchGD(30),
				LinearModel(8, 1),
				LineSearch(1.0, 0.9, 1e-5, 20),
				MSE()),
		];
		lr_errs = let
			# train_data, test_data = splitdataframe(housing_data, 0.1; shuffle=true) do df
			# 	(X = Matrix(df[:, 1:end-1]), Y = df[:, end])
			# end
			Random.seed!(2)
			test_idx = 1
			train_idx = filter(x->x!=test_idx, 1:9)
			data = (X=Matrix(susy_dataset[:, train_idx]), Y=susy_dataset[:, test_idx])
			errs = Vector{Float64}[]
			for (idx, prblms) in enumerate(lr_adapt_problems)
				cv_err = cross_validation(prblms, data.X, data.Y, 1, 10; train_size=95000)
				push!(errs, cv_err)
			end
			errs
		end
		
		plot_data(algs_lr, lr_errs)
	end

end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.9.1"
DataFrames = "~1.2.2"
Distributions = "~0.25.16"
MultivariateStats = "~0.8.0"
PlutoUI = "~0.7.9"
StatsBase = "~0.33.10"
StatsPlots = "~0.14.26"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

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

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "c907e91e253751f5840135f4c9deb1308273338d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.1"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "30ee06de5ff870b45c78f529a6b093b3323256a3"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.1"

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
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

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

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "6071cb87be6a444ac75fdbf51b8e7273808ce62f"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.35.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

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
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

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

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

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
git-tree-sha1 = "f985af3b9f4e278b1d24434cbb546d6092fca661"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.3"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "0f5e8d0cb91a6386ba47bd1527b240bd5725fbae"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.10"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "a3b7b041753094f3b17ffa9d2e2e07d8cace09cd"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.3"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

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

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

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
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

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

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

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

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

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
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

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
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "86197a8ecb06e222d66797b0c2d2f0cc7b69e42b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.2"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

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

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

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
git-tree-sha1 = "c870a0d713b51e4b49be6432eff0e26a4325afee"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.6"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

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
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

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
git-tree-sha1 = "9ff1c70190c1c30aebca35dc489f7411b256cd23"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.13"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "2dbafeadadcf7dadff20cd60046bba416b4912be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.21.3"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

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
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "7dff99fbc740e2f8228c6878e2aad6d7c2678098"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.1"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "1f27772b89958deed68d2709e5f08a5e5f59a5af"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.7"

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
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

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
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e7d1e79232310bd654c7cef46465c537562af4fe"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.26"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "1700b86ad59348c0f9f68ddc95117071f947072d"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.1"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "019acfd5a4a6c5f0f38de69f2ff7ed527f1881da"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.1.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "368d04a820fe069f9080ff1b432147a6203c3c89"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

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

[[WeakRefStrings]]
deps = ["DataAPI", "Parsers"]
git-tree-sha1 = "4a4cfb1ae5f26202db4f0320ac9344b3372136b0"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.3.0"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "eae2fbbc34a79ffd57fb4c972b08ce50b8f6a00d"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.3"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

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

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

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
# ╟─75441ce6-2137-4fcf-bba2-6ed67b9acb59
# ╠═dc004086-db6e-4813-841e-d427520402f7
# ╠═c9797979-7d16-4ba4-8f82-c7ec33f6e517
# ╟─693a3933-c1c2-4249-8c03-f5151267222f
# ╟─bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
# ╠═def97306-1703-42bc-bc09-da623c545e87
# ╟─a9d3c6c3-1cb7-4417-ba6a-54998841c87d
# ╟─7b513344-1cad-4eef-9faf-e77ba176323e
# ╟─4f4029a2-c590-4bd3-a0db-d2380d4b4620
# ╠═dcfecc35-f25c-4856-8429-5c31d94d0a42
# ╠═d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
# ╠═7cd46d84-a74c-44dc-8339-68010924bc39
# ╟─8745fec1-47c8-428b-9ea4-1e6828618830
# ╠═2d43a3ba-2a2c-4114-882f-5834d42e302a
# ╟─401bf2ad-53b9-44e7-be90-41f749d98e66
# ╠═8044511f-77e4-4d77-b2a4-a04c24eab53a
# ╠═f4dc2bf9-d1bd-4683-9306-8b77de643266
# ╟─84d3240a-4350-43a5-98bf-5dd8daf2586b
# ╠═cdbe526a-d0c1-45f8-9935-6de78c19304c
# ╟─fd8a21a8-3683-4691-af38-48178c805332
# ╠═157e00c4-9154-4e2d-84cc-0c73ea25549d
# ╟─8171b686-c64a-4e9f-b6a9-14b08e1e559c
# ╠═69e10195-fca8-47e0-9136-7a80b2d44b76
# ╟─15ff9bb7-bff4-4cf4-830c-79197a6c1c21
# ╟─1ba63749-06cb-4f84-a018-ae3c4aa5992f
# ╠═d5fa8c7e-5baf-4dd3-9d39-98b3ab6fdf7e
# ╟─b1aa4dff-b2f2-4151-9e02-46199674a956
# ╠═2ac36388-1789-4133-8dd0-1797fd19899e
# ╟─69ee4c2d-b133-4b82-b4d1-48988d26d184
# ╟─3e04ec5e-393a-4031-9507-7e3339118ddc
# ╟─d1c5c8c8-1f29-4a60-b362-ae3d300413cd
# ╟─ae98fbde-b1c4-46b6-aca3-a222acf20bc8
# ╠═bf05c0a2-de24-4a44-841c-8a7f4b40f8b8
# ╠═68386c6b-d025-4b84-8afb-6879d74f74a7
# ╠═4c4ad7c7-6e81-49c1-92cd-1d133cd2c905
# ╠═57931788-0280-4c73-b7e9-8e2c581fca9e
# ╟─257c33a7-da35-4d88-b6fb-43b14c00bdc5
# ╟─7dcfc207-0e32-42cf-a9cc-fe9bbef7e92a
# ╠═350a0c24-fb1e-4b62-b9e9-26bba9ac35f3
# ╠═9aabea45-d91d-475f-93ed-a5c5a4292cf4
# ╠═ec01437e-4df6-495d-a22d-cd04f14534b7
# ╠═3d81c783-e101-441e-922f-b5a1aebd5b6f
# ╟─3664982b-3543-418b-9b18-33d16dd25b05
# ╠═cab98efd-c83f-4cbf-955d-4fc33d122c90
# ╠═acf1b36c-0412-452c-ab4d-a388f84fd1fb
# ╟─159cecd9-de77-4586-9479-383661bf3397
# ╠═d9935cc8-ec24-47e9-b39a-92c21377a161
# ╟─5714c84f-1653-4c4a-a2e4-003d8560484a
# ╠═69cf84e2-0aba-4595-8cb0-c082dbccdbe2
# ╠═9d96ede3-533e-42f7-ada1-6e71980bc6c2
# ╠═6ff92fca-6d66-4f27-8e09-11a3887e66ba
# ╠═2782903e-1d2e-47de-9109-acff4595de42
# ╟─a17e5acd-d78d-4fab-9ab2-f01bd888339d
# ╟─a3387a7e-436c-4724-aa29-92e78ea3a89f
# ╠═7e777dba-b389-4549-a93a-9b0394646c57
# ╠═4f43373d-42ee-4269-9862-f53695351ea3
# ╠═ada800ba-25e2-4544-a297-c42d8b36a9ff
# ╠═299116ea-66f3-4e52-ab0f-594249b9dd23
# ╟─af8acfdf-32bd-43c1-82d0-99008ee4cb3e
# ╟─3738f45d-38e5-415f-a4e6-f8922df84d09
# ╠═36c1f5c8-ac43-41ea-9100-8f85c1ee3708
# ╟─eb5d3e74-f156-43a1-9966-4880f80a3d60
# ╠═0f6929b6-d869-4c3c-90f6-c923b265b164
# ╠═8b8fd9b8-a41b-4fef-96b7-a146986c6f82
# ╠═344092df-c60b-4f8d-8992-cae088664632
# ╟─e5b9d8ab-6929-4de5-b197-67c1e93cd472
# ╠═8ff69e30-811c-482e-833e-82357d258f95
# ╠═11effb8a-b4f9-45c6-9b59-e8952fe40f3e
# ╟─4479273c-e94e-4630-94e8-881ab18d8642
# ╠═0c8f46e8-1838-47fd-bc42-28df86ec6eb3
# ╠═05a5b2eb-cc06-4d23-9a23-15bf6db484db
# ╠═03060875-d423-4448-bd81-9729161b7ea5
# ╟─7d108fb6-3a4e-4158-b416-d39f3ee43a54
# ╟─3a428568-9991-449d-bae8-42395cdaa80e
# ╠═13c83a66-46c0-4469-bd0c-77e7ec2fa459
# ╟─72b573a4-b426-48f0-937c-7bdf841d9687
# ╠═ccbc151e-c24f-4588-9420-7dd527887daf
# ╠═f9070a63-d154-47be-afd9-d45f13bbfc17
# ╟─a253f0f3-5e77-4adc-985e-96aa7891e954
# ╠═bb39c8ff-dfec-47e0-9508-bec9068ca827
# ╠═81f5a184-84ab-4868-9ecb-73fd17e4dd6b
# ╠═1558fb11-49d7-4931-b4fb-c447b03193c9
# ╟─fa610de0-f8c7-4c48-88d8-f5398ea75ae2
# ╠═d695b118-6d0d-401d-990f-85ba467cc53e
# ╠═6edc243e-59ac-4c6f-b507-80d3ec13bc21
# ╠═3bdde6cf-3b68-46d3-bf76-d68c20b661e9
# ╠═77fc32cf-4b3d-4320-a7d3-8c70df73811c
# ╠═6196c8e4-8f80-4b16-bf30-0f0b828c41de
# ╠═7905f581-1593-4e06-8aaf-faec05c3b306
# ╠═4cb0e4e1-99b3-41fd-b1d8-d19e7103cafe
# ╠═69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
# ╟─eef918a9-b8af-4d41-85b1-bebf1c7889cc
# ╟─fd75ff49-b5de-48dc-ae89-06bf855d81b2
# ╠═db416236-b920-42a0-b5c6-e472d2a3c224
# ╠═d339a276-296a-4378-82ae-fe498e9b5181
# ╟─58e626f1-32fb-465a-839e-1f413411c6f3
# ╟─5ec88a5a-71e2-40c1-9913-98ced174341a
# ╟─12d0a092-e5d5-4aa1-ac82-5d262f54aa6d
# ╟─d2c516c0-f5e5-4476-b7d6-89862f6f2472
# ╟─72641129-5274-47b6-9967-fa37c8036552
# ╠═90f34d85-3fdc-4e2a-ada4-085154103c6b
# ╟─00c94b56-d39d-4289-9a61-1f07dee2cd57
# ╠═10cc73dc-f07d-4674-8aff-e7025be20ccb
# ╟─788e10bf-fb30-43f3-bbd3-aafd59467636
# ╟─81e13ad0-53c6-42e1-8e0f-8345d6f42cc6
# ╠═3218558d-01e2-4dba-a90a-cc64f2952869
# ╟─14b329fb-8053-4148-8d24-4458e592e7e3
# ╟─eebf5285-2336-4c07-a4fd-b1fd841dee52
# ╟─9c345c36-6519-4bc6-8947-df88885bceb0
# ╟─0187a663-34aa-4464-8504-c969d2d9ec11
# ╠═c8577196-e64f-4cfd-9c9d-bc0604fd1f8c
# ╟─aa012d54-38b7-411f-ab89-35fb4a074d39
# ╟─b689d666-37da-40f7-adb8-44aa2b9f5139
# ╟─55ce32ff-dec3-4bd4-b6a2-95483e7637e9
# ╟─d381d944-5069-4f16-8194-bd49eb2fe1cd
# ╟─80406819-83d2-4625-8ed3-959c127e3e2c
# ╟─5a4ac4ed-6ed5-4d28-981e-8bf7b6b8889d
# ╟─0903dd95-5525-44e5-891d-acbe2fb2190f
# ╟─c6f5e697-b72a-441e-9a05-e47a09bee2f7
# ╟─c01ff616-e570-4013-a0b2-d97fcda6f279
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
