
export dsf_mesolveProblem

function _DSF_mesolve_Condition(u, t, integrator)
    internal_params = integrator.p
    op_l_vec = internal_params.op_l_vec
    δα_list = internal_params.δα_list

    condition = false
    @inbounds for i in eachindex(δα_list)
        op_vec = op_l_vec[i]
        δα = δα_list[i]
        Δα = dot(op_vec, u)
        if δα < abs(Δα)
            condition = true
        end
    end
    return condition
end

function _DSF_mesolve_Affect!(integrator)
    internal_params = integrator.p
    op_l = internal_params.op_l
    op_l_vec = internal_params.op_l_vec
    αt_list = internal_params.αt_list
    δα_list = internal_params.δα_list
    H = internal_params.H_fun
    c_ops = internal_params.c_ops_fun
    e_ops = internal_params.e_ops_fun
    e_ops_vec = internal_params.e_ops
    dsf_cache = internal_params.dsf_cache
    dsf_params = internal_params.dsf_params
    expv_cache = internal_params.expv_cache
    dsf_identity = internal_params.dsf_identity
    dsf_displace_cache_full = internal_params.dsf_displace_cache_full

    op_l_length = length(op_l)
    fill!(dsf_displace_cache_full.coefficients, 0)

    for i in eachindex(op_l)
        op_vec = op_l_vec[i]
        αt = αt_list[i]
        δα = δα_list[i]
        Δα = dot(op_vec, integrator.u)

        if δα < abs(Δα)
            # Here we commented several ways to displace the states
            # The uncommented code is the most efficient way to do it

            # Dᵢ = exp(Δα*op' - conj(Δα)*op)
            # copyto!(dsf_cache, integrator.u)
            # mul!(integrator.u, sprepost(Dᵢ', Dᵢ), dsf_cache)

            # This is equivalent to the code above, assuming that transpose(op) = adjoint(op)
            # Aᵢ = kron(Δα * op - conj(Δα) * op', dsf_identity) + kron(dsf_identity, conj(Δα) * op - Δα * op')

            # @. dsf_displace_cache_full[i] = Δα * dsf_displace_cache_left[i] - conj(Δα) * dsf_displace_cache_left_dag[i] + conj(Δα) * dsf_displace_cache_right[i] - Δα * dsf_displace_cache_right_dag[i]
            # Aᵢ = dsf_displace_cache_full[i]

            # dsf_cache .= integrator.u
            # arnoldi!(expv_cache, Aᵢ, dsf_cache)
            # expv!(integrator.u, expv_cache, one(αt), dsf_cache)

            dsf_displace_cache_full.coefficients[i] = Δα
            dsf_displace_cache_full.coefficients[i+op_l_length] = -conj(Δα)
            dsf_displace_cache_full.coefficients[i+2*op_l_length] = conj(Δα)
            dsf_displace_cache_full.coefficients[i+3*op_l_length] = -Δα

            αt_list[i] += Δα
        end
    end

    copyto!(dsf_cache, integrator.u)
    arnoldi!(expv_cache, dsf_displace_cache_full, dsf_cache)
    expv!(integrator.u, expv_cache, 1, dsf_cache)

    op_l2 = op_l .+ (αt_list .* Ref(I))
    e_ops2 = e_ops(op_l2, dsf_params)
    _mat2vec = op -> mat2vec(op')
    @. e_ops_vec = _mat2vec(e_ops2)
    return copyto!(internal_params.L, liouvillian(H(op_l2, dsf_params), c_ops(op_l2, dsf_params), dsf_identity))
end

function dsf_mesolveProblem(
    H::Function,
    ψ0::AbstractArray{T},
    t_l::AbstractVector,
    c_ops::Function,
    op_list::Vector{TOL},
    α0_l::AbstractVector{<:Number} = zeros(length(op_list)),
    dsf_params::NamedTuple = NamedTuple();
    alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm = Tsit5(),
    e_ops::Function = (op_list, p) -> Vector{TOL}([]),
    params::NamedTuple = NamedTuple(),
    δα_list::Vector{<:Real} = fill(0.2, length(op_list)),
    krylov_dim::Int = min(5, max(cld(length(ket2dm(ψ0)), 3), 6)),
    kwargs...,
) where {T,TOL}
    op_l = op_list
    H₀ = H(op_l .+ (α0_l .* Ref(I)), dsf_params)
    c_ops₀ = c_ops(op_l .+ (α0_l .* Ref(I)), dsf_params)
    e_ops₀ = e_ops(op_l .+ (α0_l .* Ref(I)), dsf_params)

    αt_list = convert(Vector{T}, α0_l)
    op_l_vec = map(op -> mat2vec(op'), op_l)

    # Create the Krylov subspace with kron(H₀, H₀) just for initialize
    expv_cache = arnoldi(kron(H₀, H₀), mat2vec(ket2dm(ψ0)), krylov_dim)
    dsf_identity = I(size(H₀, 1))
    dsf_displace_cache_left = map(op -> kron(op, dsf_identity), op_l)
    dsf_displace_cache_left_dag = map(op -> kron(sparse(op'), dsf_identity), op_l)
    dsf_displace_cache_right = map(op -> kron(dsf_identity, op), op_l)
    dsf_displace_cache_right_dag = map(op -> kron(dsf_identity, sparse(op')), op_l)
    dsf_displace_cache_full = OperatorSum(
        zeros(length(op_l) * 4),
        vcat(
            dsf_displace_cache_left,
            dsf_displace_cache_left_dag,
            dsf_displace_cache_right,
            dsf_displace_cache_right_dag,
        ),
    )

    params2 = params
    params2 = merge(
        params,
        (
            H_fun = H,
            c_ops_fun = c_ops,
            e_ops_fun = e_ops,
            op_l = op_l,
            op_l_vec = op_l_vec,
            αt_list = αt_list,
            δα_list = δα_list,
            dsf_cache = similar(ψ0, length(ψ0)^2),
            expv_cache = expv_cache,
            dsf_identity = dsf_identity,
            dsf_params = dsf_params,
            dsf_displace_cache_full = dsf_displace_cache_full,
        ),
    )

    cb_dsf = DiscreteCallback(_DSF_mesolve_Condition, _DSF_mesolve_Affect!, save_positions = (false, false))
    kwargs2 = (; kwargs...)
    kwargs2 =
        haskey(kwargs2, :callback) ? merge(kwargs2, (callback = CallbackSet(cb_dsf, kwargs2.callback),)) :
        merge(kwargs2, (callback = cb_dsf,))
    
    L₀ = liouvillian(H₀, c_ops₀, dsf_identity)

    return mesolveProblem(L₀, ψ0, t_l; e_ops = e_ops₀, params = params2, kwargs2...)
end

# function _DSF_mcsolve_Condition(u, t, integrator)
#     internal_params = integrator.p
#     op_l = internal_params.op_l
#     δα_list = internal_params.δα_list

#     ψt = u

#     condition = false
#     @inbounds for i in eachindex(op_l)
#         op = op_l[i]
#         δα = δα_list[i]
#         Δα = dot(ψt, op.data, ψt) / dot(ψt, ψt)
#         if δα < abs(Δα)
#             condition = true
#         end
#     end
#     return condition
# end

# function _DSF_mcsolve_Affect!(integrator)
#     internal_params = integrator.p
#     op_l = internal_params.op_l
#     αt_list = internal_params.αt_list
#     δα_list = internal_params.δα_list
#     H = internal_params.H_fun
#     c_ops = internal_params.c_ops_fun
#     e_ops = internal_params.e_ops_fun
#     e_ops0 = internal_params.e_ops_mc
#     c_ops0 = internal_params.c_ops
#     ψt = internal_params.dsf_cache1
#     dsf_cache = internal_params.dsf_cache2
#     expv_cache = internal_params.expv_cache
#     dsf_params = internal_params.dsf_params
#     dsf_displace_cache_full = internal_params.dsf_displace_cache_full

#     copyto!(ψt, integrator.u)
#     normalize!(ψt)

#     op_l_length = length(op_l)
#     fill!(dsf_displace_cache_full.coefficients, 0)

#     for i in eachindex(op_l)
#         op = op_l[i]
#         αt = αt_list[i]
#         δα = δα_list[i]
#         Δα = dot(ψt, op.data, ψt)

#         if δα < abs(Δα)
#             # Here we commented several ways to displace the states
#             # The uncommented code is the most efficient way to do it

#             # Dᵢ = exp(Δα*op' - conj(Δα)*op)
#             # dsf_cache .= integrator.u
#             # mul!(integrator.u, Dᵢ.data', dsf_cache)

#             # Aᵢ = -Δα*op.data' + conj(Δα)*op.data
#             # dsf_cache .= integrator.u
#             # arnoldi!(expv_cache, Aᵢ, dsf_cache)
#             # expv!(integrator.u, expv_cache, one(αt), dsf_cache)

#             dsf_displace_cache_full.coefficients[i] = conj(Δα)
#             dsf_displace_cache_full.coefficients[i+op_l_length] = -Δα

#             αt_list[i] += Δα
#         end
#     end

#     copyto!(dsf_cache, integrator.u)
#     arnoldi!(expv_cache, dsf_displace_cache_full, dsf_cache)
#     expv!(integrator.u, expv_cache, 1, dsf_cache)

#     op_l2 = op_l .+ (αt_list .* Ref(I))
#     e_ops0 = e_ops(op_l2, dsf_params)
#     e_ops0 = c_ops(op_l2, dsf_params)
#     T = eltype(ψt)
#     H_eff = H(op_l2, dsf_params) - lmul!(T(0.5im), mapreduce(op -> op' * op, +, c_ops0))
#     return mul!(internal_params.U, -1im, H_eff)
# end

# function _dsf_mcsolve_prob_func(prob, i, repeat)
#     internal_params = prob.p

#     prm = merge(
#         internal_params,
#         (
#             U = copy(internal_params.U),
#             e_ops_mc = copy(internal_params.e_ops_mc),
#             c_ops = copy(internal_params.c_ops),
#             expvals = similar(internal_params.expvals),
#             cache_mc = similar(internal_params.cache_mc),
#             weights_mc = similar(internal_params.weights_mc),
#             cumsum_weights_mc = similar(internal_params.weights_mc),
#             random_n = Ref(rand()),
#             progr_mc = ProgressBar(size(internal_params.expvals, 2), enable = false),
#             jump_times_which_idx = Ref(1),
#             jump_times = similar(internal_params.jump_times),
#             jump_which = similar(internal_params.jump_which),
#             αt_list = copy(internal_params.αt_list),
#             dsf_cache1 = similar(internal_params.dsf_cache1),
#             dsf_cache2 = similar(internal_params.dsf_cache2),
#             expv_cache = copy(internal_params.expv_cache),
#             dsf_displace_cache_full = OperatorSum(
#                 copy(internal_params.dsf_displace_cache_full.coefficients),
#                 internal_params.dsf_displace_cache_full.operators,
#             ),
#         ),
#     )

#     return remake(prob, p = prm)
# end

# function dsf_mcsolveEnsembleProblem(
#     H::Function,
#     ψ0::QuantumObject{<:AbstractArray{T},KetQuantumObject},
#     t_l::AbstractVector,
#     c_ops::Function,
#     op_list::Vector{TOl},
#     α0_l::Vector{<:Number} = zeros(length(op_list)),
#     dsf_params::NamedTuple = NamedTuple();
#     alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm = Tsit5(),
#     e_ops::Function = (op_list, p) -> Vector{TOl}([]),
#     H_t::Union{Nothing,Function,TimeDependentOperatorSum} = nothing,
#     params::NamedTuple = NamedTuple(),
#     δα_list::Vector{<:Real} = fill(0.2, length(op_list)),
#     jump_callback::TJC = ContinuousLindbladJumpCallback(),
#     krylov_dim::Int = min(5, cld(length(ψ0.data), 3)),
#     kwargs...,
# ) where {T,TOl,TJC<:LindbladJumpCallbackType}
#     op_l = op_list
#     H₀ = H(op_l .+ α0_l, dsf_params)
#     c_ops₀ = c_ops(op_l .+ α0_l, dsf_params)
#     e_ops₀ = e_ops(op_l .+ α0_l, dsf_params)

#     αt_list = convert(Vector{T}, α0_l)
#     expv_cache = arnoldi(H₀.data, ψ0.data, krylov_dim)

#     dsf_displace_cache = map(op -> Qobj(op.data), op_l)
#     dsf_displace_cache_dag = map(op -> Qobj(sparse(op.data')), op_l)
#     dsf_displace_cache_full = OperatorSum(zeros(length(op_l) * 2), vcat(dsf_displace_cache, dsf_displace_cache_dag))

#     params2 = merge(
#         params,
#         (
#             H_fun = H,
#             c_ops_fun = c_ops,
#             e_ops_fun = e_ops,
#             op_l = op_l,
#             αt_list = αt_list,
#             δα_list = δα_list,
#             dsf_cache1 = similar(ψ0.data),
#             dsf_cache2 = similar(ψ0.data),
#             expv_cache = expv_cache,
#             dsf_params = dsf_params,
#             dsf_displace_cache_full = dsf_displace_cache_full,
#         ),
#     )

#     cb_dsf = DiscreteCallback(_DSF_mcsolve_Condition, _DSF_mcsolve_Affect!, save_positions = (false, false))
#     kwargs2 = (; kwargs...)
#     kwargs2 =
#         haskey(kwargs2, :callback) ? merge(kwargs2, (callback = CallbackSet(cb_dsf, kwargs2.callback),)) :
#         merge(kwargs2, (callback = cb_dsf,))

#     return mcsolveEnsembleProblem(
#         H₀,
#         ψ0,
#         t_l,
#         c_ops₀;
#         e_ops = e_ops₀,
#         alg = alg,
#         H_t = H_t,
#         params = params2,
#         jump_callback = jump_callback,
#         prob_func = _dsf_mcsolve_prob_func,
#         kwargs2...,
#     )
# end
