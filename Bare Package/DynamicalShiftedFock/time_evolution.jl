#=
This file includes some basic functions for the time evolution of quantum systems. They are needed to define the Dynamical Shifted Fock algorithm.
This file doesn't influence the main behavior of the code DSF algorithm.
=#

export ket2dm, mat2vec, mesolveProblem

ket2dm(ψ::AbstractVector) = ψ * ψ'

ket2dm(ρ::AbstractMatrix) = ρ

mat2vec(A::AbstractMatrix) = vec(A)

mesolve_ti_dudt!(du, u, p, t) = mul!(du, p.L, u)

function _save_func_mesolve(integrator)
    internal_params = integrator.p
    progr = internal_params.progr

    if !internal_params.is_empty_e_ops
        expvals = internal_params.expvals
        e_ops = internal_params.e_ops
        # This is equivalent to tr(op * ρ), when both are matrices.
        # The advantage of using this convention is that I don't need
        # to reshape u to make it a matrix, but I reshape the e_ops once.

        ρ = integrator.u
        _expect = op -> dot(op, ρ)
        @. expvals[:, progr.counter[]+1] = _expect(e_ops)
    end
    next!(progr)
    return u_modified!(integrator, false)
end

function mesolveProblem(
    L::MT1,
    ψ0,
    t_l;
    e_ops::AbstractVector = Vector{MT1}([]),
    params::NamedTuple = NamedTuple(),
    progress_bar::Bool = true,
    kwargs...,
) where {
    MT1<:AbstractMatrix,
}
    (length(ψ0) == size(L, 1) || length(ψ0) == isqrt(size(L, 1))) || throw(DimensionMismatch("The two quantum objects don't have the same Hilbert space dimension."))

    ρ0 = mat2vec(ket2dm(ψ0))

    progr = ProgressBar(length(t_l), enable = progress_bar)
    expvals = Array{ComplexF64}(undef, length(e_ops), length(t_l))
    e_ops2 = @. mat2vec(adjoint(e_ops))

    p = (
        L = L,
        progr = progr,
        e_ops = e_ops2,
        expvals = expvals,
        is_empty_e_ops = isempty(e_ops),
        params...,
    )

    default_values = (abstol = 1e-7, reltol = 1e-5, saveat = [t_l[end]])
    kwargs2 = merge(default_values, kwargs)
    if !isempty(e_ops) || progress_bar
        cb1 = PresetTimeCallback(t_l, _save_func_mesolve, save_positions = (false, false))
        kwargs2 =
            haskey(kwargs, :callback) ? merge(kwargs2, (callback = CallbackSet(kwargs2.callback, cb1),)) :
            merge(kwargs2, (callback = cb1,))
    end

    tspan = (t_l[1], t_l[end])
    return ODEProblem{true}(mesolve_ti_dudt!, ρ0, tspan, p; kwargs2...)
end
