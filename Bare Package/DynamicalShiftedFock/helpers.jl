#=
This file contains some helper functions, such as the definition of the bosonic annihilation operator, the definition of coherent states, and many others.
This file doesn't influence the main behavior of the code DSF algorithm.
=#

export destroy, fock, spre, spost, sprepost, lindblad_dissipator, liouvillian

destroy(N::Int) = spdiagm(1 => Array{ComplexF64}(sqrt.(1:N-1)))

function fock(N::Int, pos::Int = 0)
    array = zeros(ComplexF64, N)
    array[pos+1] = 1
    return array
end

raw"""
    spre(A, Id_cache=I(size(A,1)))

Returns the SuperOperator form of `A` acting on the left of the density matrix operator: ``\mathcal{O} \left(\hat{A}\right) \left[ \hat{\rho} \right] = \hat{A} \hat{\rho}``.

Since the density matrix is vectorized, this SuperOperator is always a matrix ``\hat{\mathbb{1}} \otimes \hat{A}``, namely 

```math
\mathcal{O} \left(\hat{A}\right) \left[ \hat{\rho} \right] = \hat{\mathbb{1}} \otimes \hat{A} ~ |\hat{\rho}\rangle\rangle
```

The optional argument `Id_cache` can be used to pass a precomputed identity matrix. This can be useful when
the same function is applied multiple times with a known Hilbert space dimension.
"""
spre(A, Id_cache = I(size(A, 1))) = kron(Id_cache, A)

raw"""
    spost(B, Id_cache=I(size(B,1)))

Returns the SuperOperator form of `B` acting on the right of the density matrix operator: ``\mathcal{O} \left(\hat{B}\right) \left[ \hat{\rho} \right] = \hat{\rho} \hat{B}``.

Since the density matrix is vectorized, this SuperOperator is always a matrix ``\hat{B}^T \otimes \hat{\mathbb{1}}``, namely

```math
\mathcal{O} \left(\hat{B}\right) \left[ \hat{\rho} \right] = \hat{B}^T \otimes \hat{\mathbb{1}} ~ |\hat{\rho}\rangle\rangle
```

The optional argument `Id_cache` can be used to pass a precomputed identity matrix. This can be useful when
the same function is applied multiple times with a known Hilbert space dimension.
"""
spost(B, Id_cache = I(size(B, 1))) = kron(transpose(B), Id_cache)

raw"""
    sprepost(A, B)

Returns the SuperOperator form of `A` and `B` acting on the left and right of the density matrix operator, respectively: ``\mathcal{O} \left( \hat{A}, \hat{B} \right) \left[ \hat{\rho} \right] = \hat{A} \hat{\rho} \hat{B}``.

Since the density matrix is vectorized, this SuperOperator is always a matrix ``\hat{B}^T \otimes \hat{A}``, namely

```math
\mathcal{O} \left(\hat{A}, \hat{B}\right) \left[ \hat{\rho} \right] = \hat{B}^T \otimes \hat{A} ~ |\hat{\rho}\rangle\rangle = \textrm{spre}(A) * \textrm{spost}(B) ~ |\hat{\rho}\rangle\rangle
```

See also [`spre`](@ref) and [`spost`](@ref).
"""
function sprepost(A, B)
    size(A) != size(B) && throw(DimensionMismatch("The two quantum objects don't have the same Hilbert space dimension."))

    return kron(transpose(sparse(B)), A)
end

raw"""
    lindblad_dissipator(O, Id_cache=I(size(O,1))

Returns the Lindblad SuperOperator defined as

```math
\mathcal{D} \left( \hat{O} \right) \left[ \hat{\rho} \right] = \frac{1}{2} \left( 2 \hat{O} \hat{\rho} \hat{O}^\dagger - 
\hat{O}^\dagger \hat{O} \hat{\rho} - \hat{\rho} \hat{O}^\dagger \hat{O} \right)
```

The optional argument `Id_cache` can be used to pass a precomputed identity matrix. This can be useful when
the same function is applied multiple times with a known Hilbert space dimension.
"""
function lindblad_dissipator(O, Id_cache = I(size(O, 1)))
    Od_O = O' * O
    return sprepost(O, O') - spre(Od_O, Id_cache) / 2 - spost(Od_O, Id_cache) / 2
end

function liouvillian(H, c_ops::AbstractVector, Id_cache = I(size(H, 1)))
    return -1im * (spre(H, Id_cache) - spost(H, Id_cache)) + mapreduce(lindblad_dissipator, +, c_ops)
end

struct OperatorSum{CT<:Vector{<:Number},OT<:AbstractVector{<:AbstractMatrix}}
    coefficients::CT
    operators::OT
    function OperatorSum(coefficients::CT, operators::OT) where {CT<:Vector{<:Number},OT<:Vector{<:AbstractMatrix}}
        length(coefficients) == length(operators) ||
            throw(DimensionMismatch("The number of coefficients must be the same as the number of operators."))
        
        T = promote_type(
            mapreduce(eltype, promote_type, operators),
            mapreduce(eltype, promote_type, coefficients),
        )
        coefficients2 = T.(coefficients)
        return new{Vector{T},OT}(coefficients2, operators)
    end
end

Base.size(A::OperatorSum) = size(A.operators[1])
Base.size(A::OperatorSum, inds...) = size(A.operators[1], inds...)

function update_coefficients!(A::OperatorSum, coefficients)
    length(A.coefficients) == length(coefficients) ||
        throw(DimensionMismatch("The number of coefficients must be the same as the number of operators."))
    return A.coefficients .= coefficients
end

@inline function LinearAlgebra.mul!(y::AbstractVector{T}, A::OperatorSum, x::AbstractVector, α, β) where {T}
    # Note that β is applied only to the first term
    mul!(y, A.operators[1], x, α * A.coefficients[1], β)
    @inbounds for i in 2:length(A.operators)
        A.coefficients[i] == 0 && continue
        mul!(y, A.operators[i], x, α * A.coefficients[i], 1)
    end
    return y
end
