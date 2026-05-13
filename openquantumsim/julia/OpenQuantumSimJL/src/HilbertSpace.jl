abstract type AbstractHilbertSpace end

struct FockSpace <: AbstractHilbertSpace
    N::Int
    label::Union{Nothing,String}
    function FockSpace(N::Int; label::Union{Nothing,String}=nothing)
        N > 0 || throw(ArgumentError("FockSpace dimension N must be positive."))
        new(N, label)
    end
end

struct SpinSpace <: AbstractHilbertSpace
    S::Float64
    label::Union{Nothing,String}
    function SpinSpace(S::Real; label::Union{Nothing,String}=nothing)
        dim_value = 2 * Float64(S) + 1
        S >= 0 || throw(ArgumentError("SpinSpace requires S >= 0."))
        isapprox(dim_value, round(dim_value); atol=1e-12, rtol=0.0) ||
            throw(ArgumentError("SpinSpace requires 2S + 1 to be an integer."))
        new(Float64(S), label)
    end
end

struct DickeSpace <: AbstractHilbertSpace
    n_spins::Int
    label::Union{Nothing,String}
    function DickeSpace(n_spins::Int; label::Union{Nothing,String}=nothing)
        n_spins >= 0 || throw(ArgumentError("DickeSpace requires n_spins >= 0."))
        new(n_spins, label)
    end
end

struct CompositeSpace <: AbstractHilbertSpace
    spaces::Tuple
    label::Union{Nothing,String}
    function CompositeSpace(spaces::Tuple; label::Union{Nothing,String}=nothing)
        length(spaces) > 0 || throw(ArgumentError("CompositeSpace requires at least one space."))
        all(space -> space isa AbstractHilbertSpace, spaces) ||
            throw(ArgumentError("CompositeSpace entries must be Hilbert spaces."))
        new(spaces, label)
    end
end

dim(space::FockSpace)::Int = space.N
dim(space::SpinSpace)::Int = Int(round(2 * space.S + 1))
dim(space::DickeSpace)::Int = space.n_spins + 1
dim(space::CompositeSpace)::Int = prod(dim(space_i) for space_i in space.spaces)

function tensor_space(spaces::AbstractHilbertSpace...)::CompositeSpace
    return CompositeSpace(Tuple(spaces))
end
