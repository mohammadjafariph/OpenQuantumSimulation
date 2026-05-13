using PackageCompiler

create_sysimage(
    [:OpenQuantumSimJL];
    project=joinpath(@__DIR__, "..", "src", "OpenQuantumSimJL"),
    sysimage_path=joinpath(@__DIR__, "..", "OpenQuantumSimJL_sysimage.so"),
)
