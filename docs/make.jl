using Documenter
using SignatureTensors  # <- tu módulo principal

makedocs(
    sitename = "SignatureTensors.jl",
    modules = [SignatureTensors],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/leonardSchmitz/signature-tensors-in-OSCAR.git",
)

# deploydocs()  # Descomenta solo si quieres subir a GitHub Pages