using Documenter
using SignatureTensors
using DocumenterCitations   


bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"))  # <- docs/refs.bib

makedocs(
    sitename = "SignatureTensors.jl",
    modules = [SignatureTensors],
    pages = [
        "Home" => "index.md",
        "Documentation" => "api.md",
        "References" => "references.md",
    ],
    checkdocs = :warn,
    plugins = [bib],  
)

deploydocs(
    repo = "github.com/leonardSchmitz/signature-tensors-in-OSCAR.git",
)

