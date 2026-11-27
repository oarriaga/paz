import subprocess
from pathlib import Path

PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath, bm, graphicx}
\usepackage[dvipsnames]{xcolor}
\usepackage{tikz}
\usetikzlibrary{bayesnet, positioning}
\begin{document}
\begin{tikzpicture}[node distance=0.4cm]
"""

MODEL = r"""\node[latent] (image) {$\mathcal{I}$};
\node[det, fill=Dandelion!50, above=0.25 of image] (render) {$\mathcal{R}$};
\node[latent, above=0.25 of render] (object) {$\mathcal{O}$};
\node[latent, above=0.5 of object, xshift=-3.5cm] (material) {$\mathcal{M}$};
\node[latent, above=0.5 of object, xshift=3.5cm] (transform) {$\mathcal{A}$};
\node[latent, above=of object, yshift=1.0cm] (classes) {$\bm{\kappa}$};
\node[latent, above=0.5 of material, xshift=-2cm] (color) {$c$};
\node[const, above=of color, xshift=-0.5cm] (empty) {};
\node[latent, above=0.5 of material, xshift=-1cm] (ambient) {$k^a$};
\node[latent, above=0.5 of material] (specular) {$k^s$};
\node[latent, above=0.5 of material, xshift=1cm] (diffuse) {$k^d$};
\node[latent, above=0.5 of material, xshift=2cm] (shininess) {$\alpha$};
\node[latent, above=0.5 of transform, xshift=-2.0cm] (shift) {$t$};
\node[latent, above=0.5 of transform] (theta) {$\theta$};
\node[latent, above=0.5 of transform, xshift=2.5cm] (scale) {$\mathbf{s}$};
\node[const, above=of shift, xshift=-0.5cm] (shift_mean) {$\mu_t$};
\node[const, above=of shift, xshift=0.5cm] (shift_stdv) {$\sigma_t$};
\node[const, above=of theta, xshift=-0.5cm] (theta_mean) {$\mu_\theta$};
\node[const, above=of theta, xshift=0.5cm] (theta_stdv) {$\alpha_\theta$};
\node[const, above=of scale, xshift=-0.5cm] (scale_mean) {$\mu_s$};
\node[const, above=of scale, xshift=0.5cm] (scale_stdv) {$\sigma_s$};
\node[const, above=of classes, xshift=-0.5cm] (count) {$\bm{\sigma}$};
\node[const, above=of classes, xshift=0.5cm] (temperature) {$t$};
\node[const, right=of render] (camera) {camera};
\node[const, above=0.15 of camera] (lights) {lights};
\node[const, below=0.15 of camera] (ground) {ground};
\factor[above=of shift, yshift=0.15cm] {shift-f}
    {left:$\mathrm{Tr}\mathcal{N}$} {shift_mean, shift_stdv} {shift};
\factor[above=of theta, yshift=0.25cm] {theta-f}
    {left:Mises} {theta_mean, theta_stdv} {theta};
\factor[above=of scale, yshift=0.10cm] {scale-f}
    {left:$\mathrm{Log}\mathcal{N}$} {scale_mean, scale_stdv} {scale};
\factor[above=of classes, yshift=0.25cm] {classes-f}
    {left:$\mathrm{Gmb}$} {count, temperature} {classes};
\factor[above=of ambient, yshift=0.25cm] {ambient-f} {above:GMM} {} {ambient};
\factor[above=of diffuse, yshift=0.25cm] {diffuse-f} {above:GMM} {} {diffuse};
\factor[above=of specular, yshift=0.25cm] {specular-f} {above:GMM} {}
    {specular};
\factor[above=of shininess, yshift=0.25cm] {shininess-f} {above:GMM} {}
    {shininess};
\factor[above=of color, yshift=0.25cm] {color-f} {above:GMM} {} {color};
\edge[-] {classes, material, transform} {object};
\edge[-] {ambient, diffuse, color, specular, shininess} {material};
\edge[-] {shift, theta, scale} {transform};
\edge[-] {object} {render};
\edge[-] {render} {image};
\edge[-] {lights, camera, ground} {render};
\plate {colors} {(color)(color-f)(empty)} {$\{r, g, b\}$};
\plate {shifts} {(shift)(shift-f)(shift_mean)(shift_stdv)} {$\{x, y\}$};
\plate {scales} {(scale)(scale-f)} {$\{s_x, s_y, s_z\}$};
"""

# Conditioned program without the pose variables (translation and angle).
CONDITIONED_MODEL = r"""\definecolor{concept}{rgb}{0.90, 0.90, 0.90}
\tikzset{concept/.style={latent, draw=concept!60, fill=concept!51}}
\node[obs, draw=black!20, fill=black!20] (image) {$\mathcal{I}$};
\node[det, draw=Dandelion!50, fill=Dandelion!51, above=0.22 of image]
    (render) {$\mathcal{R}$};
\node[concept, above=0.22 of render] (object) {$\mathcal{O}$};
\node[concept, above=0.7 of object] (material) {$\mathcal{M}$};
\node[concept, left=2.4 of material] (classes) {$\bm{\kappa}$};
\node[concept, right=2.4 of material] (transform) {$\mathcal{A}$};
\node[concept, above=0.6 of material, xshift=-2cm] (color) {$c$};
\node[concept, above=0.6 of material, xshift=-1cm] (ambient) {$k^a$};
\node[concept, above=0.6 of material] (specular) {$k^s$};
\node[concept, above=0.6 of material, xshift=1cm] (diffuse) {$k^d$};
\node[concept, above=0.6 of material, xshift=2cm] (shininess) {$\alpha$};
\node[concept, above=0.6 of transform] (scale) {$\mathbf{s}$};
\edge[-] {classes, material, transform} {object};
\edge[-] {ambient, diffuse, color, specular, shininess} {material};
\edge[-] {scale} {transform};
\edge[-] {object} {render};
\edge[-] {render} {image};
"""


def write_prior_program(directory, sample_paths):
    cells = [frame_image(path, "white") for path in sample_paths[:9]]
    return write_program(directory, "prior_program", MODEL, cells)


def write_posterior_program(directory, observation_path, sample_paths):
    cells = [frame_image(path, "white") for path in sample_paths[:8]]
    cells.insert(4, frame_image(observation_path, "Dandelion"))
    name = "posterior_program"
    return write_program(directory, name, CONDITIONED_MODEL, cells)


def frame_image(path, color):
    path = Path(path).resolve()
    image = rf"\includegraphics[width=2.2cm]{{\detokenize{{{path}}}}}"
    frame = r"\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{1.5pt}"
    return frame + rf"\fcolorbox{{{color}}}{{white}}{{{image}}}"


def write_program(directory, name, model, cells):
    rows = [" & ".join(cells[arg : arg + 3]) for arg in (0, 3, 6)]
    table = r"\begin{tabular}{@{}c@{}c@{}c@{}}" + "\n"
    table = table + (r"\\[-1pt]" + "\n").join(rows) + "\n" + r"\end{tabular}"
    node = r"\node[below=0.35 of image, inner sep=0pt] (samples)"
    samples = f"{node} {{{table}}};"
    edge = r"\edge[-] {image} {samples};"
    ending = "\\end{tikzpicture}\n\\end{document}\n"
    source = "\n".join([PREAMBLE, model, samples, edge, ending])
    filepath = Path(directory) / f"{name}.tex"
    filepath.write_text(source)
    command = ["pdflatex", "-interaction=nonstopmode", filepath.name]
    subprocess.run(command, cwd=directory, check=True, capture_output=True)
    return filepath.with_suffix(".pdf")
