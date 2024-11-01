using CairoMakie

# Use the command `\the\textwidth` in LaTeX to get the width of the line in points
# Use the command
# ```
# \makeatletter
# \f@size
# \makeatother
# ```
# to get the font size in points
plot_figsize_width_two_columns_pt = 510
plot_figsize_width_one_column_pt = 246
plot_figsize_height_pt = 300

my_theme_Cairo = Theme(
    fontsize = 9,
    figure_padding = (1,7,1,5),
    size = (plot_figsize_width_one_column_pt, plot_figsize_height_pt),
    Axis = (
        spinewidth=0.7,
        xgridvisible=false,
        ygridvisible=false,
        xtickwidth=0.75,
        ytickwidth=0.75,
        xminortickwidth=0.5,
        yminortickwidth=0.5,
        xticksize=3,
        yticksize=3,
        xminorticksize=1.5,
        yminorticksize=1.5,
        xlabelpadding=1,
        ylabelpadding=1,
    ),
    Legend = (
        merge=true,
        framevisible=false,
        patchsize=(15,2),
    )
)
my_theme_Cairo = merge(my_theme_Cairo, theme_latexfonts())

CairoMakie.set_theme!(my_theme_Cairo)
CairoMakie.activate!(type = "svg", pt_per_unit = 1.5)
CairoMakie.enable_only_mime!(MIME"image/svg+xml"())
