using Plots
using Statistics

function boxplot_2d(x, y; ax, whis=1.5)
    xlimits = [percentile(x, q) for q in (25, 50, 75)]
    ylimits = [percentile(y, q) for q in (25, 50, 75)]

    # the box
    box = Shape([xlimits[1], xlimits[3], xlimits[3], xlimits[1]], [ylimits[1], ylimits[1], ylimits[3], ylimits[3]])
    plot!(ax, box, color=:black, linecolor=:black)

    # the x median
    vline = plot!([xlimits[2], xlimits[2]], [ylimits[1], ylimits[3]], color=:black)

    # the y median
    hline = plot!([xlimits[1], xlimits[3]], [ylimits[2], ylimits[2]], color=:black)

    # the central point
    scatter!([xlimits[2]], [ylimits[2]], color=:black, markersize=4)

    # the x-whisker
    iqr_x = xlimits[3] - xlimits[1]

    # left
    left_x = minimum(filter(xi -> xi > xlimits[1] - whis * iqr_x, x))
    whisker_line_x = plot!([left_x, xlimits[1]], [ylimits[2], ylimits[2]], color=:black)
    whisker_bar_x = plot!([left_x, left_x], [ylimits[1], ylimits[3]], color=:black)

    # right
    right_x = maximum(filter(xi -> xi < xlimits[3] + whis * iqr_x, x))
    whisker_line_x = plot!([right_x, xlimits[3]], [ylimits[2], ylimits[2]], color=:black)
    whisker_bar_x = plot!([right_x, right_x], [ylimits[1], ylimits[3]], color=:black)

    # the y-whisker
    iqr_y = ylimits[3] - ylimits[1]

    # bottom
    bottom_y = minimum(filter(yi -> yi > ylimits[1] - whis * iqr_y, y))
    whisker_line_y = plot!([xlimits[2], xlimits[2]], [bottom_y, ylimits[1]], color=:black)
    whisker_bar_y = plot!([xlimits[1], xlimits[3]], [bottom_y, bottom_y], color=:black)

    # top
    top_y = maximum(filter(yi -> yi < ylimits[3] + whis * iqr_y, y))
    whisker_line_y = plot!([xlimits[2], xlimits[2]], [top_y, ylimits[3]], color=:black)
    whisker_bar_y = plot!([xlimits[1], xlimits[3]], [top_y, top_y], color=:black)

    # outliers
    # mask = (x .< left_x) .| (x .> right_x) .| (y .< bottom_y) .| (y .> top_y)
    # scatter!(ax, x[mask], y[mask], color=:black, markersize=2, marker=:circle, markerstrokecolor=:black)
end

# some fake data
x = rand(1000) .^ 2
y = sqrt.(rand(1000))

# plotting the original data
scatter(x, y, color=:red, markersize=1, label="Data")

# creating a new plot for the boxplot
plot(layout=(1, 2))

# doing the box plot
boxplot_2d(x, y, ax=current(), whis=1)


