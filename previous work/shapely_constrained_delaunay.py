import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import polygonize
from shapely.plotting import plot_polygon

def plot_triangles(triangles, title):
    fig, ax = plt.subplots()
    for tri in triangles.geoms:
        x, y = tri.exterior.xy
        ax.fill(x, y, edgecolor='black', facecolor='lightblue', alpha=0.6)
    ax.set_title(title)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

# Example 1: Square
square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
triangles = shapely.constrained_delaunay_triangles(square)
plot_triangles(triangles, "Constrained Delaunay Triangles - Square")

# Example 2: Polygon with hole
polygon_with_hole = Polygon(
    shell=[(0, 0), (3, 0), (3, 3), (0, 3)],
    holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]]
)
triangles_with_hole = shapely.constrained_delaunay_triangles(polygon_with_hole)
plot_triangles(triangles_with_hole, "Constrained Delaunay Triangles - With Hole")

# Example 3: Intersecting diagonals inside a square
lines = MultiLineString([
    # Outer square
    [(0, 0), (2, 0)],
    [(2, 0), (2, 2)],
    [(2, 2), (0, 2)],
    [(0, 2), (0, 0)],
    # Crossing diagonals (internal constraints)
    [(0, 0), (2, 2)],
    [(0, 2), (2, 0)],
])

# polygonize extracts polygons from closed rings in the lines
polygons = list(polygonize(lines))

# Display results
print(f"Found {len(polygons)} polygon(s)")

# Just for visualization
fig, ax = plt.subplots()
for poly in polygons:
    plot_polygon(poly, ax=ax, add_points=False)
plt.title("Polygonized from Edges")
plt.axis('equal')
plt.show()

nonmanifold_tris = shapely.constrained_delaunay_triangles(polygons[0])
plot_triangles(nonmanifold_tris, "Constrained Delaunay Triangles - Edges to Polygon Check")

# Weird, jagged polygon (valid)
coords = [
    (0, 0), (2, 0.5), (3, 1.5), (2.5, 2), (3, 3),
    (1.5, 2.7), (1, 3.5), (0.5, 2.5), (-0.5, 3),
    (-1.5, 2), (-1, 1), (-2, 0.5), (-1.5, -0.5), (-0.5, -1), (0, 0)
]

weird_shape = Polygon(coords)

# Triangulate
weird_tris = shapely.constrained_delaunay_triangles(weird_shape)

# Plot
plot_triangles(weird_tris, "Constrained Delaunay Triangles - Edges to Polygon Check")
