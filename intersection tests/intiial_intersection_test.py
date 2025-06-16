from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
import matplotlib.pyplot as plt

## Intersecting Polygons Test
sq1 = [
    LineString([(0, 0), (2, 0)]),
    LineString([(2, 0), (2, 2)]),
    LineString([(2, 2), (0, 2)]),
    LineString([(0, 2), (0, 0)])
]

sq2 = [
    LineString([(1, 1), (3, 1)]),
    LineString([(3, 1), (3, 3)]),
    LineString([(3, 3), (1, 3)]),
    LineString([(1, 3), (1, 1)])
]

sq3 = [
    LineString([(1.5, 0.5), (1.5, 1.5)]),
    LineString([(1.5, 1.5), (0.5, 1.5)]),
    LineString([(0.5, 1.5), (0.5, 0.5)]),
    LineString([(0.5, 0.5), (1.5, 0.5)])
]

sq4 = [
    LineString([(2, 0), (3, 0)]),
    LineString([(3, 0), (3, 2)]),
    LineString([(3, 2), (2, 2)]),
    LineString([(2, 2), (2, 0)])
]

# convex polygon
cp1 = [
    LineString([(1, 1), (2, 2)]),
    LineString([(2, 2), (3, 1)]),
    LineString([(3, 1), (4, 2)]),
    LineString([(4, 2), (2, 4)]),
    LineString([(2, 4), (2, 3)]),
    LineString([(2, 3), (1, 2)]),
    LineString([(1, 2), (1, 1)]),
]

cp2 = [
    LineString([(1, 1), (2, 2)]),
    LineString([(2, 2), (3, 1)]),
    LineString([(3, 1), (1, 1)]),
]

cp3 = [
    LineString([(1, 1), (3, 3)]),
    LineString([(3, 3), (3, 1)]),
    LineString([(3, 1), (1, 1)])
]

cp4 = [
    LineString([(2, 2), (3, 3)]),
    LineString([(3, 3), (3, 1)]),
    LineString([(3, 1), (2, 2)])
]

# Disconnected/multi polygon test
d1 = [
    LineString([(0, 0), (1, 0)]),
    LineString([(1, 0), (1, 1)]),
    LineString([(1, 1), (0, 1)]),
    LineString([(0, 1), (0, 0)])
]

d2 = [
    LineString([(2, 2), (3, 2)]),
    LineString([(3, 2), (3, 3)]),
    LineString([(3, 3), (2, 3)]),
    LineString([(2, 3), (2, 2)])
]

d3 = [
    LineString([(1, 1), (3, 1)]),
    LineString([(3, 1), (3, 3)]),
    LineString([(3, 3), (1, 3)]),
    LineString([(1, 3), (1, 1)])
]

d4 = [
    LineString([(3, 4), (3, 4)]),
    LineString([(3, 4), (3, 3)]),
    LineString([(3, 3), (4, 3)]),
    LineString([(4, 3), (4, 4)])
]

multi_edges1 = d1 + d2
polygons1 = list(polygonize(multi_edges1))

multi1 = unary_union(polygons1)

multi_edges2 = d1 + d3
polygons2 = list(polygonize(multi_edges2))

multi2 = unary_union(polygons2)

multi_edges3 = d1 + d4
polygons3 = list(polygonize(multi_edges3))

multi3 = unary_union(polygons3)


def test(p1, p2):
    poly1 = list(polygonize(p1))[0]
    poly2 = list(polygonize(p2))[0]
    return (poly1.intersects(poly2), poly1.overlaps(poly2), poly1.contains(poly2))

def plot(e1, e2, xlim, ylim, intersect, overlap, contains):
    # Plot the two polygons above
    fig, ax = plt.subplots()

    # Plot edges for polygon 1
    for edge in e1:
        x, y = edge.xy
        ax.plot(x, y, color='blue', linewidth=2, label='Polygon 1' if edge == e1[0] else "")

    # Plot edges for polygon 2
    for edge in e2:
        x, y = edge.xy
        ax.plot(x, y, color='red', linewidth=2, linestyle='--', label='Polygon 2' if edge == e2[0] else "")

    # Formatting the plot
    ax.set_title(f"Intersects: {intersect}, Overlaps: {overlap}, Contains: {contains}")
    ax.set_aspect('equal')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True)
    ax.legend()
    plt.show()

def main():
    print("(intersect, overlap, contains)")
    test1 = test(sq1, sq1)
    print("Test 1, overlapping triangles: ", test1)
    test2 = test(sq1, sq3)
    print("Test 2, triangles within triangles: ", test2)
    test3 = test(sq1, sq4)
    print("Test 3, triangles with shared edges:  ", test3)
    test4 = test(sq3, sq4)
    print("Test 4, triangles with no overlap: ", test4)
    test5 = test(cp1, sq4)
    print("Test 5, polygons with overlap: ", test5)
    test6 = test(cp1, cp2)
    print("Test 6, polygons with shared edges: ", test6)
    test7 = test(cp1, cp3)
    print("Test 7, polygons with shared edges AND overlap: ", test7)
    test8 = test(cp1, cp4)
    print("Test 8, polygon with triangle inside and shared edge: ", test8)
    test9 = test(multi1, cp3)
    print("Test 9, multi polygon with triangle overlapping one part: ", test9)
    test10 = test(multi2, cp3)
    print("Test 10, multi polygon with triangle within one part: ", test10)
    test11 = test(multi3, cp3)
    print("Test 11, multi polygon with triangle not overlapping, 2 corners touch: ", test11)
    test12 = test(multi3, cp4)
    print("Test 12, multi polygon with triangle not overlapping, 1 corner touch: ", test12)


    # overlap
    plot(sq1, sq2, (-1, 4), (-1, 4), test1[0], test1[1], test1[2])
    # contains
    plot(sq1, sq3, (-1, 4), (-1, 4), test2[0], test2[1], test2[2])
    # shared edge
    plot(sq1, sq4, (-1, 4), (-1, 4), test3[0], test3[1], test3[2])
    # no overlap
    plot(sq3, sq4, (-1, 4), (-1, 4), test4[0], test4[1], test4[2])
    # overlap POLYGON
    plot(cp1, sq4, (-1, 5), (-1, 5), test5[0], test5[1], test5[2])
    # shared edge POLYGON
    plot(cp1, cp2, (-1, 5), (-1, 5), test6[0], test6[1], test6[2])
    # shared edge and overlap
    plot(cp1, cp3, (-1, 5), (-1, 5), test7[0], test7[1], test7[2])
    # shared edge and overlap
    plot(cp1, cp4, (-1, 5), (-1, 5), test8[0], test8[1], test8[2])
    # multi polygon, triangle overlaps one disconnected comp
    plot(multi_edges1, cp3, (-1, 5), (-1, 5), test9[0], test9[1], test9[2])
    # multi polygon, triangle within one disconnected comp
    plot(multi_edges2, cp3, (-1, 5), (-1, 5), test10[0], test10[1], test10[2])
    # multi polygon, triangle not overlapping 
    # NOTE: intersects returns TRUE for this case when we want FALSE
    plot(multi_edges3, cp3, (-1, 5), (-1, 5), test11[0], test11[1], test11[2])
    # multi polygon, triangle within one disconnected comp
    plot(multi_edges3, cp4, (-1, 5), (-1, 5), test12[0], test12[1], test12[2])

if __name__ == "__main__":
    main()