# ventura-research

## Where we are at Currently

Reconstructing the geometry of the .glb (see uv_new_delaunay_geometry_w_contains.py)

Our current rendition has some edges stretched out more than we would like - it may have to do with which xyz coords we are assigning each pix on each layer.

We tested the methods we used using the code in 'intersection tests' - specifically the layer intersection tests visualizes a layer of a glb and the result of running contains on the new geometry created with Delaunay.

In the .glb folder there are example .glbs made with the original UV script, and the ones ending with '...contains.glb' are made using the modified uv script.

NOTE: after using this UV script, need to rescale with the rescale script (the ones in the glb folder have not been rescaled).

## Background + Steps Taken

The goal we had was to improve the simplification method of our current process such that the edges do not appear jagged.

![simplification_comparison]](image.png)
So, we looked into the simplification process we are currently using with meshoptimizer.

https://github.com/zeux/meshoptimizer

We do use the -slb option when we simplify, yet it seems that we are not identifying border edges properly. Some hypotheses we had for this was that it's because our mesh is non-manifold/not watertight. It seems as if the way the simplification identifies borders is checking if there is only one face attached to an edge. However, our edges may have redundant triangles attached because of the way we make the geometry.

Originally, we tested out making the geometry non-manifold/removing degenerate/duplicate faces with trimesh but it did not seem to result the way we wanted to (see previous work\remove_manifold_degenerate.ipynb). We also attempted to use Shapely's constrained_deluanay method (see previous work\shapely_constrained_delaunay.py).

However, in the end, it seemed easiest to re-construct the geometry so that it is nicer; hopefully, the simplification method used above will identify the borders correctly if we remove the redundant geometry.
