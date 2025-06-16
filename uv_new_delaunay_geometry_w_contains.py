from skimage import measure
import numpy as np 
import os   
import PyTexturePacker 
import plistlib 
from PIL import Image
import trimesh  
import re       
import pandas as pd
from trimesh.exchange.ply import _parse_header
from scipy import stats as st
import cv2
import imageio
import glob
import time
from scipy.spatial import Delaunay
from argparse import ArgumentParser
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

parser = ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

paths = []
timings = []

for path_in in sorted(glob.glob(os.path.join(args.path,'*inpaint.ply'))):
    # if 'ascheberg' not in path_in: continue
    path_out = path_in[:-4]+'.glb'

    if os.path.exists(path_out):
        print(f'skipping {path_out}: file exists')
        continue

    print(path_out)

    start_time = time.time()

    # load data from .ply file
    with open(path_in, 'rb') as f:
        print(f'reading PLY header from {path_in}')
        elements, is_ascii, image_name = _parse_header(f)
        assert is_ascii
        print(f'loading vertex data')
        vertex_data = np.loadtxt(f, max_rows=elements['vertex']['length'])
        print(f'loading face data')
        face_data = np.loadtxt(f, dtype='int32')[:, 1:]
        
    # Extract vertex data
    xyz = vertex_data[:, :3]                  # 3D position [[0, 0 ,0] [0,1,0] [1, 1, 1]]
    rgb = vertex_data[:, 3:6].astype('uint8') # color
    idx = vertex_data[:, 6].astype('int32')   # layer index
    pix = vertex_data[:, 7:9].astype('int32') # 2D pixel location in panorama (row, col order)

    H = pix[:,0].max()+1
    W = pix[:,1].max()+1
    print(f'panorama dimensions: {H}x{W}')


    # Ensure all vertices of a face are in the same layer
    # Initialize lists to hold the new vertices and their corresponding attributes
    new_xyz = []
    new_rgb = []
    new_idx = []
    new_pix = []
    new_face_data = face_data.copy()

    same_face = np.all(idx[face_data]==idx[face_data][:,0:1],axis=1)

    next_vertex = len(xyz)

    for i in np.where(~same_face)[0]:
        # what is face here?
        face = face_data[i]
        layers = idx[face]
        y = pix[face][:,0]
        x = pix[face][:,1]

        # Check if vertices are in different layers
        if len(np.unique(layers)) > 1:
            # move everything to furthest layer
            target_layer = np.max(layers)

            for j, vertex in enumerate(face):
                vertex_layer = layers[j]
                if vertex_layer != target_layer:
                    new_xyz.append(xyz[vertex])
                    new_rgb.append(rgb[vertex])
                    new_idx.append(target_layer)
                    new_pix.append(pix[vertex])
                    
                    # remap vertex reference
                    new_face_data[i, j] = next_vertex
                    next_vertex += 1
                    
                    row,col = pix[vertex]

    if len(new_xyz)>0:
        new_xyz = np.array(new_xyz)
        new_rgb = np.array(new_rgb)
        new_idx = np.array(new_idx)
        new_pix = np.array(new_pix)
        
        
        # Extending original arrays
        xyz = np.vstack((xyz, new_xyz))
        rgb = np.vstack((rgb, new_rgb))
        idx = np.concatenate((idx, new_idx))
        pix = np.vstack((pix, new_pix))
        
        face_data = new_face_data

    # Check for faces that wrap around sphere
    on_right_edge = np.any(pix[face_data][...,1]==W-1,axis=1)
    on_left_edge = np.any(pix[face_data][...,1]==0,axis=1)


    new_xyz = []
    new_rgb = []
    new_idx = []
    new_pix = []
    new_face_data = face_data.copy()

    for i in np.where(on_right_edge & on_left_edge)[0]:
        face = face_data[i]
        x = pix[face][...,1]

        for j, vertex in enumerate(face):
            if x[j] == 0:
                new_xyz.append(xyz[vertex])
                new_rgb.append(rgb[vertex])
                new_idx.append(idx[vertex])

                my_pix = pix[vertex].copy()
                my_pix[1] = W
                
                new_pix.append(my_pix)
                
                new_vertex_index = len(xyz) + len(new_xyz) - 1

                # remap vertex reference
                new_face_data[i, j] = new_vertex_index

    if len(new_xyz)>0:
        new_xyz = np.array(new_xyz)
        new_rgb = np.array(new_rgb)
        new_idx = np.array(new_idx)
        new_pix = np.array(new_pix)
        
        
        # Extending original arrays
        xyz = np.vstack((xyz, new_xyz))
        rgb = np.vstack((rgb, new_rgb))
        idx = np.concatenate((idx, new_idx))
        pix = np.vstack((pix, new_pix))
        
        face_data = new_face_data

    # INTERSECTION TEST: getting OLD layer 0 bounding polygon
    # --------------------------------------
    # layer_10_mask = np.all(idx[face_data] == 10, axis=1)
    # old_faces_layer_10 = face_data[layer_10_mask]

    # # get all old triangles from layer 0
    # with open("triangles_halde_outer_polygon_layer_10.txt", "w") as f:
    #     for face in old_faces_layer_10:
    #         v0, v1, v2 = face[0], face[1], face[2]
    #         triangle = [pix[v0].tolist(), pix[v1].tolist(), pix[v2].tolist()]
    #         f.write(f"{triangle}\n")

    # ---------------------------------------
    
    print("Making new geometry with Delaunay triangulation...")
    # get the first occurence of each 2d pixel in the Delaunay triangulation and map it to a 3d point
    # this is an attempt to remove the complication/distorted geometry of the mesh

    # need to keep track of the indexes of each layer/flatten the layered array
    num_layers = idx.max() + 1
    new_faces = []
    new_xyz = []
    new_pix = []
    new_rgb = []
    new_idx = []

    # here, go through each layer
    for i in range(num_layers):
        layer_pix = pix[idx == i]
        layer_xyz = xyz[idx == i]
        layer_rgb = rgb[idx == i]

        # to store layer pixels to perform Delaunay triangulation on
        unique_layer_pix = []
        # get current length of new_pix to see how much we need to offset for Delaunay triangulation index (for faces)
        offset = len(new_pix)

        # Use np.unique to find unique 2D pixel coordinates
        unique_pix, unique_indices = np.unique(layer_pix, axis=0, return_index=True)

        # Get corresponding data
        unique_xyz = layer_xyz[unique_indices]
        unique_rgb = layer_rgb[unique_indices]

        for j in range(len(unique_pix)):
            # calculate the new xyz coord using the old xyz's euclidean distance and the 2d pixel coord
            # 2d pixel coord info
            # x, y = unique_pix[j]
            
            old_x, old_y, old_z = unique_xyz[j]
            # calculate the euclidean distance (r)
            # r = np.sqrt(old_x ** 2 + old_y ** 2 + old_z ** 2)

            # solve for the theta and phi using eqns 8, 9 from panorama projections paper
            # x and y is the 2d image coordinate
            # x = w((theta/(2pi)) + 1/2)
            # y = h((phi/pi) + 1/2)
            # theta = ((x / W) - 0.5) * 2 * np.pi
            # phi = ((y / H) - 0.5) * np.pi

            # use theta and phi to solve for new x,y,z and append this to the new list
            # eqns 10-12 from panorama projections paper
            # new_x = r * np.sin(theta) * np.cos(phi)
            # new_y = r * np.sin(phi)
            # new_z = r * np.cos(theta) * np.cos(phi)

            # append info (new xyz coord, pix, rgb, idx) to all new lists
            # new_xyz.append([new_x, new_y, new_z])
            new_xyz.append([ old_x, old_y, old_z ]) # use original x, y, z coords
            new_pix.append(unique_pix[j])
            new_rgb.append(unique_rgb[j])
            new_idx.append(i)


        if len(unique_pix) < 3:
            # cannot triangulate if less than 3 points in a layer
            continue

        # do delauney triangulation for all 2d vertixes (pix) on the layer
        tri = Delaunay(unique_pix)

        # check to make sure tri.points and given array is the same
        if np.array_equal(tri.points, unique_pix):
            print("Finished Delaunay triangulation for layer ", i)
        else:
            print("Delaunay point list is DIFFERENT")
            print("layer points: ", unique_pix)
            print("tri.points: ", tri.points)

        # INTERSECTION TEST: getting NEW layer 10 triangles
        # ------------------------------------------------
        # if i == 10:
        #     with open("halde_new_edges_layer_10_pix.txt", "w") as f:
        #         for face in tri.simplices:
        #             v0, v1, v2 = face[0] + offset, face[1] + offset, face[2] + offset
        #             triangle = [new_pix[v0].tolist(), new_pix[v1].tolist(), new_pix[v2].tolist()]
        #             f.write(f"{triangle}\n")
        # ------------------------------------------------

        # tri.simplices is a list of all the triangles from Delaunay triangulation
        # simplex[0] is the index of first point of the triangle/face in tri.points
        # use tri.points to get p1 - the actual pix 2d coord from the points list
        # use p1 as a tuple to retrieve the dictionary value of its euclidean distance
        
        # TODO: modify this number
        # we should only run contains test on later/inner layers

        print(f"# of Delaunay triangles in layer {i}: ", len(tri.simplices))

        if i > 0:
            # create a multipolygon of the original geometry to compare to
            # get old geometry for this layer
            layer_mask = np.all(idx[face_data] == i, axis=1)
            old_faces_layer_i = face_data[layer_mask]

            # for each face, convert it to a Polygon
            old_triangles = []
            for f in old_faces_layer_i:
                # make edges
                coords = [tuple(pix[v]) for v in f]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])  # close the polygon if needed
                # convert edges to polygon
                old_triangle = Polygon(coords)
                old_triangles.append(old_triangle)

            old_multi = unary_union(old_triangles)  
        
            count = 0

            for tri in tri.simplices:
                triangle_coords = unique_pix[tri] 
                triangle = Polygon(triangle_coords)

                # contains test - if contained...
                if old_multi.contains(triangle):
                    count += 1
                    # create face
                    face = []
                    
                    for tri_pt in tri:
                        # add offset from prev layers to indices for face connectivity (so they are accurate when we use them for new_pix)
                        face.append(tri_pt + offset)
                
                    # add this face to new_faces
                    new_faces.append(face)

            print("# of Delaunay triangles AFTER contains test: ", count)
        else:
            # early layers that we don't need to test for contains
            for tri in tri.simplices:
                face = []

                for tri_pt in tri:
                    # add offset from prev layers to indices for face connectivity (so they are accurate when we use them for new_pix)
                    face.append(tri_pt + offset)
                
                new_faces.append(face)

    # convert them all to numpy arrays
    new_faces = np.array(new_faces)
    new_pix = np.array(new_pix)
    new_xyz = np.array(new_xyz)
    new_rgb = np.array(new_rgb)
    new_idx = np.array(new_idx)

    print("Delaunay results")

    print("Faces: ", new_faces)
    print("Pixel coords: ", new_pix)
    print("XYZ coords: ", new_xyz)
    print("RGB:", new_rgb)
    print("idx:", new_idx)

    # go onto uv mapping

    # UV mapping point with updated pixel data
    # Initialize dataframe to store pixel information
    pixel_info = pd.DataFrame({
        'row': new_pix[:, 0],
        'col': new_pix[:, 1],
        'layer': new_idx
    })

    output_folder = "./connected_comps"
    os.system(f'rm -rf {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    #uv_folder = './uv_images/'
    #os.system(f'rm -rf /tmp/uv_images/*')
    #os.makedirs(uv_folder, exist_ok=True)

    # Create binary maps layer by layer
    num_layers = new_idx.max() + 1
    blob_info = []
    for i in range(num_layers):
        print(f'processing layer {i}...')
        sel = (new_idx == i)  # sel is a binary mask

        uv_map = np.zeros((H, W+1, 4), dtype='uint8')
        # update the uvmap to use the new pixel and new rgb data
        uv_map[new_pix[sel, 0], new_pix[sel, 1], :3] = new_rgb[sel]
        uv_map[new_pix[sel, 0], new_pix[sel, 1], 3] = 255

        #print(f'saving uv map {i}')
        #plt.imsave(f'./uv_images/uv_map_{i}.png', uv_map)

        binary_map = np.zeros((H, W+1), dtype='bool')
        binary_map[new_pix[sel, 0], new_pix[sel, 1]] = True

        # Label blobs and extract properties
        blobs_labels = measure.label(binary_map)
        regions = measure.regionprops(blobs_labels)

        # Update dataframe for every region in the layer
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            blob_label = region.label

            # Identify pixels that belong to this blob
            blob_pixels = np.argwhere(blobs_labels == blob_label)

            # Find each pixel and populate with info about the blob it's in
            for pixel in blob_pixels:
                r, c = pixel
                blob_info.append({
                    'row': r,
                    'col': c,
                    'layer': i,
                    'minr': minr,
                    'minc': minc,
                    'maxr': maxr,
                    'maxc': maxc,
                    'blob': blob_label
                })

            # Save images to put into texture packer
            cropped_image = uv_map[minr:maxr, minc:maxc]
            cropped_image = np.ascontiguousarray(cropped_image)
            
            cropped_image = cv2.inpaint(cropped_image[...,:3],255-cropped_image[:,:,3],3,cv2.INPAINT_NS)
            
            filename_full = os.path.join(output_folder, f'blob_{blob_label}_layer_{i}_{minr}_{minc}_{maxr}_{maxc}.png')
            imageio.imwrite(filename_full, cropped_image)

    blob_info = pd.DataFrame(blob_info)
    #blob_info.to_csv('blob_info.csv')
    #pixel_info.to_csv('pixel_info.csv')

    print('merging pixel info with blob info')
    pixel_info = pixel_info.merge(blob_info, on=['row', 'col', 'layer'], how='left')

    print('texture packing')
    packer = PyTexturePacker.Packer.create(
        max_width=16384, max_height=16384,
        enable_rotated=False, bg_color=0xffffffff, reduce_border_artifacts=True, inner_padding=10, border_padding=0, shape_padding=0
    )
    packer.pack(output_folder, "cc_map_%d")

    # texture_map_image = imageio.imread("cc_map_0.png")

    # Set total texture pack dimensions
    with open("cc_map_0.plist", "rb") as pf:
        metadata = plistlib.load(pf)
        size = metadata["metadata"]["size"].strip("{}").split(",")
        TPACK_H = float(size[1])
        TPACK_W = float(size[0])

    print('reading texture pack result')
    # Map blobs to texture rects for quicker lookup
    texture_rects = []
    for region in metadata["frames"]:
        frame = metadata["frames"][region]
        texture_rect = frame["frame"]
        match = re.search(r'\{\{(\d+),(\d+)\}', texture_rect)
        if match:
            texture_rect = [int(match.group(1)), int(match.group(2))]
            texture_rects.append({
                'filename': region,
                'tx': texture_rect[0],
                'ty': texture_rect[1]
            })
    texture_rects = pd.DataFrame(texture_rects)
    pixel_info['filename'] = pixel_info.apply(
        lambda row: f'blob_{row["blob"]}_layer_{row["layer"]}_{row["minr"]}_{row["minc"]}_{row["maxr"]}_{row["maxc"]}.png',
        axis=1
    )
    pixel_info = pixel_info.merge(texture_rects, on='filename', how='left')
    pixel_info['u'] = pixel_info['tx'] + (pixel_info['col'] - pixel_info['minc'])
    pixel_info['v'] = (TPACK_H-1) - (pixel_info['ty'] + (pixel_info['row'] - pixel_info['minr']))
    pixel_info['u'] = pixel_info['u'].astype('float64')
    pixel_info['v'] = pixel_info['v'].astype('float64')

    pixel_info['u'] = (pixel_info['u'] + 0.5) / TPACK_W
    pixel_info['v'] = (pixel_info['v'] + 0.5) / TPACK_H
    #pixel_info.to_csv('pixel_info.csv')

    uv_coordinates = pixel_info[['u','v']].to_numpy()
    texture_map_image = Image.open("cc_map_0.png")

    # Make the mesh and export
    print('making output mesh')
    visuals = trimesh.visual.TextureVisuals(uv=uv_coordinates, image=texture_map_image)
    mesh = trimesh.Trimesh(vertices=new_xyz, faces=new_faces, visual=visuals)

    print(f'writing {path_out}')
    res = mesh.export(path_out)

    end_time = time.time()

    print(f'completed {path_out} in {end_time-start_time} s')

    paths.append(path_in)
    timings.append(end_time-start_time)

df = pd.DataFrame({'path':paths,'timing':timings})
print(df)