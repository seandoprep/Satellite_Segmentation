import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import fiona
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

from collections import defaultdict
from shapely import MultiLineString
from shapely.geometry import Polygon, MultiPolygon, mapping
from netCDF4 import Dataset
from fiona.crs import from_epsg


def save_nc(save_path : str, prediction_array : np.array, lat_grid : np.array, lon_grid : np.array) :
    '''
    Convert prediction mask numpy array to nc file format data.

    save_path : Path where nc file will be saved
    original_array_path : Original nc file for geocoordinate information
    prediction_array : Numpy array that needs to be converted
    '''
    height, width = prediction_array.shape

    # Set NC file
    ncfile = Dataset(save_path, mode='w', format='NETCDF4') 
    lat_dim = ncfile.createDimension('lat', height) # latitude axis
    lon_dim = ncfile.createDimension('lon', width) # longitude axis

    # Title of NC file
    ncfile.title='inference_result'

    # Latitude 
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'

    # Longitude 
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'

    # Define a 3D variable to hold the data
    inf_result = ncfile.createVariable('prediction',np.int64,('lat','lon')) # Note: unlimited dimension is leftmost

    # Add Coordinate Information
    lat[:] = lat_grid 
    lon[:] = lon_grid
    inf_result[:,:] = prediction_array
    
    ncfile.close()


def label_binary_image(binary_array : np.array):
    """Label mask ndarray(binarized image) for grouping pixel"""
    height, width = binary_array.shape
    labeled_image = [[0 for _ in range(width)] for _ in range(height)]
    label = 1

    # 8-directional movement offsets(5 x 5)
    offsets = [(-2,-2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
               (-1,-2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
               (0,-2),  (0, -1),           (0, 1),  (0, 2),
               (1,-2),  (1, -1),  (1, 0),  (1, 1),  (1, 2),
               (2,-2),  (2, -1),  (2, 0),  (2, 1),  (2, 2)]
    
    # Function to check if a pixel is within the image bounds
    def is_valid_pixel(x, y):
        return 0 <= x < width and 0 <= y < height

    # Function to check if a pixel has already been labeled
    def is_labeled(x, y):
        return labeled_image[y][x] != 0

    # Function to get all unlabeled neighboring pixels
    def get_unlabeled_neighbors(x, y):
        neighbors = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if is_valid_pixel(nx, ny) and not is_labeled(nx, ny) and binary_array[ny][nx] == 1:
                neighbors.append((nx, ny))
        return neighbors

    # Function to perform label propagation from a seed pixel
    def propagate_label(seed_x, seed_y):
        stack = [(seed_x, seed_y)]
        while stack:
            x, y = stack.pop()
            labeled_image[y][x] = label
            neighbors = get_unlabeled_neighbors(x, y)
            stack.extend(neighbors)

    # Main loop to label connected components in the binary image
    for y in range(height):
        for x in range(width):
            if binary_array[y][x] == 1 and not is_labeled(x, y):
                propagate_label(x, y)
                label += 1

    return np.array(labeled_image)


def mask_to_boundary(mask, output_shapefile, lon_grid, lat_grid, min_area=20.):
    """Convert a mask ndarray(binarized image) to MultiLineString"""
    # Find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiLineString()
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # Create actual polygons filtering by area (removes artifacts)
    boundaries = ()
    for idx, cnt in enumerate(contours):
        coords = ()
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            for point in cnt:
                coord = (lon_grid[point[0][0]-1], lat_grid[point[0][1]-1])
                coords = coords + (coord,)
            boundaries = boundaries + (coords,)

    # Save into MultiLineString type data
    schema = {
        'geometry': 'MultiLineString',
        'properties' : {}
    }

    with fiona.open(output_shapefile, 'w', 'ESRI Shapefile', schema, crs = "EPSG:4326") as output:
        output.write({
            'geometry': {'type' : 'MultiLineString', 'coordinates' : boundaries},
            'properties' : {}
        })

    return 


def mask_to_area(mask, output_shapefile, lon_grid, lat_grid, min_area=20.):
    """Convert a mask ndarray (binarized image) to MultiPolygon"""
    # Find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # Create actual polygons filtering by area (removes artifacts)
    polygons = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if idx not in child_contours and area >= min_area:
            coords = []
            for point in cnt:
                coord = (lon_grid[point[0][0]-1], lat_grid[point[0][1]-1])
                coords.append(coord)
            polygon = Polygon(coords)
            if polygon.area >= min_area:
                # Check for holes (children contours)
                holes = []
                for child in cnt_children.get(idx, []):
                    hole_coords = []
                    for point in child:
                        coord = (lon_grid[point[0][0]-1], lat_grid[point[0][1]-1])
                        hole_coords.append(coord)
                    holes.append(hole_coords)
                if holes:
                    polygon = Polygon(shell=coords, holes=holes)
                polygons.append(polygon)

    multi_polygon = MultiPolygon(polygons)

    # Save into MultiPolygon type data
    schema = {
        'geometry': 'MultiPolygon',
        'properties': {}
    }

    with fiona.open(output_shapefile, 'w', driver='ESRI Shapefile', schema=schema, crs=from_epsg(4326)) as output:
        output.write({
            'geometry': mapping(multi_polygon),
            'properties': {}
        })

    return multi_polygon


def polygons_from_hexbins(collection):
    """Convert Matploylib PolyCollection data to Polygons"""
    hex_polys = collection.get_paths()[0].vertices
    hex_array = []
    for xs,ys in collection.get_offsets():
        hex_x = np.add(hex_polys[:,0],  xs)
        hex_y = np.add(hex_polys[:,1],  ys)
        hex_array.append(Polygon(np.vstack([hex_x, hex_y]).T))
        
    color = collection.get_array()
    cmap = plt.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=color.min(), vmax=color.max())
    hex_colors = [matplotlib.colors.to_hex(cmap(norm(value))) for value in color]
    
    return gpd.GeoDataFrame({'colors': color, 'hex_colors': hex_colors, 'geometry': hex_array})


def mask_to_hexagon(inference_output, output_path, grid_size, bins, mincnt, alpha):
    """Convert a mask ndarray(binarized image) to Hexbin plot(Polygons)"""
    flipped_inference_output = np.flipud(inference_output)
    aquaculture = np.where(flipped_inference_output == 1)

    hb = plt.hexbin(aquaculture[1], aquaculture[0], 
            gridsize=grid_size, 
            cmap=plt.cm.viridis_r, 
            bins = bins, 
            mincnt = mincnt,
            alpha=alpha)

    hex_gdf = polygons_from_hexbins(hb)
    hex_gdf.to_file(output_path)