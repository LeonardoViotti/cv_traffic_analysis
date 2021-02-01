
#------------------------------------------------------------------------------
# Draft homography 

# https://towardsdatascience.com/a-social-distancing-detector-using-a-tensorflow-object-detection-model-python-and-opencv-4450a431238


# TODO
#   - Function that display points in video frame and sat side by side
#   - Do first with a conflcit trajectory to test.
#   - Have it flexible enough that can be just applied to the entire results df
#   - Function to plot one or to trajectories to assert it is working


#------------------------------------------------------------------------------
# ???


# def compute_perspective_transform(corner_points,width,height,image):
# 	""" Compute the transformation matrix
# 	@ corner_points : 4 corner points selected from the image
# 	@ height, width : size of the image
# 	return : transformation matrix and the transformed image
# 	"""
# 	# Create an array out of the 4 corner points
# 	corner_points_array = np.float32(corner_points)
# 	# Create an array with the parameters (the dimensions) required to build the matrix
# 	img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
# 	# Compute and return the transformation matrix
# 	matrix = cv2.getPerspectiveTransform(corner_points_array,img_params) 
# 	img_transformed = cv2.warpPerspective(image,matrix,(width,height))
# 	return matrix,img_transformed

# # p0, p1, p2, p3 = (231, 34), (13, 137), (413, 330), (477, 81)
# # mt, img_plane = compute_perspective_transform([p0, p3, p1, p2],  img_0.shape[0],  img_0.shape[1], img_0)


# # Order has to be top-left, top-right, bottom-left, bottom-right
# p0, p1, p2, p3 = (231, 34), (477, 81), (13, 137), (413, 330)
# # drawCentroid(img_0, p4)
# # stable_show(img_0)

# mt, img_plane = compute_perspective_transform([p0, p1, p2, p3],  img_0.shape[0],  img_0.shape[1], img_0)
# ishow(img_plane)

# # Plot all points
# img = cp.deepcopy(img_0)


# cv2.circle(img, p2, 2, (255, 0, 255), -1)




#------------------------------------------------------------------------------
# Draft homography (2)

# https://medium.com/hal24k-techblog/how-to-track-objects-in-the-real-world-with-tensorflow-sort-and-opencv-a64d9564ccb1

class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert lonlat_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T

#------------------------------------------------------------------------------
# Coordinates mapping 

# Create one instance of PixelMapper to convert video frames to coordinates
quad_coords = {
    "lonlat": np.array([
        [9.035743, 38.853931], #  top right
        [9.035832, 38.853546], #  top left
        [9.035769, 38.853273], # bottom left
        [9.035634, 38.853421] #  bottom right
    ]),
    "pixel": np.array([
        [452, 37], # top right
        [259, 36], #  top left
        [53, 115], #  bottom left
        [404, 144] # bottom right
    ])
}

pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

# Create another instance to map sat image to coordinates
quad_coords_sat = {
    "lonlat": np.array([
        [9.035743, 38.853931], #  top right
        [9.035832, 38.853546], #  top left
        [9.035769, 38.853273], # bottom left
        [9.035634, 38.853421] #  bottom right
    ]),
    "pixel": np.array([
        [324, 149], # top right
        [209, 125], #  top left
        [131, 141], #  bottom left
        [161, 182] # bottom right
    ])
}

pm_sat = PixelMapper(quad_coords_sat["pixel"], quad_coords_sat["lonlat"])


#------------------------------------------------------------------------------
# Test a trajectory on image and longlat

# Anotate trajectory on initial video frame
img_0_conflict = img_0.copy()
# draw_trajectories(img_0_conflict, t_car)
ishow(img_0_conflict)

# Transform trajectory into long lat
# t_car_long_lat = pm.pixel_to_lonlat(t_car[0]) # t_car created in draft-intersections.py

# Load sat image
img_0_lat_long = cv2.imread('../data/2-sat.jpg')
ishow(img_0_lat_long)

# Transform lat long trajectory into pixels of sat image
# t_car_satpixels = pm_sat.lonlat_to_pixel(t_car_long_lat).astype(int)

# Anotate trajectory on sat image
draw_trajectories(img_0_lat_long, t_car_satpixels)
ishow(img_0_lat_long)

def draw_trajectories(img, trajectory_array):
	# if len(trajectory_array.shape) < 3:
    if len(trajectory_array.shape) < 3:
        for p in range(1, len(trajectory_array)):
            cv2.line(img, tuple(trajectory_array[p-1]), tuple(trajectory_array[p]), (0, 0, 255), 2)