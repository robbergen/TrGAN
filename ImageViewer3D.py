import ipywidgets as ipyw
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline


class ImageSliceViewer3D:
    """
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(self, volume, v=[0., 0.], figsize=(8, 8), cmap='gray'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        if ((v[0] == 0.) & (v[1] == 0.)):
            self.v = [np.min(volume), np.max(volume)]
        else:
            self.v = v

        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y', 'y-z', 'z-x'], value='x-y',
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z": [1, 2, 0], "z-x": [2, 0, 1], "x-y": [0, 1, 2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1

        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice,
                      z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False,
                                       description='Image Slice:'),
                      zz=ipyw.FloatSlider(min=np.min(self.volume), max=np.max(self.volume), step=0.5,
                                          continuous_update=False, description='Level:'),
                      zzz=ipyw.FloatSlider(min=0.5, max=np.max(np.abs(self.volume)), step=0.5, continuous_update=False,
                                           description='Window:'))

    def plot_slice(self, z, zz, zzz):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:, :, z], cmap=plt.get_cmap(self.cmap),
                   vmin=zz - zzz, vmax=zz + zzz)