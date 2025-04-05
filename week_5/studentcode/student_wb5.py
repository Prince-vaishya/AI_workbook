# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use
        
        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot
        
        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
   # ====> insert your code below here
    # get the data from file into a numpy array


    # create a K-Means cluster model with  the specified number of clusters

    # create a canvas(fig) and axes to hold your visualisation

    # make the visualisation
    # remember to put your user name into the title as specified


    # save it to file as specified

    # if you don't delete the line below there will be problem!
    raise NotImplementedError("Complete the function")
    
    return fig,ax
    
    # <==== insert your code above here
