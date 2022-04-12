#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------
Mass assignment function definitions
------------
"""


#########


def build_grid(cat,cell,box,mas_method,batch_size,wrap):
    """ 
    * cat  is a list of positions x,y,z
    * cell is the cell size in the same units as the positions
    * box  is the dimensions of the box in cells: e.g., (10,20,15) 
    """
    import numpy as np
    import sys

    if len(cat.shape) == 1:
        nobj = len(cat)
        ndim = 1
    else:
        nobj , ndim = cat.shape

        if nobj < ndim:
            cat = cat.T
            nobj , ndim = cat.shape

    if ndim ==1:
        xbins = cell*np.arange(box+1)
        allbins = xbins
    elif ndim ==2:
        xbins = cell*np.arange(box[0]+1)
        ybins = cell*np.arange(box[1]+1)
        allbins = (xbins,ybins)
    elif ndim ==3:
        xbins = cell*np.arange(box[0]+1)
        ybins = cell*np.arange(box[1]+1)
        zbins = cell*np.arange(box[2]+1)
        allbins = (xbins,ybins,zbins)
    else:
        print("Sorry, more than 3 dimensions are not currently supported.")
        print("Aborting now...")
        sys.exit(-1)

    # Maximum size of batches:
    nmax = batch_size
    # Number of batches
    nbat = nobj//nmax
    # Rest
    nres = np.mod(nobj,nmax)

    # Fix bins for histogram; initialize histogram/box
    #if ndim ==1:
    #    hist_bins = np.arange(box+1)
    #elif ndim ==2:
    #    hist_bins = (np.arange(box[0]+1),np.arange(box[1]+1))
    #elif ndim ==3:
    #    hist_bins = (np.arange(box[0]+1),np.arange(box[1]+1),np.arange(box[2]+1))
    #else:
    #    print("Sorry, more than 3 dimensions are not currently supported.")
    #    print("Aborting now...")
    #    sys.exit(-1)

    # initialize grid
    full_grid = np.zeros(box)

    for nb in range(nbat+1):
        print("Processing batch #",nb, "of", nbat)
        if nb <= nbat:
            nstart = nb*nmax
            nend = np.min( ( (nb+1)*nmax , nobj ))
            this_cat = cat[nstart:nend]
            print("Processing objects #",nstart, "to", nend)
        else:
            nstart = nbat*nmax
            this_cat = cat[nstart:]
            print("Processing objects #",nstart, "to end")
        cat_len = len(this_cat)
        # each sub-catalog with nmax objects will give rise to a new catalog with (5**ndim)*nmax objects
        if ndim ==1:
            arr_coord = np.zeros((cat_len,5))
            arr_weigh = np.ones((cat_len,5))
            this_coord = this_cat
            g , s = grid_pos_s(this_coord,cell,box,wrap)
            # These are the coordinates of each dimension
            arr_coord[:,:] = g
            # Update weights
            arr_weigh[:,:] = weights(s,mas_method)

        elif ndim ==2:
            arr_coord = np.zeros((cat_len,5,5,2))
            arr_weigh = np.ones((cat_len,5,5))
            # Do first coordinate
            this_coord = this_cat[:,0]
            this_box = box[0]
            g , s = grid_pos_s(this_coord,cell,this_box,wrap)
            for i in range(5):
                arr_coord[:,:,i,0] = g
                arr_weigh[:,:,i] *= weights(s,mas_method)
            # Do second coordinate
            this_coord = this_cat[:,1]
            this_box = box[1]
            g , s = grid_pos_s(this_coord,cell,this_box,wrap)
            for i in range(5):
                arr_coord[:,i,:,1] = g
                arr_weigh[:,i,:] *= weights(s,mas_method)

        elif ndim ==3:
            arr_coord = np.zeros((cat_len,5,5,5,3))
            arr_weigh = np.ones((cat_len,5,5,5))

            # Do first coordinate
            this_coord = this_cat[:,0]
            this_box = box[0]
            g , s = grid_pos_s(this_coord,cell,this_box,wrap)
            for i in range(5):
                for j in range(5):
                    arr_coord[:,:,i,j,0] = g
                    arr_weigh[:,:,i,j] *= weights(s,mas_method)
            #h0 = np.histogram(arr_coord[:,:,:,:,0].flatten()+0.5,bins=this_box)[0]
            #print("Histogram of x:")
            #print(h0)

            # Do second coordinate
            this_coord = this_cat[:,1]
            this_box = box[1]
            g , s = grid_pos_s(this_coord,cell,this_box,wrap)
            for i in range(5):
                for j in range(5):
                    arr_coord[:,j,:,i,1] = g
                    arr_weigh[:,j,:,i] *= weights(s,mas_method)
            #h1 = np.histogram(arr_coord[:,:,:,:,1].flatten()+0.5,bins=this_box)[0]
            #print("Histogram of y:")
            #print(h1)

            # Do third coordinate
            this_coord = this_cat[:,2]
            this_box = box[2]
            g , s = grid_pos_s(this_coord,cell,this_box,wrap)
            for i in range(5):
                for j in range(5):
                    arr_coord[:,i,j,:,2] = g
                    arr_weigh[:,i,j,:] *= weights(s,mas_method)
            #h2 = np.histogram(arr_coord[:,:,:,:,2].flatten()+0.5,bins=this_box)[0]
            #print("Histogram of z:")
            #print(h2)

        else:
            print("Sorry, more than 3 dimensions are not currently supported.")
            print("Aborting now...")
            sys.exit(-1)
        list_coords = np.reshape(arr_coord,(cat_len*5**ndim,ndim))
        list_weights = np.reshape(arr_weigh,(cat_len*5**ndim))
        # Add weights/clouds to histogram
        #full_grid += np.histogramdd((list_coords + 0.5)*cell, bins=allbins , weights=list_weights)[0]
        full_grid += np.histogramdd(list_coords*cell, bins=allbins , weights=list_weights)[0]

    return full_grid



def grid_pos_s(catlist,cell,box,wrap):
    import numpy as np
    for i in range(4):
        catcell = catlist/cell
        grid_pos = np.outer(catcell,np.ones(5))
        # These are the grid edges
        grid = np.fix(grid_pos)
        grid += np.arange(-2,3)
        # These are the centers of the grid cells
        grid += 0.5
        # distance to grid points
        s = np.abs(grid_pos - grid)
        # wrap around
        #grid = np.mod(np.int0(grid), box)
        if wrap:
            grid = np.mod(grid, box)
        return grid , s



def weights(s,mas_method):
    import numpy as np
    snew = np.copy(s)
    if mas_method == "NGP":
        snew[s<0.5]=1.0
        snew[s>0.5]=0
    elif mas_method == "CIC":
        snew[s<=1.0]=1.0-s[s<=1.0]
        snew[s>1.0]=0
    elif mas_method == "TSC":
        snew[s<=1.5]=0.5*(1.5-s[s<=1.5])**2
        snew[s<0.5]=0.75-s[s<0.5]**2
        snew[s>1.5]=0
    elif mas_method == "PCS":
        snew[s<=2.0]=(1./6)*(2-s[s<=2.0])**3
        snew[s<1.0]=(1./6)*(4 - 6*s[s<1.0]**2 + 3*s[s<1.0]**3)
        snew[s>2.0]=0
    else:
        print("Did not recognize Mass Assignement Function scheme!")
        print("Allowed options (strings): NGP, CIC, TSC and PCS")
        print("Aborting now...")
        sys.exit(-1)
    return snew

