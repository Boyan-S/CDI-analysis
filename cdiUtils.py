import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
from astropy.stats.circstats import *
from pathlib import Path

'''
Oct 09, 2022 - Fixed a bug that caused code to crash when running the old (cohere-scripts-main) reconstructions
'''


'''
Added a bunch of try/except catches in various functions to (hopefully) allow this code to work with 
both the old (cohere-scripts-main) and new(cohere-ui) versions of the cohere package from Argonne
'''




'''
Function that centers phase field of reconstruction around a 0 circular mean
'''
def centerPhase(path, scannum, Nbins=30):
    
    #Load reconstruction, filter by support

    try:

        path_rec = Path(path+str(scannum)+'/results/image.npy')
        rec = np.load(path_rec)
        path_supp = Path(path+str(scannum)+'/results/support.npy')
        support = np.load(path_supp)
        rec = rec*support

    except FileNotFoundError:
        path_rec = Path(path+str(scannum)+'/results_phasing/image.npy')
        rec = np.load(path_rec)
        path_supp = Path(path+str(scannum)+'/results_phasing/support.npy')
        support = np.load(path_supp)
        rec = rec*support

    
    
    #Calculate phase and center circular mean around 0
    phase = np.angle(rec)
    phase = phase[phase != 0]
    hist, bin_edges = np.histogram(phase, bins=Nbins)
    rec = rec*np.exp(-1j*circmean(phase))
    phase = np.angle(rec)
    phase[phase>=np.pi] = 0
    phase = phase[phase != 0]
    
    return phase

'''Returns the maximum of the derivative of the strain for a given phase field'''
def findDislocation(path, scannum):
    
    Nloops = 20
    #Load reconstruction, filter by support
    try:
        path_rec = Path(path+str(scannum)+'/results/image.npy')
        rec = np.load(path_rec)
        path_supp = Path(path+str(scannum)+'/results/support.npy')
        support = np.load(path_supp)
        rec = rec*support

    except FileNotFoundError:
        path_rec = Path(path+str(scannum)+'/results_phasing/image.npy')
        rec = np.load(path_rec)
        path_supp = Path(path+str(scannum)+'/results_phasing/support.npy')
        support = np.load(path_supp)
        rec = rec*support

    phase_old = np.angle(rec*support)
    N_arr = np.empty((Nloops,rec.shape[0]-2, rec.shape[1]-2, rec.shape[2]-2))
    F_arr = np.empty((Nloops,rec.shape[0]-1, rec.shape[1]-1, rec.shape[2]-1))
    for i, phi0 in enumerate(np.linspace(0,np.pi,Nloops)):

        phase_new = phase_old + phi0
        phase = np.where(np.abs(rec)>0,phase_new, phase_old)
        
        Fx = np.angle(np.exp(1j*phase[1::])*np.exp(-1j*phase[:-1:]))
        Fy = np.angle(np.exp(1j*phase[:,1::,:])*np.exp(-1j*phase[:,:-1,:]))
        Fz = np.angle(np.exp(1j*phase[:,:,1::])*np.exp(-1j*phase[:,:,:-1]))

        Fxx = np.angle(np.exp(1j*Fx[1::])*np.exp(-1j*Fx[:-1:]))
        Fxy = np.angle(np.exp(1j*Fx[:,1::,:])*np.exp(-1j*Fx[:,:-1,:]))
        Fxz = np.angle(np.exp(1j*Fx[:,:,1::])*np.exp(-1j*Fx[:,:,:-1]))

        Fyx = np.angle(np.exp(1j*Fy[1::])*np.exp(-1j*Fy[:-1:]))
        Fyy = np.angle(np.exp(1j*Fy[:,1::,:])*np.exp(-1j*Fy[:,:-1,:]))
        Fyz = np.angle(np.exp(1j*Fy[:,:,1::])*np.exp(-1j*Fy[:,:,:-1]))

        Fzx = np.angle(np.exp(1j*Fz[1::])*np.exp(-1j*Fz[:-1:]))
        Fzy = np.angle(np.exp(1j*Fz[:,1::,:])*np.exp(-1j*Fz[:,:-1,:]))
        Fzz = np.angle(np.exp(1j*Fz[:,:,1::])*np.exp(-1j*Fz[:,:,:-1]))


        Fmax = np.maximum(Fx[:,:-1,:-1],Fy[:-1,:,:-1],Fz[:-1,:-1])

        Kx = (Fxx[:,:-1,:-1]*Fxy[:-1,:,:-1]*Fxz[:-1,:-1,:])[:,:-1,:-1]
        Ky = (Fyx[:,:-1,:-1]*Fyy[:-1,:,:-1]*Fyz[:-1,:-1,:])[:-1,:,:-1]
        Kz = (Fzx[:,:-1,:-1]*Fzy[:-1,:,:-1]*Fzz[:-1,:-1,:])[:-1,:-1,:]    

        N_arr[i] = np.maximum(np.abs(Kx), np.abs(Ky), np.abs(Kz))
        F_arr[i] = Fmax
        

#         Fx = np.angle(np.exp(1j*phase[2::])*np.exp(-1j*phase[:-2:]))/2
#         Fy = np.angle(np.exp(1j*phase[:,2::,:])*np.exp(-1j*phase[:,:-2,:]))/2
#         Fz = np.angle(np.exp(1j*phase[:,:,2::])*np.exp(-1j*phase[:,:,:-2]))/2

#         Fxx = np.angle(np.exp(1j*Fx[2::])*np.exp(-1j*Fx[:-2:]))/2
#         Fxy = np.angle(np.exp(1j*Fx[:,2::,:])*np.exp(-1j*Fx[:,:-2,:]))/2
#         Fxz = np.angle(np.exp(1j*Fx[:,:,2::])*np.exp(-1j*Fx[:,:,:-2]))/2

#         Fyx = np.angle(np.exp(1j*Fy[2::])*np.exp(-1j*Fy[:-2:]))/2
#         Fyy = np.angle(np.exp(1j*Fy[:,2::,:])*np.exp(-1j*Fy[:,:-2,:]))/2
#         Fyz = np.angle(np.exp(1j*Fy[:,:,2::])*np.exp(-1j*Fy[:,:,:-2]))/2

#         Fzx = np.angle(np.exp(1j*Fz[2::])*np.exp(-1j*Fz[:-2:]))/2
#         Fzy = np.angle(np.exp(1j*Fz[:,2::,:])*np.exp(-1j*Fz[:,:-2,:]))/2
#         Fzz = np.angle(np.exp(1j*Fz[:,:,2::])*np.exp(-1j*Fz[:,:,:-2]))/2


#         Fmax = np.maximum(Fx[:,:-2,:-2],Fy[:-2,:,:-2],Fz[:-2,:-2])



#         Kx = (Fxx[:,1:-1:,1:-1:]*Fxy[1:-1:,:,1:-1:]*Fxz[1:-1:,1:-1:,:])[:,1:-1:,1:-1:]
#         Ky = (Fyx[:,1:-1:,1:-1:]*Fyy[1:-1:,:,1:-1:]*Fyz[1:-1:,1:-1:,:])[1:-1:,:,1:-1:]
#         Kz = (Fzx[:,1:-1:,1:-1:]*Fzy[1:-1:,:,1:-1:]*Fzz[1:-1:,1:-1:,:])[1:-1:,1:-1:,:]

#         N_arr[i] = np.maximum(np.abs(Kx), np.abs(Ky), np.abs(Kz))
#         F_arr[i] = Fmax

    N = np.min(N_arr, axis=0)
    
    return N, Fmax




'''
This function loads the 3D reciprocal space data from a rocking-curve CDI scan together with the corresponding reconstruction, 
and returns the 3D Phase Retrieval Transfer Function (PRTF), defined as the ratio of the amplitude of the Fourier Transform of the reconstruction 
over the measured intensity.
path - path where the \\results and \\data folders live, as output from the cohere CDI python reconstruction software
GA - whether Generation Algorithm was used in cohere package for reconstructions. If so, loads all generations and averages them for PRTF calculation
'''
def calcPRTF3D(path, scannum, nRecs = None):

    if nRecs==None:
        try:

            path_rec = Path(path+str(scannum)+'/results/image.npy')
            rec = np.load(path_rec)
            path_supp = Path(path+str(scannum)+'/results/support.npy')
            support = np.load(path_supp)
            rec = rec*support


        except FileNotFoundError:
            path_rec = Path(path+str(scannum)+'/results_phasing/image.npy')
            rec = np.load(path_rec)
            path_supp = Path(path+str(scannum)+'/results_phasing/support.npy')
            support = np.load(path_supp)
            rec = rec*support
    else:
        #rec = np.load(path+str(scannum)+'\\results\\'+str(nRecs)+'\\image.npy')

        for i in range(nRecs):
            try:
                if i==0:
                    path = Path(path+str(scannum)+'/results/'+str(i)+'/')


                    rec = np.load(path+'image.npy')/nRecs
                else:
                    rec += np.load(path+'image.npy')/nRecs
                    support = np.load(path+'support.npy')

            except FileNotFoundError:
                if i==0:
                    path = Path(path+str(scannum)+'/results_phasing/'+str(i)+'/')


                    rec = np.load(path+'image.npy')/nRecs
                else:
                    rec += np.load(path+'image.npy')/nRecs
                    support = np.load(path+'support.npy')                
                #rec
            
            
    try:

        path_data = Path(path+str(scannum)+'/data/data.tif')
        dataAmp = tf.imread(path_data).transpose()[::-1,::-1,::] #This is the square root of the measured intensity

    except FileNotFoundError:
        path_data = Path(path+str(scannum)+'/phasing_data/data.tif')
        dataAmp = tf.imread(path_data).transpose()[::-1,::-1,::] #This is the square root of the measured intensity  
    
    #This block calculates the normalization of the PRTF - since rec. amplitude is normalized to 1 while data is not
    FT = np.fft.fftshift(np.fft.fftn(rec)) #Fourier transform of the reconstruction
    FT_phase = np.angle(FT)
    rec_test = np.fft.ifftn(dataAmp*FT_phase)
    norm = np.max(np.abs(rec_test))
    # print(norm)
    
    print('Reconstruction shape is '+str(rec.shape))

    dataAmp[dataAmp < 3] = np.nan
    PRTF_3D = np.abs(FT)/(dataAmp/norm)
    #PRTF_3D = (np.abs(FT)/np.nanmean(np.abs(FT)))/(dataAmp/np.nanmean(dataAmp))
    #PRTF_3D = (np.abs(FT)/dataAmp)
    #PRTF_3D[PRTF_3D == np.inf] = np.nan
    PRTF_3D[PRTF_3D == np.inf] = 0
    #PRTF_3D[PRTF_3D < 1 ] = np.nan
    
    return PRTF_3D

def loadspec(path, scannum):
    specpath_disp = Path(path+str(scannum)+'/conf/config_disp')
    specpath_data = Path(path+str(scannum)+'/conf/config_data')
    
    with open(specpath_disp,'r') as disp:
        line_arr = disp.readlines()
        energy = float(line_arr[1].split()[-1])
        delta = float(line_arr[2].split()[-1])
        gamma = float(line_arr[3].split()[-1])
        detdist = float(line_arr[4].split()[-1])
        theta = float(line_arr[5].split()[-1])
        chi = float(line_arr[6].split()[-1])
        phi = float(line_arr[7].split()[-1])
        scanMot = line_arr[8].split()[-1][1:-1]
        stepSize = float(line_arr[9].split()[-1])
        detector = line_arr[10].split()[-1][1:-1]
        if detector == "34idcTIM2":
            pixelSize = 55e-6 #m
        else:
            raise ValueError('Unknown detector, pixel size undefined!')
        
    with open(specpath_data,'r') as data:
        lines = data.readlines()
        try:
            binning = lines[1][-10:].split()
            binning = np.array([int(binning[0][1]), int(binning[1][0]), int(binning[2][0])])
        except IndexError:
            binning = np.fromstring(lines[1].split()[-1][1:-1], dtype=int, sep=',')
        
    return energy, delta, gamma, detdist, theta, chi, phi, scanMot, stepSize, pixelSize, binning



'''This function calculates the spherically-averaged Phase Retrieval Transfer Function (PRTF).
To do this, it first calculates the location of each data voxel in reciprocal space in the scattering
coordinate system (#2)
    
There are 3 coordinate systems we're working with here:
1. The first one is the 34-ID-C geometry, we use that as reference to write the vectors and other bases in. It goes as follows:
x is along incoming beam; z is vertically up; y is perpendicular to those

2. The second one is the labReciBasis, which is the "scattering" coordinate system:
in it, the z direction is the scattering vector direction, x is perpendicular to the scattering plane, and y is perpendicular to those. 
This basis tells us the scattering geometry

3. The third basis is reciBasis, which is the coordinate system we measure in - x, y are horizontal/vertical directions on detector, while z is +theta

'''
def calcPRTF(path, scannum, nPoints = 31, nRecs=None):


    h = 4.1357e-15 #eV*s
    c = 299792458 #m/s
    print(scannum)

    
    PRTF_3D = calcPRTF3D(path,scannum, nRecs)
    energy, delta, gamma, camDist, theta, chi, phi, scanMot, stepSize, pixelSize, binSize = loadspec(path, scannum)
    wavelength = h*c/(energy*1e3)*1e9  #nm



    k = 2*np.pi/wavelength
    dq1 = k*binSize[0]*pixelSize/(camDist*1e-3)
    dq2 = k*binSize[1]*pixelSize/(camDist*1e-3)
    K0 = k*np.array([1,0,0])
    Kf = k*np.array([np.cos(gamma*np.pi/180)*np.cos(delta*np.pi/180), 
                     np.cos(gamma*np.pi/180)*np.sin(delta*np.pi/180),
                     np.sin(gamma*np.pi/180)])
    H = Kf - K0
    Q1 = dq1*np.array([np.sin(gamma*np.pi/180)*np.cos(delta*np.pi/180),
                       np.sin(gamma*np.pi/180)*np.sin(delta*np.pi/180),
                       -1*np.cos(gamma*np.pi/180)])
    Q2 = dq2*np.array([-1*np.sin(delta*np.pi/180), np.cos(delta*np.pi/180), 0])
    if scanMot == 'th':
        Q3 =binSize[2]*stepSize*np.pi/180*np.cross([H[0],H[1],0],[0,0,1]) #direction of +theta

    elif scanMot == 'phi':
        Q3 = binSize[2]*stepSize*np.pi/180*np.cross([H[1],0,H[3]],[0,-1,0])
    else:
        raise ValueError('Cannot read scan type, not th or phi')

    labReciBasis = np.empty((3,3))
    labReciBasis[:,0] = dq1*np.cross(H, Kf)/np.linalg.norm(np.cross(H,Kf)) #perpendicular to scattering plane
    labReciBasis[:,2] = dq1*H/np.linalg.norm(H) #Along scattering direction
    labReciBasis[:,1] = dq1*np.cross(labReciBasis[:,2],labReciBasis[:,0])/np.linalg.norm(np.cross(labReciBasis[:,2],labReciBasis[:,0])) #perpendicular to other 2



    '''The next line solves for the transformation from the "scattering" (#2) coordinate system to the "measurement" (#3) coordinate system
    Note that [Q1,Q2,Q3] is the transformation from #1 to #3, which is not the same thing
    [Q1,Q2,Q3] @ [unit vectors] = labReciBasis @ [reciBasis vectors]'''
    '''#3 unit vectors written in #1 coord. system = #3 vectors written in #2 coord. system '''
    reciBasis = dq1*np.linalg.inv(labReciBasis) @ np.array([Q1, Q2, Q3])
    q3D = np.zeros(PRTF_3D.shape+(3,))

    #Creates meshgrid of position indices for the data
    Position_x = np.arange(np.size(PRTF_3D,0)) - 0.5*(np.size(PRTF_3D,0)-1)
    Position_y = np.arange(np.size(PRTF_3D,1)) - 0.5*(np.size(PRTF_3D,1)-1)
    Position_z = np.arange(np.size(PRTF_3D,2)) - 0.5*(np.size(PRTF_3D,2)-1)
    Position = np.moveaxis(np.array(np.meshgrid(Position_x, Position_y, Position_z, indexing='ij')),0,3)


    #This calculates the array of locations of each voxel in reciprocal space in #2 coordinate system. 
    q3D = np.moveaxis(np.tensordot(reciBasis,Position,axes=[[1],[3]]),0,3)

    
    
    PRTF = np.zeros(nPoints)
    qMod = np.sqrt(q3D[:,:,:,0]**2 + q3D[:,:,:,1]**2 + q3D[:,:,:,0]**2)
    qMax = 0.5*np.max(qMod)
    qStep = qMax/(nPoints-1)
    qPoints = np.linspace(0, qMax, nPoints)
    for i in range(nPoints):
        qShell = PRTF_3D[np.abs(qMod-qPoints[i]) < qStep]
        PRTF[i] = np.nanmean(qShell)



    #determine resolution
    Y = np.nanmin(PRTF)
    if Y<0.5:
#         resPRTF = 2*np.pi/np.mean(qPoints[np.abs(PRTF-0.5)<0.05])
        #Linear interpolation to determine resolution; qPoints[PRTF<0.5][0] is first point where <0.5; qPoints[ndx] is point right before that
        ndx = np.argmin(qPoints[PRTF>0.5]-0.5)
        resPRTF = 2*np.pi/(qPoints[PRTF<0.5][0] - (qPoints[PRTF<0.5][0]-qPoints[ndx])*(0.5-PRTF[PRTF<0.5][0])/(PRTF[ndx] - PRTF[PRTF<0.5][0]))
    else:
        resPRTF = 2*np.pi/qMax


    print('Resolution is '+str(resPRTF)+' nm')
    fig = plt.figure(figsize = (8,6))
    plt.plot(qPoints,PRTF)
    plt.plot(qPoints, 0.5*np.ones(nPoints), 'k--')
    try:
        plt.ylim(0,np.max(PRTF)+0.2)
        plt.xlabel('Q [rad/nm]')
        plt.ylabel('PRTF')
        plt.title('Scan '+str(scannum) + '\n' + 'Resolution = '+str(int(resPRTF))+' nm')
    except ValueError:
        print('resolution is nan')
    plt.savefig('Scan_'+str(scannum)+'_PRTF')
    plt.show()
    return qPoints, PRTF

def CompareSlice(path, scannum, sliceNum=35, log=False, nRecs=None):
    #FT of reconstruction
    
    if nRecs==None:
        try:

            path_rec = Path(path+str(scannum)+'/results/image.npy')
            path_supp = Path(path+str(scannum)+'/results/support.npy')

            rec = np.load(path_rec)
            support = np.load(path_supp)
            rec = rec*support

        except FileNotFoundError:
            path_rec = Path(path+str(scannum)+'/results_phasing/image.npy')
            path_supp = Path(path+str(scannum)+'/results_phasing/support.npy')

            rec = np.load(path_rec)
            support = np.load(path_supp)
            rec = rec*support
    else:
        for i in range(nRecs):
            rec = 0
            try:
                path_rec = Path(path+str(scannum)+'/results/'+str(i)+'/image.npy')
                rec += np.load(path_rec)/nRecs
            except FileNotFoundError:
                path_rec = Path(path+str(scannum)+'/results_phasing/'+str(i)+'/image.npy')
                rec += np.load(path_rec)/nRecs
            # if i==0:



            #     rec = np.load(path+str(scannum)+'\\results\\'+str(i)+'\\image.npy')/nRecs
            # else:
            #     rec += np.load(path+str(scannum)+'\\results\\'+str(i)+'\\image.npy')/nRecs
                
    #rec = np.load(path+str(scannum)+'\\results\\image.npy') #Loads reconstruction
    try:

        path_data = Path(path+str(scannum)+'/data/data.tif')
        dataAmp = tf.imread(path_data).transpose()[::-1,::-1,::] #This is the square root of the measured intensity

    except FileNotFoundError:
        path_data = Path(path+str(scannum)+'/phasing_data/data.tif')
        dataAmp = tf.imread(path_data).transpose()[::-1,::-1,::] #This is the square root of the measured intensity        

    FT = np.fft.fftshift(np.fft.fftn(rec))
    
#     FT_phase = np.angle(FT)
#     rec_test = np.fft.ifftn(dataAmp*FT_phase)
#     norm = np.max(np.abs(rec_test))
    #print(norm)
    
    
    fig1 = plt.figure()
    #plt.title('Sqrt[data]')
    plt.title('Measured amplitude')
    #plt.imshow(np.sqrt(data)[::,::,sliceNum])
    if log:
        
        plt.imshow(np.log10(dataAmp[::,::,sliceNum]))
    else:
        plt.imshow(dataAmp[::,::,sliceNum], vmax=np.max(dataAmp[::,::,sliceNum])/1.5)
    plt.colorbar()
    plt.show()
    
    fig2 = plt.figure()
    plt.title('|FT| of reconstruction')
    if log:
        plt.imshow(np.log10(np.abs(FT[::,::,sliceNum])))
    else:
        plt.imshow(np.abs(FT[::,::,sliceNum]), vmax=np.max(np.abs(FT[::,::,sliceNum]))/1.5)
    plt.colorbar()
    plt.show()
    

def binning(array, binsizes):
    """
    This function does the binning of the array. The array is binned in each dimension by the corresponding binsizes elements.
    If binsizes list is shorter than the array dimensions, the remaining dimensions are not binned.
    
    Parameters
    ----------
    array : ndarray
        the original array to be binned
    binsizes : list
        a list defining binning factors for corresponding dimensions
        
    Returns
    -------
    binned_array : ndarray
        binned array
    """

    data_dims = array.shape
    # trim array
    for ax in range(len(binsizes)):
        cut_slices = range(data_dims[ax] - data_dims[ax] % binsizes[ax], data_dims[ax])
        array = np.delete(array, cut_slices, ax)

    binned_array = array
    new_shape = list(array.shape)

    for ax in range(len(binsizes)):
        if binsizes[ax] > 1:
            new_shape[ax] = binsizes[ax]
            new_shape.insert(ax, int(array.shape[ax] / binsizes[ax]))
            binned_array = np.reshape(binned_array, tuple(new_shape))
            binned_array = np.sum(binned_array, axis=ax + 1)
            new_shape = list(binned_array.shape)
    return binned_array