import numpy as np
import scipy as sp
from PIL import Image
import PIL.ImageFilter as ftr
from numpy.matlib import repmat
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
import sklearn.neighbors
from scipy import ndimage
from numpy.lib.stride_tricks import as_strided

#from compiler.ast import flatten

def arrary_filter(a,windowSize):
    temp_image = a * 255
    meanImage = Image.fromarray(temp_image.astype(np.uint8)).filter(ftr.Kernel((windowSize, windowSize), np.ones(windowSize*windowSize), windowSize*windowSize))
    meanImage = np.array(meanImage)
    return meanImage

def boxfilter(a,windowSize):
    size=windowSize*windowSize
    radius=(windowSize-1)/2
    tmp=a.copy()
    result=a.shape
    m=result[0]
    n=result[1]
    radius=int(radius)
    if(len(result)!=3):
        for i in range(radius,m-radius):
            for j in range(radius,n-radius):
                sum=np.sum(tmp[i-radius:i+radius+1,j-radius:j+radius+1])
                a[i,j]=sum/size
    else:
        for k in range(3):
            for i in range(radius, m - radius):
                for j in range(radius, n - radius):
                    sum = np.sum(tmp[i - radius:i + radius+1, j - radius:j + radius+1,k])
                    a[i, j,k] = sum / size
    return a

def localRGBnormalDistributions(image, windowRadius=1, epsilon=1e-8):
    result= image.shape
    h=result[0]
    w=result[1]
    N = h * w
    windowSize = 2 * windowRadius + 1

    #meanImage = imboxfilt(image, windowSize)
    #meanImage = boxfilter(image, windowSize)
    meanImage = np.zeros(image.shape)
    meanImage[:,:,0] = ndimage.uniform_filter(image[:,:,0], windowSize)
    meanImage[:, :, 1] = ndimage.uniform_filter(image[:, :, 1], windowSize)
    meanImage[:, :, 2] = ndimage.uniform_filter(image[:, :, 2], windowSize)
    # temp_image=image*255
    # meanImage= Image.fromarray(temp_image.astype(np.uint8)).filter(ftr.Kernel((3,3),(1,1,1,1,1,1,1,1,1),9))
    # meanImage=np.array(meanImage)
    covarMat = np.zeros((3, 3, N))

    for r in range(0,3):
        for c in range(r,3):
            # temp_image = image[:,:,r]*image[:,:,c]*255
            # temp_image = Image.fromarray(temp_image.astype(np.uint8)).filter(ftr.Kernel((3, 3), (1, 1, 1, 1, 1, 1, 1, 1, 1), 1/9))
            # temp = np.array(temp_image)-meanImage[:,:,r]*meanImage[:,:,c]
            temp=ndimage.uniform_filter(image[:,:,r]*image[:,:,c],windowSize)-meanImage[:,:, r]*meanImage[:,:, c]
            #temp = imboxfilt(image(:,:, r).*image(:,:, c), windowSize) - meanImage(:,:, r).*meanImage(:,:, c)
            covarMat[r,c,:] = temp.T.flatten()

    for i in range(0,3):
        covarMat[i,i,:] = covarMat[i,i,:] + epsilon

    for r in range(1,3):
        for c in range(0,r):
            covarMat[r,c,:] = covarMat[c,r,:]
    return meanImage,covarMat

# def im2col(indices,windowSize):
#     m,n=indices.shape
#     num=(m-2)*(n-2)
#     neighInd=np.zeros(windowSize*windowSize*num).reshape(windowSize*windowSize,num)
#     for i in range(0,num):
#         first=i%(m-2)
#         second=i//(m-2)
#         for j in range(0,windowSize):
#             for k in range(0,windowSize):
#                 neighInd[j*windowSize+k,i]=indices[first+j,second+k]
#     return neighInd

def im2col(indices, windowSize, stepsize=1):
    # Parameters
    M,N = indices.shape
    col_extent = N - windowSize + 1
    row_extent = M - windowSize + 1

    # Get Starting block indices
    start_idx = np.arange(windowSize)[:,None]*N + np.arange(windowSize)

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (indices,start_idx.T.ravel()[:,None] + offset_idx.T.ravel()[::stepsize])

def localMattingAffinity(image,inMap,windowRadius=1,epsilon=1e-7):
    windowSize=windowRadius*2+1
    neighSize=windowSize*windowSize
    h,w,c=image.shape
    N=h*w
    epsilon = epsilon / neighSize


    # meanImage, covarMat = localRGBnormalDistributions(image, windowRadius, epsilon)

    # Determine pixels and their local neighbors
    indices = np.reshape(range(0, h * w), (w,h)).T
    neighInd = im2col(indices, windowSize)
    neighInd=neighInd.T

    inMap = inMap[windowRadius:h- windowRadius,windowRadius:w-windowRadius]
    # np.delete(inMap,[0,h-1],axis=0)
    # np.delete(inMap,[0,w-1], axis=1)
    inMap=inMap.T.flatten()
    #neighInd = neighInd(inMap,:);
    neighInd = neighInd[inMap==1,:]
    #neighInd=np.array([neighInd[i,:] for i in range(len(inMap)) if inMap[i]==1])
    #inInd = neighInd(:, (neighSize + 1) / 2);
    inInd=neighInd[:,np.int((neighSize + 1) / 2)]
    pixCnt = len(inInd)

    # Prepare in & out data
    image = image.reshape(N,c,order='F')
    # meanImage = meanImage.reshape(N, c,order='F')
    flowRows = np.zeros((neighSize, neighSize, pixCnt))
    flowCols = np.zeros((neighSize, neighSize, pixCnt))
    flows = np.zeros((neighSize, neighSize, pixCnt))

    #Compute matting affinity
    for i in range(0,len(inInd)):
        neighs = neighInd[i,:]
        #shiftedWinColors = image(neighs,:) - repmat(meanImage(inInd(i),:), [size(neighs, 2), 1])
        # shiftedWinColors = np.array([image[np.int(j),:] for j in neighs]) - repmat(meanImage[np.int(inInd[i]),:], len(neighs), 1)

        # shiftedWinColors = image[neighs,:] - repmat(meanImage[np.int(inInd[i]), :],len(neighs), 1)
        # flows[:,:,i] = np.dot(shiftedWinColors ,np.dot(np.linalg.inv(covarMat[:,:,np.int(inInd[i])]),shiftedWinColors.T))
        shiftedWinColors = image[neighs, :] - repmat(np.mean(image[neighs, :],0), len(neighs), 1)
        winI=image[neighs, :]
        win_mu=np.mean(winI,0)
        covarMat=winI.T.dot(winI)/neighSize-win_mu.dot(win_mu.T) +epsilon/neighSize*np.eye(c)
        flows[:, :, i] = np.dot(shiftedWinColors,np.dot(np.linalg.inv(covarMat), shiftedWinColors.T))

        neighs = repmat(neighs, len(neighs), 1)
        flowRows[:,:,i] = neighs
        flowCols[:,:,i] = neighs.T


    flows = (flows + 1)/neighSize
    #W = sparse(flowRows(:), flowCols(:), flows(:), N, N);
    # W=np.zeros((N,N))
    flowRows=flowRows.flatten()
    flowCols=flowCols.flatten()
    flows=flows.flatten()
    W=csc_matrix((flows, (flowRows, flowCols)), shape=(N, N))
    # for i in range(len(flows)):
    #     W[flowRows[i],flowCols[i]]=flows[i]

    # Make sure it's symmetric
    W = W + W.T

    # Normalize
    #sumW = full(sum(W, 2));
    sumW=W.sum(axis=1)
    sumW=np.array(sumW)
    sumW[sumW < 0.05] = 1
    # tmp=[i for i in range(len(sumW)) if sumW[i]<0.05]
    # for i in tmp:
    #     sumW[i]=1
    #W = spdiags(1. / sumW(:), 0, N, N) *W;
    #W=np.dot(sp.sparse.diags(np.array(1/sumW).flatten(),0),W)
    W = sp.sparse.diags(np.array(1 / sumW).flatten(), 0).dot(W)
    return W

def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)

# Returns sparse matting laplacian
def computeLaplacian(img, eps=10**(-7), win_rad=1):
    win_size = (win_rad*2+1)**2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - win_rad - 1, w - win_rad - 1
    win_diam = win_rad*2+1

    indsM = np.arange(h*w).reshape((h, w))
    # image2 = np.zeros((w, h, d))
    # image2[:, :, 0] = img[:, :, 0].T
    # image2[:, :, 1] = img[:, :, 1].T
    # image2[:, :, 2] = img[:, :, 2].T
    #img = np.reshape(image2, (h*w, d))
    ravelImg = img.reshape(h*w, d)
    win_inds = rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=2, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI)/win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = sp.sparse.csc_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L

def closed_form_matte(img, scribbled_img, mylambda=100):
    h, w,c  = img.shape
    # consts_map = (np.sum(abs(img - scribbled_img), axis=-1)>0.001).astype(np.float64)
    consts_map = np.logical_or(scribbled_img > 0.8 , scribbled_img < 0.2).astype(np.float64)
    #scribbled_img = rgb2gray(scribbled_img)

    consts_vals = scribbled_img*consts_map

    D_s = consts_map.ravel()
    b_s = consts_vals.ravel()
    print("Computing Matting Laplacian")
    L = computeLaplacian(img)
    sD_s = sp.sparse.diags(D_s)
    print("Solving for alpha")
    x = sp.sparse.linalg.spsolve(L + mylambda*sD_s, mylambda*b_s)
    # x, info = sp.sparse.linalg.cg(L + mylambda*sD_s, mylambda*b_s, maxiter=2000)
    alpha = np.minimum(np.maximum(x.reshape(h, w), 0), 1)
    return alpha

def getDist(dataSet, Sample):
    return np.sum(np.power(dataSet - Sample, 2))

def sortDist(dist):
    n = np.shape(dist)[0]
    id=np.arange(0,n)
    for i in range(1, n):
        for j in range(n - i):
            if (dist[j] > dist[j + 1]):
                temp1 = dist[j]
                dist[j] = dist[j + 1]
                dist[j + 1] = temp1
                temp2 = id[j]
                id[j] = id[j + 1]
                id[j + 1] = temp2
    return id

def knn(featuresout, featuresin,  K ):
    n, d = featuresin.shape
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = getDist(featuresin[i], featuresout)
    #id = sortDist(dist)
    id=np.argsort(dist)
    neighid=id[0:K]
    return  neighid

def knnsearch(featuresout, featuresin,  K ):
    featuresout=np.array(featuresout)
    featuresin=np.array(featuresin)
    n = featuresout.shape[0]
    neighid = np.zeros((n,K))
    for i in range(n):
        neighid[i,:]=knn(featuresout[i,:], featuresin,K)
        print(i)
    return neighid

def findNonlocalNeighbors(image, K, xyWeight, inMap, outMap, eraseSelfMatches=1):
    h, w, c = image.shape
    N=h*w

    # image2=np.zeros((w,h,c))
    # image2[:,:,0]=image[:,:,0].T
    # image2[:, :, 1] = image[:, :, 1].T
    # image2[:, :, 2] = image[:, :, 2].T
    features = np.reshape(image, (h * w, c))
    if xyWeight > 0:
        [x, y] = np.meshgrid(np.arange(w),np.arange(h))
        #x= [[float(i) for i in inner] for inner in x]
        #y = [[float(i) for i in inner] for inner in y]
        x = xyWeight * (x+1) / (w)
        y = xyWeight * (y+1) / (h)
        #features = [features x(:) y(:)]
        x = np.reshape(x, (h * w, 1))
        y = np.reshape(y, (h * w, 1))
        features=np.column_stack((features,x,y))

    inMap = inMap.ravel()
    outMap = outMap.ravel()
    indices = np.arange(h*w)
    inInd = [indices[i] for i in range(len(inMap)) if inMap[i]==1]
    outInd = [indices[i] for i in range(len(outMap)) if outMap[i]==1]

    if eraseSelfMatches:
        # Find K + 1 matches to count for self - matches
        #neighbors = knnsearch([features[i,:] for i in range(0,N) if outMap[i]!=0], [features[i,:] for i in range(0,N) if inMap[i]!=0], K + 1)

        # print('         Buiding kdTree...')
        # tree = cKDTree(np.array([features[i,:] for i in range(0,N) if outMap[i]!=0]))
        # print('         Quering kdTree...')
        # d, neighbors = tree.query(np.array([features[i,:] for i in range(0,N) if inMap[i]!=0]), k=K+1, p=np.inf, eps=0.0)

        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=K+1, n_jobs=4).fit(np.array([features[i,:] for i in range(0,N) if outMap[i]!=0]))
        neighbors = nbrs.kneighbors(np.array([features[i,:] for i in range(0,N) if inMap[i]!=0]))[1]

        # Get rid of self - matches
        validNeighMap = np.ones(neighbors.shape)
        #validNeighMap(inMap(inInd) & outMap(inInd), 1) = 0
        validNeighMap[[i for i in range(len(inInd)) if inMap[inInd[i]]==1 and outMap[inInd[i]]==1], 0] = 0
        validNeighMap[:, -1] = 1-validNeighMap[:, 0]
        validNeighbors = np.zeros((neighbors.shape[0], neighbors.shape[1] - 1))
        for i in range(validNeighbors.shape[0]):
            validNeighbors[i,:] = neighbors[i, [j for j in range(validNeighMap.shape[1]) if validNeighMap[i,j]==1]]
        #neighInd = outInd[validNeighbors]
        neighInd = [[outInd[int(i)] for i in inn] for inn in validNeighbors]
    else:
        #neighbors = knnsearch([features[i, :] for i in range(0, N) if outMap[i] != 0], [features[i, :] for i in range(0, N) if inMap[i] != 0], K)
        print('         Buiding kdTree...')
        tree = cKDTree(np.array([features[i, :] for i in range(0, N) if outMap[i] != 0]))
        print('         Quering kdTree...')
        d, neighbors = tree.query(np.array([features[i, :] for i in range(0, N) if inMap[i] != 0]), k=K, p=np.inf,eps=0.0)
        #neighInd = outInd[neighbors]
        neighInd = [[outInd[int(i)] for i in inn] for inn in neighbors]
    return inInd, neighInd, features

def findNonlocalNeighborsKU(image, K, xyWeight, inMap, outMap, eraseSelfMatches=1):
    h, w, c = image.shape
    N=h*w

    image2=np.zeros((w,h,c))
    image2[:,:,0]=image[:,:,0].T
    image2[:, :, 1] = image[:, :, 1].T
    image2[:, :, 2] = image[:, :, 2].T
    features = np.reshape(image2, (h * w, c))
    if xyWeight > 0:
        [x, y] = np.meshgrid(np.arange(w),np.arange(h))
        #x= [[float(i) for i in inner] for inner in x]
        #y = [[float(i) for i in inner] for inner in y]
        x = xyWeight * x / w
        y = xyWeight * y / h
        #features = [features x(:) y(:)]
        x=np.reshape(x.T,(h*w,1))
        y = np.reshape(y.T, (h * w, 1))
        features=np.column_stack((features,x,y))

    inMap = inMap.T.flatten()
    outMap = outMap.T.flatten()
    indices = np.arange(h*w)
    inInd = [indices[i] for i in range(len(inMap)) if inMap[i]==1]
    outInd = [indices[i] for i in range(len(outMap)) if outMap[i]==1]

    if eraseSelfMatches:
        # Find K + 1 matches to count for self - matches
        #neighbors = knnsearch([features[i,:] for i in range(0,N) if outMap[i]!=0], [features[i,:] for i in range(0,N) if inMap[i]!=0], K + 1)

        print('         Buiding kdTree...')
        # tree = cKDTree(np.array([features[i,:] for i in range(0,N) if outMap[i]!=0]))
        tree= sklearn.neighbors.KDTree(features[outMap==1,:])
        print('         Quering kdTree...')
        # d, neighbors = tree.query(np.array([features[i,:] for i in range(0,N) if inMap[i]!=0]), k=K+1, p=np.inf, eps=0.0)
        dist, neighbors = tree.query(features[inMap==1,:], k=K+1)

        # nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=K+1, n_jobs=4).fit(np.array([features[i,:] for i in range(0,N) if outMap[i]!=0]))
        # neighbors = nbrs.kneighbors(np.array([features[i,:] for i in range(0,N) if inMap[i]!=0]))[1]

        # Get rid of self - matches
        validNeighMap = np.ones(neighbors.shape)
        #validNeighMap(inMap(inInd) & outMap(inInd), 1) = 0
        validNeighMap[[i for i in range(len(inInd)) if inMap[inInd[i]]==1 and outMap[inInd[i]]==1], 0] = 0
        validNeighMap[:, -1] = 1-validNeighMap[:, 0]
        validNeighbors = np.zeros((neighbors.shape[0], neighbors.shape[1] - 1))
        for i in range(validNeighbors.shape[0]):
            validNeighbors[i,:] = neighbors[i, [j for j in range(validNeighMap.shape[1]) if validNeighMap[i,j]==1]]
        #neighInd = outInd[validNeighbors]
        neighInd = [[outInd[int(i)] for i in inn] for inn in validNeighbors]
    else:
        #neighbors = knnsearch([features[i, :] for i in range(0, N) if outMap[i] != 0], [features[i, :] for i in range(0, N) if inMap[i] != 0], K)
        print('         Buiding kdTree...')
        tree = cKDTree(np.array([features[i, :] for i in range(0, N) if outMap[i] != 0]))
        print('         Quering kdTree...')
        d, neighbors = tree.query(np.array([features[i, :] for i in range(0, N) if inMap[i] != 0]), k=K, p=np.inf,eps=0.0)
        #neighInd = outInd[neighbors]
        neighInd = [[outInd[int(i)] for i in inn] for inn in neighbors]
    return inInd, neighInd, features

def localLinearEmbedding(pt, neighbors, conditionerMult):
    # each column of neighbors represent a neighbor, each row a dimension
    # pt is a row vector
    corr = np.dot(neighbors.T , neighbors) + conditionerMult * np.eye(neighbors.shape[1])
    corrInv = np.linalg.inv(corr)
    ptDotN = np.dot(neighbors.T ,pt)
    alpha = 1 - np.sum(corrInv.dot(ptDotN))
    beta = np.sum(corrInv)
    lagrangeMult = alpha / beta
    w = corrInv.dot((ptDotN + lagrangeMult))
    return w

def colorMixtureAffinities(image, K, inMap, outMap, xyWeight=1, useXYinLLEcomp=0):
    h,w,c = image.shape
    N = h * w

    if len(outMap)==0:
        outMap=np.ones((h,w))

    inInd, neighInd, features = findNonlocalNeighbors(image, K, xyWeight, inMap, outMap)
    inInd=np.array(inInd)
    neighInd=np.array(neighInd)

    if not useXYinLLEcomp:
        features = features[:,0:-2]

    #flows = zeros(size(inInd, 1), size(neighInd, 2))
    flows = np.zeros((inInd.shape[0],neighInd.shape[1]))

    for i in range(0,len(inInd)):
        #flows[i,:] = localLinearEmbedding(features(inInd(i),:)', features(neighInd(i, :), :)', 1e-10)
        flows[i, :] = localLinearEmbedding(features[inInd[i],:].T, features[[j for j in neighInd[i,:]],:].T, 1e-10)

    flows_sum=np.sum(flows, axis=1)
    flows_sum=flows_sum.reshape(len(flows_sum),1)
    flows = flows / repmat(flows_sum,1, K)

    inInd=inInd.reshape(len(inInd),1)
    inInd = repmat( inInd , 1 , K )

    #Wcm = sparse(inInd(:), neighInd(:), flows, N, N);
    #Wcm=np.zeros((N,N))
    inInd=inInd.ravel()
    neighInd=neighInd.ravel()
    flows=flows.ravel()
    #Wcm = csr_matrix((flows, (inInd, neighInd)), shape=(N, N)).toarray()
    Wcm = csc_matrix((flows, (inInd, neighInd)), shape=(N, N))
    # for i in range(len(inInd)):
    #     Wcm[inInd[i],neighInd[i]]=flows[i]

    return Wcm

def colorSimilarityAffinities(image, K, inMap, outMap, xyWeight=0.05, useHSV=0):
    h,w,c = image.shape
    N = h * w

    inInd, neighInd1, features = findNonlocalNeighbors(image, K, xyWeight, inMap, outMap)

    # This behaviour below, decreasing the xy-weight and finding a new set of neighbors, is taken
    # from the public implementation of KNN matting by Chen et al.
    inInd, neighInd2, features = findNonlocalNeighbors(image, np.int(K/5+0.5), xyWeight / 100, inMap, outMap)
    neighInd = np.column_stack((neighInd1, neighInd2))
    features[:, -2:] = features[:, -2:] / 100

    inInd=np.array(inInd)
    inInd=inInd.reshape(len(inInd),1)
    inInd = repmat(inInd, 1, neighInd.shape[1])
    inInd=inInd.T.flatten().reshape((inInd.shape[0]*inInd.shape[1]))
    neighInd=neighInd.T.flatten().reshape((neighInd.shape[0]*neighInd.shape[1]))
    #flows = max(1 - sum(abs(features(inInd(:), :) - features(neighInd(:), :)), 2) / size(features, 2), 0);
    flows=np.maximum( 1-np.sum((features[inInd,:]-features[neighInd,:]).__abs__(),axis=1)/features.shape[1] ,0)
    #Wcs = sparse(inInd(:), neighInd(:), flows, N, N);
    # Wcs = np.zeros((N, N))
    # for i in range(len(inInd)):
    #     Wcs[inInd[i],neighInd[i]]=flows[i]
    flows=flows.reshape(len(flows))
    Wcs = csc_matrix((flows, (inInd, neighInd)), shape=(N, N))
    Wcs = (Wcs + Wcs.T) / 2 # If p is a neighbor of q, make q a neighbor of p
    return Wcs

def knownToUnknownColorMixture(image, trimap, K=7, xyWeight=10):
    # Known-to-Unknown Information Flow
    # This function implements the known - to - unknown information flow in
    # Yagiz Aksoy, Tunc Ozan Aydin, Marc Pollefeys, "Designing Effective
    # Inter - Pixel Information Flow for Natural Image Matting", CVPR, 2017.
    # All parameters other than image and the trimap are optional.The outputs
    # are the weight of FG pixels inside the unknown region, and the confidence
    # on these estimated values.
    # - K defines the number of neighbors found in FG and BG from which
    #   LLE weights are computed.
    # - xyWeight determines how much importance is given to the spatial
    #   coordinates in the nearest neighbor selection.
    bg = trimap < 0.2
    fg = trimap > 0.8
    unk = np.logical_not(np.logical_or(bg , fg))
    w,h=trimap.shape

    # Find neighbors of unknown pixels in FG and BG
    inInd, bgInd, features = findNonlocalNeighbors(image, K, xyWeight, unk, bg)
    inInd2 ,fgInd,features = findNonlocalNeighbors(image, K, xyWeight, unk, fg)
    neighInd = np.column_stack((fgInd, bgInd))
    neighInd=np.array(neighInd)
    inInd=np.array(inInd)

    # Compute LLE weights and estimate FG and BG colors that got into the mixture
    features = features[:, 0 : -2]
    flows = np.zeros((inInd.shape[0],neighInd.shape[1]))
    fgCols = np.zeros((inInd.shape[0],3))
    bgCols = np.zeros((inInd.shape[0],3))
    for i in range(inInd.shape[0]):
        flows[i, :] = localLinearEmbedding(features[inInd[i],:].T, features[neighInd[i, :], :].T, 1e-10)
        #tmp=np.array(flows[i, : K])
        fgCols[i, :] = sum(features[neighInd[i,: K], :] * repmat(flows[i, : K].reshape(K,1), 1,3), 0)
        bgCols[i, :] = sum(features[neighInd[i, K:], :] * repmat(flows[i, K:].reshape(K,1), 1,3), 0)

    # Estimated alpha is the sum of weights of FG neighbors
    alphaEst = trimap
    # alphaEst=np.zeros(trimap.shape)
    #alphaEst.T[unk==1] = np.sum(flows[:, :K], axis=1)
    tmp=np.sum(flows[:, :K], axis=1)
    # tmp=np.maximum(tmp,0)
    # tmp = np.minimum(tmp, 1)
    alphaEst[unk==1] = tmp
    # index=0
    # for j in range(0,h):
    #     for i in range(0,w):
    #         if unk[i,j]==1:
    #             alphaEst[i,j]=tmp[index]
    #             index=index+1

    #Compute the confidence based on FG - BG color difference
    unConf = fgCols - bgCols
    unConf = np.sum(unConf * unConf, axis=1) / 3
    conf = np.double(np.logical_or(fg , bg))
    conf[unk==1] = unConf

    # alphaEst[trimap < 0.2] = 0
    # alphaEst[trimap > 0.8] = 1
    # im = Image.fromarray((alphaEst*255).astype(np.uint8))
    # im.show()
    return alphaEst, conf

def affinityMatrixToLaplacian(aff):
    spdiag=sp.sparse.diags(np.array(aff.sum(1)).flatten(),0)
    Lap = spdiag - aff
    # re = np.diag(np.array(aff.sum(1)).flatten())
    # h,w=re.shape
    # Lap=np.zeros((h,w))
    # for i in range(h):
    #     Lap[i,:]=re[i,:]-aff[i,:]
    return Lap