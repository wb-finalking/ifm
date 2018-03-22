from laplacian_matrices import *
#from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spilu
# from scikits.umfpack import spsolve

def imdilate(org , window):
    h,w=org.shape
    radius=np.int((window.shape[0]-1)/2)
    out=np.zeros((h,w))

    for i in range(radius,h-radius):
        for j in range(radius,w-radius):
            if np.sum(window*(org[i-radius:i+radius+1,j-radius:j+radius+1]))>0:
                out[i,j]=1

    #out=np.bool(out)
    return out

def LabelExpansion(image, trimap, maxDist, colorThresh):
    h,w,c =  image.shape
    fg = trimap > 0.8
    bg = trimap < 0.2
    knownReg = np.logical_or(bg , fg)
    extendedTrimap = trimap

    window=np.ones((2 * maxDist + 1,2 * maxDist + 1))
    searchReg= np.logical_or(np.logical_and(imdilate(fg, window) , np.logical_not(fg) ) , np.logical_and(imdilate(bg, window) , np.logical_not(bg)))
    cols,rows = np.meshgrid(np.arange(w), np.arange(h))
    cols = cols[searchReg==1]
    rows = rows[searchReg==1]

    winCenter = (2 * maxDist) / 2
    distPlane = repmat(np.arange(0 , 2 * maxDist + 1).reshape(2 * maxDist + 1,1), 1 , 2 * maxDist + 1).T
    distPlane = np.sqrt((distPlane - winCenter) **2 + (distPlane.T - winCenter) ** 2)

    for pixNo in range(0,len(cols)):
        r = rows[pixNo]
        c = cols[pixNo]
        minR = max(r - maxDist, 0) # pixel limits
        minC = max(c - maxDist, 0)
        maxR = min(r + maxDist,h-1)
        maxC = min(c + maxDist, w-1)
        winMinR = np.int(winCenter - (r - minR) )# pixel limits in window
        winMinC = np.int(winCenter - (c - minC))
        winMaxR = np.int(winCenter + (maxR - r))
        winMaxC = np.int(winCenter + (maxC - c))

        pixColor = image[r, c, :]
        imgWin = image[minR : maxR+1, minC : maxC+1, :] # colors
        trimapWin = trimap[minR : maxR+1, minC : maxC+1]

        winColorDiff=np.zeros(imgWin.shape)
        winColorDiff[:,:,0]= imgWin[:, :, 0] - pixColor[0]
        winColorDiff[:, :, 1] = imgWin[:, :, 1] - pixColor[1]
        winColorDiff[:, :, 2] = imgWin[: ,:, 2] - pixColor[2]
        winColorDiff = np.sqrt(np.sum(winColorDiff * winColorDiff, axis=2))

        candidates= np.logical_and((winColorDiff < colorThresh) , knownReg[minR : maxR+1, minC : maxC+1]) # known pixels under thresh
        if np.sum(candidates) > 0:
            distWin = distPlane[winMinR : winMaxR+1, winMinC : winMaxC+1] # distance plane
            distWin = distWin[candidates==1] # distances of known
            #minDistInd = distWin.index(min(distWin)) # location of minimum
            minDistInd = distWin.argmin()
            trimapWin = trimapWin[candidates==1]
            extendedTrimap[r, c] = trimapWin[minDistInd]

    return extendedTrimap

def trimmingFromKnownUnknownEdges(image, trimap, paramU=9 / 256, paramD=1 / 256, iterCnt=9):
    # bg = trimap < 0.2
    # fg = trimap > 0.8
    paramD = paramU - paramD

    for i in range(1,iterCnt):
        iterColorThresh = paramU - i * paramD / iterCnt # color threshold = paramU - iterNo * (paramU - paramD) / maxIter
        trimap = LabelExpansion(image, trimap, i, iterColorThresh) #distance threshold 1 to iterCnt

    return trimap

def solveForAlphas(Lap, trimap, lamb, usePCG, alphaHat, conf, aHatMult=0.1):
    result = trimap.shape
    h=result[0]
    w=result[1]
    N = h * w
    known = np.logical_or(trimap > 0.8 , trimap < 0.2)
    #A = lamb * np.diag(np.double(known.T.flatten()))
    # A = lamb * sp.sparse.diags(known.T.flatten())
    A = lamb * sp.sparse.diags(known.flatten().astype(np.float64))

    if len(alphaHat)!=0:
        conf[known==1] = 0
        A = A + aHatMult * sp.sparse.diags(conf.flatten())
        b = A.dot(alphaHat.reshape(N,1))
    else:
        # b = A.dot(np.double(trimap.T.flatten()[:,None] > 0.8))
        b = A.dot(np.double(trimap.reshape(N,1) > 0.8))

    A = A + Lap

    # x0=np.array(trimap.T).ravel()

    #alphas = np.dot(np.linalg.inv(A) , b)
    # alphas = sp.sparse.linalg.spsolve(A,b)
    #alphas = spsolve(A, b)
    #alphas, info = cg(A, b)
    #alphas = np.dot(sp.sparse.linalg.inv(A), b)

    # M_x = lambda x: sp.sparse.linalg.spsolve(A, x)
    # M = sp.sparse.linalg.LinearOperator((N, N), M_x)

    # M_inverse = sp.sparse.linalg.spilu(A)
    # M2 = sp.sparse.linalg.LinearOperator((N, N), M_inverse.solve)
    alphas,info = sp.sparse.linalg.cg(A, b,maxiter=2000)
    # alphas = sp.sparse.linalg.spsolve(A, b)

    alphas[alphas < 0] = 0
    alphas[alphas > 1] = 1
    return alphas

def patchBasedTrimming(image, trimap, minDist=0.25, maxDist=0.9, windowRadius=1, K=10):
    h, w, c = image.shape

    epsilon = 1e-8

    fg = trimap > 0.8
    bg = trimap < 0.2
    unk = np.logical_not(np.logical_or(fg , bg))

    meanImage, covarMat = localRGBnormalDistributions(image, windowRadius, epsilon)

    unkInd, fgNeigh , features = findNonlocalNeighbors(meanImage, K, -1, unk, fg)
    unkInd2, bgNeigh , features = findNonlocalNeighbors(meanImage, K, -1, unk, bg)

    meanImage = np.reshape(meanImage, (h * w, meanImage.shape[2]))

    fgBhatt = np.zeros((K, 1))
    bgBhatt = np.zeros((K, 1))
    unkInd=np.array(unkInd)
    fgNeigh=np.array(fgNeigh)
    bgNeigh=np.array(bgNeigh)
    trimap=trimap.T.flatten()
    for i in range(unkInd.shape[0]):
        pixMean = meanImage[unkInd[i], :].T
        pixCovar = covarMat[:, :, unkInd[i]]
        pixDet = np.linalg.det(pixCovar)
        for n in range(K):
            nMean = meanImage[fgNeigh[i, n], :].T - pixMean
            nCovar = covarMat[:, :, fgNeigh[i, n]]
            nDet = np.linalg.det(nCovar)
            nCovar = (pixCovar + nCovar) / 2
            fgBhatt[n] = 0.125 * nMean.T.dot(np.dot(np.linalg.inv(nCovar),nMean)) + 0.5 * np.log(np.linalg.det(nCovar) / np.sqrt(pixDet * nDet)) # Bhattacharyya distance
        for n in range(K):
            nMean = meanImage[bgNeigh[i, n], :].T - pixMean
            nCovar = covarMat[:, :, bgNeigh[i, n]]
            nDet = np.linalg.det(nCovar)
            nCovar = (pixCovar + nCovar) / 2
            bgBhatt[n] = 0.125 * nMean.T.dot(np.dot(np.linalg.inv(nCovar),nMean)) + 0.5 * np.log(np.linalg.det(nCovar) / np.sqrt(pixDet * nDet)) # Bhattacharyya distance
        minFGdist = min(fgBhatt)
        minBGdist = min(bgBhatt)
        if minFGdist < minDist:
            if minBGdist > maxDist:
                trimap[unkInd[i]] = 1
        elif minBGdist < minDist:
            if minFGdist > maxDist:
                trimap[unkInd[i]] = 0

    trimap=trimap.reshape(w,h).T

    return trimap

def informationFlowMatting(image, trimap, params, suppressMessages=0):

    if (not suppressMessages):
        print('Information-Flow Matting started...')

    # Decide to use the K - to - U flow
    useKU = params.useKnownToUnknown > 0
    # useKU= False

    if not suppressMessages:
        if useKU:
            print('     Known-to-unknown information flow will be used.')
        else:
            print('     Known-to-unknown information flow will NOT be used.')

    if params.mattePostTrim or useKU:
        # Trimap trimming for refining kToU flow or final matte
        if (not suppressMessages):
            print('     Trimming trimap from edges...')
        #edgeTrimmed = trimmingFromKnownUnknownEdges(image, trimap)

    edgeTrimmed=trimap

    # Compute L_IFM
    unk = np.logical_and(trimap < 0.8 , trimap > 0.2)
    #dilUnk = imdilate(unk, np.ones(2 * params.loc_win + 1))
    dilUnk=unk
    if (not suppressMessages):
        print('     Computing color mixture flow...')
    Lap = affinityMatrixToLaplacian(colorMixtureAffinities(image, params.cm_K, unk, [], params.cm_xyw))
    Lap = params.cm_mult * np.dot(Lap.T, Lap)

    if (not suppressMessages):
        print('     Computing local matting Laplacian...')
    # Lap = affinityMatrixToLaplacian(
    #     localMattingAffinity(image, dilUnk, params.loc_win, params.loc_eps))
    Lap=Lap + params.loc_mult * computeLaplacian(image)

    if (not suppressMessages):
        print('     Computing intra-U flow...')
    Lap = Lap + params.iu_mult * affinityMatrixToLaplacian(colorSimilarityAffinities(image, params.iu_K, unk, unk, params.iu_xyw))

    if useKU:
        #Compute kToU flow
        if (not suppressMessages):
            print('     Trimming trimap using patch similarity...')
        # patchTrimmed = patchBasedTrimming(image, trimap, 0.25, 0.9, 1, 5) # We set K = 5 here for better computation time
        if (not suppressMessages):
            print('     Computing K-to-U flow...')
        #kToU, kToUconf = knownToUnknownColorMixture(image, patchTrimmed, params.ku_K, params.ku_xyw)
        kToU, kToUconf = knownToUnknownColorMixture(image, trimap, params.ku_K, params.ku_xyw)
        kToU[edgeTrimmed < 0.2] = 0
        kToU[edgeTrimmed > 0.8] = 1
        kToUconf[edgeTrimmed < 0.2] = 1
        kToUconf[edgeTrimmed > 0.8] = 1
        if (not suppressMessages):
            print('     Solving for alphas...')
        alpha = solveForAlphas(Lap, trimap, params.lamb, params.usePCGtoSolve, kToU, kToUconf, params.ku_mult)
        # alpha=kToU.flatten()
        # alpha = kToUconf.T.flatten()
    else:
        if (not suppressMessages):
            print('     Solving for alphas...')
        alpha = solveForAlphas(Lap, trimap, params.lamb, params.usePCGtoSolve,[],[])

    # alpha = alpha.reshape(image.shape[1], image.shape[0]).T
    alpha = alpha.reshape(image.shape[0], image.shape[1])

    if params.mattePostTrim:
        alpha[edgeTrimmed < 0.2] = 0
        alpha[edgeTrimmed > 0.8] = 1

    return alpha