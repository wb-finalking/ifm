import matplotlib.image as mpimg  # mpimg 用于读取图片
from ifm import *
from PIL import Image

class parameters:
    def __init__(self):
        self.lamb = 100
        self.usePCGtoSolve = 1
        # Switch to use known-to-unknown information flow
        # -1: automatic selection, 0: do not use, 1: use

        self.useKnownToUnknown = 1

        # Switch to apply edge - based trimming after matte estimation
        # The value reported in the paper is true, although we leave the
        # default as false here.
        self.mattePostTrim = 0

        # Color mixture information flow parameters
        self.cm_K = 20
        self.cm_xyw = 1
        self.cm_mult = 1

        # Intra - unknown information flow parameters
        self.iu_K = 5
        self.iu_xyw = 0.05
        self.iu_mult = 0.01

        # Known - to - unknown information flow parameters
        self.ku_K = 7
        self.ku_xyw = 10
        self.ku_mult = 0.05

        # Intra - unknown information flow parameters self.iu_K = 5 self.iu_xyw = 0.05 self.iu_mult = 0.01

        # Local information flow parameters
        self.loc_win = 1
        self.loc_eps = 1e-6
        self.loc_mult = 1

        # Parameter for Information Flow Matting matte refinement
        self.refinement_mult = 0.1

params=parameters()
#print(param.lamb)

image=mpimg.imread('input2.png')
image=image[:,:,0:3]
trimap=mpimg.imread('trimap2.png')
trimap=trimap[:,:,1].copy()

alpha=informationFlowMatting(image, trimap, params)
alpha=alpha*255
im=Image.fromarray(alpha.astype(np.uint8))
im.show()
