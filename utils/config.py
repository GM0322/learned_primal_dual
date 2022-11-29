import argparse

def getArgs():
    args = argparse.ArgumentParser()
    # parbeam scan parameters
    args.add_argument('--nViews',type=int,default=360)
    args.add_argument('--nBins',type=int,default=600)
    args.add_argument('--nSize',type=int,default=512)
    args.add_argument('--fCellSize',type=float,default=1.0)
    args.add_argument('--fPixelSize',type=float,default=600.0/512)
    args.add_argument('--dtheta',type=float,default=1.0)
    args.add_argument('--fRotateDir',type=float,default=1.0)
    #
    args.add_argument('--model',type=str,default='PrimalDual')
    args.add_argument('--iteration',type=int,default=5)
    args.add_argument('--isGenerateData',type=bool,default=False)
    args.add_argument('--isAddNoisy',type=bool,default=False)
    args.add_argument('--train_data',type=str,default=r'E:\oscpData\train\label')
    args.add_argument('--batchsize',type=int,default=1)
    args.add_argument('--lr',type=float,default=1e-4)
    args.add_argument('--epoch',type=int,default=100)
    args.add_argument('--gpu',type=list,default=[0])
    return args.parse_args()

def getScanParam():
    args = getArgs()
    geom={'nViews':args.nViews,
          'nBins':args.nBins,
          'nSize':args.nSize,
          'fCellSize':args.fCellSize,
          'fPixelSize':args.fPixelSize,
          'dtheta':args.dtheta,
          'fRotateDir':args.fRotateDir}
    return geom
