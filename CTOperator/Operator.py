import cupy
from cupy.cuda import runtime
from cupy.cuda.texture import ChannelFormatDescriptor, CUDAarray, ResourceDescriptor, TextureDescriptor,TextureReference

source_texref = r'''

#define PI (3.14159265f)
extern "C"{
texture<float, cudaTextureType2D, cudaReadModeElementType> texFP;
texture<float, cudaTextureType2D, cudaReadModeElementType> texImage;

__global__ void BPkernel(float* image, int width, int nViews, int nBins, double fCellsize, double pixelSize, double dtheta, double fRotateDir) {
	unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;
	if (h >= width || w >= width)
		return;
	float sum = 0;
	for (size_t i = 0; i < nViews; i++) {
		float x = ((w - width / 2.0 + 0.5) * cosf(fRotateDir * i * dtheta * PI / 180)
			- (h - width / 2.0 + 0.5) * sinf(fRotateDir * i * dtheta * PI / 180))
			* pixelSize / fCellsize + nBins / 2.0;
		sum += tex2D(texFP, x, i+0.5) * fCellsize * dtheta / 180;
	}
	image[w * width + h] = sum/(width*2*pixelSize);
}

__global__ void Projkenrel(float* fFPData, int width, int nViews, int nBins, double fCellsize, double pixelSize, double dtheta, double fRotateDir) {
	unsigned int View = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int Bin = blockIdx.y * blockDim.y + threadIdx.y;
	if (Bin >= nBins || View >= nViews)
		return;
	float Vx = sinf(fRotateDir * View * dtheta * PI / 180);
	float Vy = cosf(fRotateDir * View * dtheta * PI / 180);
	float fSourcex = (Bin - nBins / 2.0 + 0.5) * fCellsize * cosf(fRotateDir * View * dtheta * PI / 180);
	float fSourcey = -(Bin - nBins / 2.0 + 0.5) * fCellsize * sinf(fRotateDir * View * dtheta * PI / 180);
	fFPData[View * nBins + Bin] = 0;
	for (float i = -width; i < width; i += 1) {
		float indexY = (fSourcex + Vx * i) / pixelSize + width / 2.0;
		float indexX = (fSourcey + Vy * i) / pixelSize + width / 2.0;
		fFPData[View * nBins + Bin] += tex2D(texImage, indexX, indexY)*pixelSize;
	}
	// fFPData[View * nBins + Bin] = fFPData[View * nBins + Bin];
}

}
'''

def fp(img,geom,proj):
    block2D = (8, 8)
    grid2D = ((geom['nViews'] + block2D[0] - 1) // block2D[0],
              (geom['nBins'] + block2D[1] - 1) // block2D[1])
    mod = cupy.RawModule(code=source_texref)
    FpKernel = mod.get_function('Projkenrel')
    channelDescImg = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArrayImg = CUDAarray(channelDescImg, geom['nSize'], geom['nSize'])
    resourceDescImg = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayImg)
    address_modeImg = (runtime.cudaAddressModeBorder, runtime.cudaAddressModeBorder)
    texDescImg = TextureDescriptor(address_modeImg, runtime.cudaFilterModeLinear, runtime.cudaReadModeElementType)
    cuArrayImg.copy_from(img)
    TextureReference(mod.get_texref('texImage'), resourceDescImg, texDescImg)
    args = (proj, geom['nSize'], geom['nViews'], geom['nBins'], geom['fCellSize'], geom['fPixelSize'], geom['dtheta'], geom['fRotateDir'])
    FpKernel(grid2D, block2D, args)

def bp(proj,geom,img):
    block2D = (8, 8)
    grid2D = ((geom['nSize'] + block2D[0] - 1) // block2D[0],
              (geom['nSize'] + block2D[1] - 1) // block2D[1])
    mod = cupy.RawModule(code=source_texref)
    BpKernel = mod.get_function('BPkernel')
    channelDescImg = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    cuArrayImg = CUDAarray(channelDescImg, geom['nBins'], geom['nViews'])
    resourceDescImg = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArrayImg)
    address_modeImg = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
    texDescImg = TextureDescriptor(address_modeImg, runtime.cudaFilterModeLinear, runtime.cudaReadModeElementType)
    cuArrayImg.copy_from(proj)
    TextureReference(mod.get_texref('texFP'), resourceDescImg, texDescImg)
    # ImgData = cupy.zeros((geom['width'],geom['width']),dtype=cupy.float32)
    args = (img, geom['nSize'], geom['nViews'], geom['nBins'], geom['fCellSize'], geom['fPixelSize'], geom['dtheta'], geom['fRotateDir'])
    BpKernel(grid2D, block2D, args)
    # return Img
