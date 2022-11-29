from CTOperator import TorchLayer
import torch

class chambolle_pock(torch.nn.Module):
    def __init__(self,n_iter):
        super(chambolle_pock, self).__init__()
        self.n_iter = n_iter
        # self.geom = config.getScanParam()
        self.fp = TorchLayer.ParBeamFPLayer()
        self.bp = TorchLayer.ParBeamBPLayer()
        self.layers = torch.nn.ModuleList()
        for i in range(n_iter):
            dual_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2,32,kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32,32,kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, 1, kernel_size=(3,3),padding=(1,1)),
            )
            primal_layer = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, 1, kernel_size=(3,3),padding=(1,1)),
            )
            self.layers.append(dual_layer)
            self.layers.append(primal_layer)

    def forward(self, primal, proj):
        dual = proj.clone()
        primal_bar = primal.clone()
        for i in range(self.n_iter):
            evalop = self.fp(primal_bar)
            update = torch.cat((dual + 0.5 * evalop, proj), dim=1)
            update = self.layers[2 * i](update)
            dual = dual + update

            evalop = self.bp(dual)
            update = primal - 0.5 * evalop
            update = self.layers[2 * i + 1](update)
            primal = primal + update

            primal_bar = primal + 1.0 * update
        return primal

class primal(torch.nn.Module):
    def __init__(self,n_iter,n_primal):
        super(primal, self).__init__()
        self.n_iter = n_iter
        self.fp = TorchLayer.ParBeamFPLayer()
        self.bp = TorchLayer.ParBeamBPLayer()
        self.layers = torch.nn.ModuleList()
        self.n_primal = n_primal
        for i in range(n_iter):
            primal_layer = torch.nn.Sequential(
                torch.nn.Conv2d(n_primal+1, 32, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, n_primal, kernel_size=(3, 3), padding=(1, 1)),
            )
            self.layers.append(primal_layer)

    def forward(self, primal, proj):
        primal = torch.cat([primal] * self.n_primal, dim=1)
        for i in range(self.n_iter):
            evalop = self.fp(primal[:,1:2,...])
            dual = evalop - proj

            evalop = self.bp(dual[:,0:1,...])
            update = torch.cat([primal,evalop],dim=1)
            update = self.layers[i](update)
            primal = primal + update
        return primal[:,0:1,...]

class primal_dual(torch.nn.Module):
    def __init__(self,n_iter,n_primal,n_dual):
        super(primal_dual, self).__init__()
        self.n_iter = n_iter
        # self.geom = config.getScanParam()
        self.fp = TorchLayer.ParBeamFPLayer()
        self.bp = TorchLayer.ParBeamBPLayer()
        self.layers = torch.nn.ModuleList()
        self.n_primal = n_primal
        self.n_dual = n_dual

        for i in range(n_iter):
            dual_layer = torch.nn.Sequential(
                torch.nn.Conv2d(n_dual+2,32,kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32,32,kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, n_dual, kernel_size=(3,3),padding=(1,1)),
            )
            primal_layer = torch.nn.Sequential(
                torch.nn.Conv2d(n_primal+1, 32, kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, n_primal, kernel_size=(3,3),padding=(1,1)),
            )
            self.layers.append(dual_layer)
            self.layers.append(primal_layer)

    def forward(self, primal, proj):
        primal = torch.cat([primal] * self.n_primal, dim=1)
        dual = torch.cat([torch.zeros_like(proj,dtype=torch.float32)]*self.n_dual,dim=1).to(proj.device)
        for i in range(self.n_iter):
            evalop = self.fp(primal[:, 1:2, ...])
            update = torch.cat((dual,evalop,proj),dim=1)
            dual = self.layers[2*i](update)

            evalop = self.bp(dual[:, 0:1, ...])
            update = torch.cat([primal, evalop], dim=1)
            update = self.layers[2*i+1](update)
            primal = primal + update
        return primal[:, 0:1, ...]
