import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from util import get_output_dir,set_seed,set_cuda,set_gpu,\
    copy_source,to_named_dict,setup_logging,\
    plot_stats,save_images

from ops import weights_init_xavier, DecodeBlock #,ResUpBlock2d #,

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--gpuid', default='0',help='GPU ids of running')
    parser.add_argument('--datapath', type=str, default='face')
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--img_channels', default=3, type=int)
    parser.add_argument('--channel_base', default=64*64, type=int)
    parser.add_argument('--channel_max', default=512, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--ninterp', default=12, type=int)
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z')
    parser.add_argument('--l_steps', type=int, default=30, help='number of langevin steps')
    parser.add_argument('--l_step_size', type=float, default=0.1, help='stepsize of langevin')
    parser.add_argument('--sigma', type=float, default=0.67, help='prior of llhd, factor analysis')
    parser.add_argument('--gammar', type=float, default=1, help='robust and perturbation of mcmc')
    parser.add_argument('--prior_sigma', type=float, default=1, help='prior of z')

    parser.add_argument('--lr', default=0.0004, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--gamma', default=0.996, help='lr decay')
    parser.add_argument('--epbase', type=int, default=50, help='epoch base for lr_sin of lagevin')
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--n_printout', type=int, default=1, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=10, help='plot each n epochs')
    parser.add_argument('--n_ckpt', type=int, default=100, help='save ckpt each n epochs')
    parser.add_argument('--n_stats', type=int, default=10, help='stats each n epochs')
    return parser.parse_args()


class GenNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.global_iter = 0
        self.img_channels = args.img_channels
        self.img_resolution = args.img_size
        self.img_resolution_log2 = int(np.log2(args.img_size))
        self.decode_block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        # 4:8:...:resolution
        decode_channels_dict = {res: min(args.channel_base // res, args.channel_max) for res in
                                self.decode_block_resolutions[:-1]}
        # 4*  :8*  :...(res//4)*64:(res//2)*32  |For res=256
        for res in self.decode_block_resolutions[:-1]:
            in_channels = decode_channels_dict[res]
            out_channels = decode_channels_dict[res * 2] if res != self.decode_block_resolutions[-2] else args.img_channels
            is_last = (res == self.decode_block_resolutions[-2])
            block = DecodeBlock(in_channels, out_channels, resolution=res, is_last=is_last)
            #block = ResUpBlock2d(in_channels, out_channels, is_last=is_last)
            setattr(self, f'decode_b{res}', block)
        self.fc_d1 = nn.Linear(args.nz, 4 * 4 * decode_channels_dict[4])
        self.decode_channels_dict = decode_channels_dict
        self.flatten = nn.Flatten()

    def decode(self, z):
        x = F.relu(self.fc_d1(z)).view(z.shape[0],self.decode_channels_dict[4],4,4)
        for res in self.decode_block_resolutions[:-1]:
            block = getattr(self, f'decode_b{res}')
            x = block(x)
        return x

    def forward(self, z):
        return self.decode(z)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.Gnet = GenNet(args)
        self.Gnet.apply(weights_init_xavier)

    def sample_langevin(self, z, x, Gnet, args, verbose=False):
        mse = nn.MSELoss(reduction='sum')
        z = z.clone().detach()
        z.requires_grad = True
        for i in range(args.l_steps):
            x_hat = Gnet(z)
            log_lkhd = 1.0 / (2.0 * args.sigma * args.sigma) * mse(x_hat, x)
            z_grad = torch.autograd.grad(log_lkhd, z)[0]

            z.data = z.data - args.gammar * 0.15 * 0.5 * args.l_step_size * args.l_step_size * (z_grad  + 1.0 / (args.prior_sigma * args.prior_sigma) * z.data)
            z.data +=  args.l_step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == args.l_steps - 1) and verbose:
                print('Langevin posterior {:3d}/{:3d}: MSE={:8.3f}'.format(i+1, args.l_steps, log_lkhd.item()))

            z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm
    def sample_langevin_multichain(self, z, x, Gnet, args, verbose=False):
        mse = nn.MSELoss(reduction='sum')
        nchain = 3
        zmix = 0
        z_grad_norm_mix=0
        for ic in range(nchain):
            z1 = z.clone().detach()
            z1.requires_grad = True
            for i in range(args.l_steps):
                x_hat = Gnet(z)
                log_lkhd = 1.0 / (2.0 * args.sigma * args.sigma) * mse(x_hat, x)
                z_grad = torch.autograd.grad(log_lkhd, z1)[0]

                z1.data = z1.data - 0.2 * 0.5 * args.l_step_size * args.l_step_size * (z_grad  + 1.0 / (args.prior_sigma * args.prior_sigma) * z1.data)
                z1.data += args.l_step_size * torch.randn_like(z1).data

                if (i % 5 == 0 or i == args.l_steps - 1) and verbose:
                    print('Langevin posterior chain{:2d}/{:3d}/{:3d}: MSE={:8.3f}'.format(ic+1,i+1, args.l_steps, log_lkhd.item()))

                z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()
            zmix = z1 + zmix
            z_grad_norm_mix = z_grad_norm_mix + z_grad_norm
        return zmix.detach()/nchain, z_grad_norm_mix/nchain
    def forward(self, z, x=None):
        return self.sample_langevin(z, x, self.Gnet, self.args)[0]



def train(args, output_dir,device,logger):
    dataset = datasets.ImageFolder(root=args.datapath,
                                  transform=transforms.Compose([
                                      transforms.Resize([args.img_size,args.img_size]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
    Nsamples = len(dataset)
    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4,
                                               drop_last=True ,**kwargs)

    model = Model(args).to(device)
    def eval_flag():
        model.eval()

    def train_flag():
        model.train()

    mse = nn.MSELoss(reduction='sum')
    optG = torch.optim.Adam(model.Gnet.parameters(), lr=args.lr,betas=(args.beta1, args.beta2))
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optG, args.gamma)
    epoch_ckpt = 0
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location='cuda:{}'.format(args.device))
        model.Gnet.load_state_dict(ckpt['Gnet'])
        optG.load_state_dict(ckpt['opt'])
        #epoch_ckpt = 100

    def sample_p_0(n=args.batch_size):
        return args.prior_sigma * torch.randn(*[n, args.nz]).to(device)

    def sample_p(n=args.batch_size, sig=args.prior_sigma,mu=0):
        return (mu.clone().detach() + torch.multiply(torch.tensor(sig),torch.randn(*[n, args.nz]))).to(device)

    ################################################
    ## train
    train_flag()
    stats = {
        'loss_g': [],
    }
    interval = []
    #z_warm = Variable(sample_p_0(n=Nsamples))
    z_cold = Variable(sample_p_0(n=args.batch_size))
    for epoch in range(epoch_ckpt, args.n_epochs):
        train_flag()
        loss_epoch = 0
        zstd = torch.zeros(args.nz)
        zmu = torch.zeros(args.nz)
        nbatch = 0
        for i, (x,y) in enumerate(train_loader, 0):
            nbatch = nbatch+1
            x = x.to(device)
            batch_size = x.shape[0]
            #Here: Nsamples / batch_size = int, since drop_last=True
            #z_g_k = model(z_warm[i*batch_size:(i+1)*batch_size], x)
            ##z_g_k = model(z_cold, x)
            z_g_k = model(sample_p_0(n=args.batch_size), x)
            optG.zero_grad()
            x_hat = model.Gnet(z_g_k.detach())
            loss_g = mse(x_hat, x) / batch_size / (2.0 * args.sigma * args.sigma)
            loss_g.backward()
            optG.step()
            # keep z with warm-start
            #z_warm[i * batch_size:(i + 1) * batch_size] = z_g_k

            # Printout
            if i % args.n_printout == 0:
                with torch.no_grad():
                    #posterior_moments = '[zmean={:8.2f}, zstd={:8.2f}, |z|max={:8.2f}]'.format(np.mean(z_g_k.cpu().numpy()),
                    #                                                                           np.mean(np.std(z_g_k.cpu().numpy(),0)),
                    #                                                         z_g_k.abs().max())
                    posterior_moments = '[zmean={:8.2f}, zstd={:8.2f}, |z|max={:8.2f}]'.format(
                        z_g_k.mean(),
                        z_g_k.std(dim=0).mean(),
                        z_g_k.abs().max())
                    logger.info(
                        'epoch{:5d}/total{:5d} batch{:5d}/total{:5d} '.format(epoch, args.n_epochs, i, len(train_loader)) +
                        'loss_g={:8.3f}, '.format(loss_g*(2.0 * args.sigma * args.sigma)) +
                        '|z_g_k|={:6.2f}, '.format(z_g_k.view(batch_size, -1).norm(dim=1).mean()) +
                        'posterior_moments={}, '.format(posterior_moments) +
                        'balance_sigma={:1.2f},'.format(args.sigma)
                    )
            with torch.no_grad():
                loss_epoch = loss_epoch + 1 / nbatch * (loss_g * (2.0 * args.sigma * args.sigma) - loss_epoch)
                zstd = zstd + 1/nbatch*(z_g_k.std(dim=0).cpu()-zstd)
                zmu = zmu + 1 / nbatch * (z_g_k.mean(dim=0).cpu() - zmu)
        lr_schedule.step(epoch=epoch)
        #eval_flag()
        logger.info(
            'epoch{:5d}/total{:5d} '.format(epoch, args.n_epochs) +
            'loss_g_epoch={:8.3f}, '.format(loss_epoch) +
            'zmean={:8.3f}, '.format(zmu.mean())+
            'zstd={:8.3f}, '.format(zstd.mean())
        )
        if 0 and (epoch % 50 == 0) and (epoch != 0):
            args.sigma = args.sigma * 1
            args.gammar = (args.gammar + 1e-1*torch.randn(1).to(device))

        #if  epoch > args.epbase:
        #    args.gammar = 1+1/2*torch.sin(torch.pi*torch.tensor((epoch-args.epbase)/args.epbase)).to(device)
            #args.gammar = args.gammar + 1e-1
        # Stats
        if epoch % args.n_stats == 0:
            stats['loss_g'].append(loss_epoch.item())
            interval.append(epoch + 1)
            plot_stats(output_dir, stats, interval)
        # Plot
        if epoch % args.n_plot == 0:
            #zstd=z_g_k.std(dim=0).cpu()
            #zmu = z_g_k.mean(dim=0).cpu()
            z_g_sr = sample_p(n=args.batch_size,sig=zstd,mu=zmu)
            z_g_srm = sample_p(n=args.batch_size, sig=zstd.mean(),mu=zmu)
            z_g_s = sample_p_0(n=args.batch_size)
            x_s = model.Gnet(z_g_s)
            x_sr = model.Gnet(z_g_sr)
            x_srm = model.Gnet(z_g_srm)
            ##interpolation
            zid1 = z_g_k[3];
            zid2 = z_g_k[17];
            # img_inter =  torch.zeros((11, args.img_channels, args.img_size, args.img_size))
            z_inter = torch.zeros(11, args.nz)
            for ii in range(11):
                z_inter[ii] = zid2 * (ii / 10) + zid1 * (1 - ii / 10)
            img_inter = model.Gnet(
                z_inter.to(device).float())
            path = '{}/samples/{:>06d}_interpolated.png'.format(output_dir, epoch)
            with torch.no_grad():
                save_images(x_s,
                            '{}/samples/{:>06d}_{:>06d}_sampling.png'.format(output_dir, epoch, i),
                            int(np.sqrt(args.batch_size)))
                save_images(x_sr,
                            '{}/samples/{:>06d}_{:>06d}_samplingrescale.png'.format(output_dir, epoch, i),
                            int(np.sqrt(args.batch_size)))
                save_images(x_srm,
                            '{}/samples/{:>06d}_{:>06d}_samplingrescalemean.png'.format(output_dir, epoch, i),
                            int(np.sqrt(args.batch_size)))
                save_images(x_hat,
                            '{}/samples/{:>06d}_{:>06d}_reconstruction.png'.format(output_dir, epoch, i),
                            int(np.sqrt(args.batch_size)))
                save_images(img_inter, path, 11)
                #save_interp(model,args,device,'{}/samples/{:>06d}_{:>06d}_interpolated.png'.format(output_dir, epoch, i),
                #            )

        # Ckpt
        if epoch > 0 and epoch % args.n_ckpt == 0:
            save_dict = {
                'epoch': epoch,
                'Gnet': model.Gnet.state_dict(),
                'opt': optG.state_dict(),
            }
            torch.save(save_dict, '{}/ckpt/ckpt_{:>06d}.pth'.format(output_dir, epoch))


def save_interp(model,args,device,path):
    n_interp = args.ninterp
    '''
    inter_image = np.zeros((n_interp ** 2,args.img_channels, args.img_size, args.img_size))
    inter_number = np.linspace(-2, 2, n_interp)
    height, width = np.meshgrid(inter_number, inter_number)
    z_inter = np.column_stack((height.reshape(-1, 1), width.reshape(-1, 1))).astype(np.float64)
    '''
    inter_image = torch.zeros((n_interp ** 2, args.img_channels, args.img_size, args.img_size))
    #inter_number = torch.linspace(-2, 2, n_interp)
    #inter_number = torch.linspace(-args.zstd, args.zstd, n_interp)
    inter_number1 = torch.linspace(-args.zstd[0], args.zstd[0], n_interp)
    inter_number2 = torch.linspace(-args.zstd[1], args.zstd[1], n_interp)
    height, width = torch.meshgrid(inter_number1, inter_number2)
    z_inter = torch.column_stack((height.reshape(-1, 1), width.reshape(-1, 1)))

    for i in range(n_interp):
        z_g_si = z_inter[(i * n_interp): ((i + 1) * n_interp)]
        #inter_image[i * n_interp:((i + 1) * n_interp)] =  model.Gnet(torch.from_numpy(z_g_si).to(device).float()).cpu().numpy()
        inter_image[i * n_interp:((i + 1) * n_interp)] = model.Gnet(
            z_g_si.to(device).float())
    save_images(inter_image,path,n_interp)


def main():
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)
    args = parse_args()
    args = to_named_dict(args)

    set_seed(args.seed)
    set_cuda(deterministic=args.gpu_deterministic)
    set_gpu(args.gpuid)
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:{}'.format(args.gpuid) if torch.cuda.is_available() else 'cpu')

    logger = setup_logging('main', output_dir, console=True)
    logger.info(args)
    train(args,output_dir,device,logger)


if __name__ == '__main__':
        main()