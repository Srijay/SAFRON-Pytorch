import argparse
import math
from collections import defaultdict
import json
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.utils as vutils
from data.data import DatasetLoader, custom_collate_fn
from model import SAFRONModel
from discriminators import Pix2PixDiscriminator
from generators import weights_init
from losses import get_gan_losses
from utils import *
import os

ROOT_DIR = os.path.expanduser(r'F:\Datasets\himico')

parser = argparse.ArgumentParser()

parser.add_argument('--train_mask_dir',
                    default=os.path.join(ROOT_DIR, 'train_A'))
parser.add_argument('--train_image_dir',
                    default=os.path.join(ROOT_DIR, 'train_B'))

parser.add_argument('--test_mask_dir', default=r'F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches\cycleGAN\himico_tumor\trainA')
parser.add_argument('--test_image_dir', default=r'F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches\cycleGAN\himico_tumor\trainA')


# Optimization hyperparameters
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_iterations', default=400000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=9000000, type=int)

# Dataset options
parser.add_argument('--num_train_samples', default=10, type=int)
parser.add_argument('--num_val_samples', default=10, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)

# Image Generator options
parser.add_argument('--generator', default='residual')  # pix2pix or residual
parser.add_argument('--generator_mode', default='safron')  # pix2pix or safron
parser.add_argument('--l1_pixel_image_loss_weight', default=100.0, type=float)  # 1.0
parser.add_argument('--normalization', default='instance')
parser.add_argument('--activation', default='leakyrelu-0.2')

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=1, type=float)  # 0.01
parser.add_argument('--gan_loss_type', default='gan')

# Image discriminator
parser.add_argument('--discriminator', default='patchgan')  # patchgan or standard

# Output options
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=1000, type=int)

# Experiment related parameters
parser.add_argument('--experimentname', default='ruqayya')
parser.add_argument('--output_dir', default=os.path.join('./output'))
parser.add_argument('--checkpoint_name', default='model.pt')

parser.add_argument('--checkpoint_path', default='./output/ruqayya/model/model.pt')
parser.add_argument('--restore_from_checkpoint', default=True, type=bool_flag)
parser.add_argument('--test_output_dir', default=os.path.join(r'F:\Datasets\himico\test\B-2032566_B5_HE\ruqayya_patches\cycleGAN\himico_tumor\output_residual_ruqayya'))

# If you want to test model, set mode to test
parser.add_argument('--mode', default='train', type=str)


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def build_dsets(args):

    if (args.mode == "train"):
        dset_kwargs = {
            'image_dir': args.train_image_dir,
            'mask_dir': args.train_mask_dir,
            'mode': args.mode
        }
    else:
        dset_kwargs = {
            'image_dir': args.test_image_dir,
            'mask_dir': args.test_mask_dir,
            'mode': args.mode
        }

    dset = DatasetLoader(**dset_kwargs)

    num_imgs = len(dset)
    print(args.mode + ' dataset has %d images' % (num_imgs))

    return dset


def build_loader(args):

    dset = build_dsets(args)

    collate_fn = custom_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }

    loader = DataLoader(dset, **loader_kwargs)

    return loader


def build_model(args):
    kwargs = {
        'normalization': args.normalization,
        'activation': args.activation,
        'generator': args.generator,
        'mode': args.mode
    }
    model = SAFRONModel(**kwargs)
    return model, kwargs


def build_img_discriminator(args):

    if (args.discriminator == 'patchgan'):
        discriminator = Pix2PixDiscriminator(in_channels=6)
    elif (args.discriminator == 'standard'):
        d_kwargs = {
            'arch': args.d_img_arch,
            'normalization': args.d_normalization,
            'activation': args.d_activation,
            'padding': args.d_padding,
        }
        discriminator = PatchDiscriminator(**d_kwargs)
    else:
        raise "Give proper name of discriminator"

    discriminator = discriminator.apply(weights_init)

    return discriminator


def check_model(args, t, loader, model, mode):
    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)

    if torch.cuda.is_available():
        float_dtype = torch.cuda.FloatTensor
        long_dtype = torch.cuda.LongTensor
    else:
        float_dtype = torch.FloatTensor
        long_dtype = torch.LongTensor

    num_samples = 0

    output_dir = os.path.join(experiment_output_dir, "training_output", mode)
    mkdir(output_dir)

    with torch.no_grad():
        for batch in loader:

            image_name, image_gt, mask_gt = batch

            if (len(image_gt) == 0):
                continue

            if torch.cuda.is_available():
                image_gt = image_gt.cuda()
                mask_gt = mask_gt.cuda()

            # Run the model as it has been run during training

            try:
                model_out = model(mask=mask_gt.float(), generator_mode=args.generator_mode)
            except Exception as e:
                print(e)
                continue

            image_pred = model_out

            num_samples += image_gt.size(0)
            if num_samples >= 10:
                break

            im_initial = image_name[0].split(".")[0]

            if (image_pred is not None):

                image_gt_path = os.path.join(output_dir, im_initial + "_gt_image.png")
                save_image(image_gt, image_gt_path)

                image_pred_path = os.path.join(output_dir, im_initial + "_pred_image.png")
                save_image(image_pred, image_pred_path)


def test_model(args, loader, model):
    if torch.cuda.is_available():
        float_dtype = torch.cuda.FloatTensor
        long_dtype = torch.cuda.LongTensor
    else:
        float_dtype = torch.FloatTensor
        long_dtype = torch.LongTensor

    pred_image_output_dir = os.path.join(args.test_output_dir, "images")
    gt_mask_output_dir = os.path.join(args.test_output_dir, "masks")

    mkdir(pred_image_output_dir)
    mkdir(gt_mask_output_dir)

    t = 1

    image_list = []

    with torch.no_grad():

        for batch in loader:

            image_name, image_gt, mask_gt = batch

            if torch.cuda.is_available():
                mask_gt = mask_gt.cuda()

            # Run the model as it has been run during training

            # try:
            image_pred = model(mask=mask_gt.float(), generator_mode=args.generator_mode)
            # except Exception as e:
            #     print(e)
            #     continue


            # Save the predicted image
            image_pred_path = os.path.join(pred_image_output_dir, image_name[0])
            save_image(image_pred, image_pred_path)

            mask_gt_path = os.path.join(gt_mask_output_dir, image_name[0])
            save_image(mask_gt, mask_gt_path)

            t += 1

            image_list.append(image_pred[0].detach().cpu())

    # image_grid = vutils.make_grid(image_list, padding=2, normalize=True)
    # # Plot the images
    # plt.plot(2, 2)
    # plt.axis("off")
    # plt.title("Generated Images")
    # plt.imshow(np.transpose(image_grid, (1, 2, 0)))
    # plt.show()


def calculate_model_losses(args, image_gt, image_pred):
    total_loss = torch.zeros(1).to(image_gt)
    losses = {}

    # Image L1 Loss
    l1_pixel_loss_images = F.l1_loss(image_pred, image_gt.float())
    total_loss = add_loss(total_loss, l1_pixel_loss_images, losses, 'L1_pixel_loss_images',
                          args.l1_pixel_image_loss_weight)

    return total_loss, losses


def main(args):
    torch.cuda.empty_cache()

    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)
    model_dir = os.path.join(experiment_output_dir, "model")

    if torch.cuda.is_available():
        float_dtype = torch.cuda.FloatTensor
    else:
        float_dtype = torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.mode == "train"):

        mkdir(experiment_output_dir)
        mkdir(model_dir)

        with open(os.path.join(experiment_output_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    loader = build_loader(args)

    model, model_kwargs = build_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5,0.999))

    # Image Discriminator
    image_discriminator = build_img_discriminator(args)
    if image_discriminator is not None:
        image_discriminator.cuda()
        image_discriminator.type(float_dtype)
        image_discriminator.train()
        optimizer_d_image = torch.optim.Adam(image_discriminator.parameters(),
                                             lr=args.learning_rate, betas=(0.5,0.999))

    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

    if args.restore_from_checkpoint or args.mode == "test":

        print("Restoring")
        restore_path = args.checkpoint_path

        if (device == "cpu"):
            checkpoint = torch.load(restore_path, map_location="cpu")
        else:
            checkpoint = torch.load(restore_path, map_location="cpu")  # to avoid memory surge

        model.load_state_dict(checkpoint['model_state'])

        if (args.mode == "train"):
            # optimizer.load_state_dict(checkpoint['optim_state']) #strict argument is not supported here

            if image_discriminator is not None:
                image_discriminator.load_state_dict(checkpoint['d_image_state'])
                optimizer_d_image.load_state_dict(checkpoint['d_image_optim_state'])
                image_discriminator.cuda()

        if (args.mode == "test"):
            model.eval()
            test_model(args, loader, model)
            print("Testing has been done and results are saved")
            return

        t = 0
        if 0 <= args.eval_mode_after <= t:
            model.eval()
        else:
            model.train()

        epoch = checkpoint['counters']['epoch']

        print("Starting Epoch : ", epoch)

    else:

        starting_epoch = 0

        if (args.mode == "test"):
            raise Exception("Give proper restoring model path")

        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'model_kwargs': model_kwargs,
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None, 'optim_state': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'd_mask_state': None, 'best_t': [],
        }

    # Loss Curves
    training_loss_out_dir = os.path.join(experiment_output_dir, 'training_loss_graph')
    mkdir(training_loss_out_dir)

    def draw_curve(epoch_list, loss_list, loss_name):
        plt.clf()
        plt.plot(epoch_list, loss_list, 'bo-', label=loss_name)
        plt.legend()
        plt.savefig(os.path.join(training_loss_out_dir, loss_name + '.png'))

    epoch_list = []
    monitor_epoch_losses = defaultdict(list)

    while True:

        if t >= args.num_iterations:
            break

        for batch in loader:

            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

            image_name, image_gt, mask_gt = batch

            if (len(image_gt) == 0):
                continue

            if torch.cuda.is_available():
                image_gt = image_gt.cuda()
                mask_gt = mask_gt.cuda()

            with timeit('forward', args.timing):
                # try:
                model_out = model(mask=mask_gt.float(), generator_mode=args.generator_mode)

                image_pred = model_out

            if (image_pred is None):
                continue

            image_pred = image_pred.cuda()

            total_loss, losses = calculate_model_losses(args, image_gt, image_pred)

            if image_discriminator is not None:
                scores_image_fake = image_discriminator(mask_gt.float(), image_pred)
                scores_image_fake = scores_image_fake.cuda()
                weight = args.discriminator_loss_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_image_fake), losses, 'g_gan_image_loss', weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                try:
                    total_loss.backward()
                except Exception as e:
                    print("Memory OOM : Iter number ", t, " image name ", image_name)
                    continue
            optimizer.step()

            image_fake = image_pred.detach()
            image_real = image_gt.detach()

            if image_discriminator is not None:
                d_image_losses = LossManager()  # For image
                scores_fake = image_discriminator(mask_gt.float(), image_fake)
                scores_real = image_discriminator(mask_gt.float(), image_real.float())
                d_image_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_image_losses.add_loss(d_image_gan_loss, 'd_image_gan_loss')
                optimizer_d_image.zero_grad()
                d_image_losses.total_loss.backward()
                optimizer_d_image.step()
                image_discriminator.cuda()

            t += 1

            if t % args.print_every == 0:

                print('t = %d / %d' % (t, args.num_iterations))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))

                if image_discriminator is not None:
                    for name, val in d_image_losses.items():
                        print(' D_img [%s]: %.4f' % (name, val))

            if t % args.checkpoint_every == 0:

                print('checking on train')
                check_model(args, t, loader, model, "train")

                checkpoint['model_state'] = model.state_dict()

                if image_discriminator is not None:
                    checkpoint['d_image_state'] = image_discriminator.state_dict()
                    checkpoint['d_image_optim_state'] = optimizer_d_image.state_dict()

                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint_path = os.path.join(model_dir, args.checkpoint_name)
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

        # Plot the loss curves
        epoch += 1
        epoch_list.append(epoch)
        for k, v in losses.items():
            monitor_epoch_losses[k].append(v)
            draw_curve(epoch_list, monitor_epoch_losses[k], k)


if __name__ == '__main__':
    print("CONTROL")
    args = parser.parse_args()
    main(args)

