import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))


import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, scale_invariant_loss
from gaussian_renderer import network_gui
from gaussian_renderer import render_imp, render_teacher
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from utils.pose_utils import gaussian_poses

import numpy as np
from lpipsPyTorch import lpips
from utils.compress_utils import save_comp, write_storage
from scene.gaussian_teacher import  TeaGaussianModel








def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(sh_degree=dataset.sh_degree,training_args=opt)
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)
    opt_dict = None
    if checkpoint:
        print("loading ckpt..........")
        (model_params, first_iter) = torch.load(checkpoint)
        (teacher_model_params, _) = torch.load(checkpoint)
        gaussians_tea = TeaGaussianModel(sh_degree=3)
        gaussians_tea.restore(teacher_model_params)
        opt_dict = gaussians.restore(model_params, opt)
        opt_dict = gaussians.filter_optimizer_state(opt_dict)

        gaussians.onedownSHdegree()

        gaussians.init_vnn(opt)
        gaussians.training_setup(opt)
        gaussians.optimizer.load_state_dict(opt_dict)
    print("current Gaussian number:", len(gaussians._xyz))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')


    low_opacity_mask = torch.zeros((len(gaussians._xyz), 1), device='cuda')
    low_opacity_scale_mask = torch.zeros((len(gaussians._xyz), 1), device='cuda')

    prune_number = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()


        gaussians.update_learning_rate(iteration)

        total_loss = 0

        opacity=None
        for _ in range(pipe.mv):
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            if iteration == args.svq_itr:
                gaussians.apply_svq(args)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
            opacity = render_pkg['opacity']

            render_pkg_teacher = render_teacher(viewpoint_cam, gaussians_tea, pipe, background)
            depth_teacher = render_pkg_teacher["render_depth"]
            img_teacher = render_pkg_teacher["render"]


            # Loss
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


            ## distill loss
            Ll1_depth = scale_invariant_loss(render_pkg["render_depth"], depth_teacher, mask=None)
            Ll1_distill = l1_loss(image, img_teacher)
            loss = loss + opt.lambda_distill * Ll1_distill + opt.lambda_depth * Ll1_depth
            loss = loss + opt.lambda_depth * Ll1_depth


            total_loss += loss

        total_loss.backward()
        iter_end.record()

        with torch.no_grad():

            ## calculate low opacity
            low_opacity = torch.quantile(opacity,  opt.pruning_quantile)

            low_opacity_gaussian_mask = opacity < low_opacity

            ## calculate low scale
            max_scales = torch.max(gaussians.get_scaling, dim=1)[0][:, None]
            low_scale = torch.quantile(max_scales, opt.pruning_quantile)

            low_scale_gaussian_mask = max_scales < low_scale

            low_opacity_scale_gaussian_mask = low_opacity_gaussian_mask * low_scale_gaussian_mask
            low_opacity_scale_mask += low_opacity_scale_gaussian_mask.float()



            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration == opt.iterations:
                save_dict = gaussians.encode(scene.model_path)
                save_comp(scene.model_path + "/comp.xz", save_dict)

                actual_storage = os.path.getsize(scene.model_path + "/comp.xz")
                with open(scene.model_path + "/storage.txt", 'w') as f:
                    byte = {'xyz': 0, 'scale':0,  'rotation':0, 'app':0, 'MLPs':0}
                    f.write(write_storage(save_dict, byte, gaussians.get_xyz.shape[0]))
                    f.write("Actual storage: " + str(round(actual_storage/2**20, 2)) + " MB")
                gaussians.decode(save_dict, decompress=False, path=scene.model_path)


            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render_imp, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % opt.pruning_interval == 0 and iteration < opt.pruning_iter:

                    pre_gaussian_count = len(gaussians._xyz)

                    prune_mask = low_opacity_scale_mask[:,0]
                    prune_mask = prune_mask > int(opt.pruning_interval*opt.vote)
                    gaussians.prune_points(prune_mask)

                    prune_number += torch.sum(prune_mask)
                    cur_gaussian_count = len(gaussians._xyz)
                    print("pruning %d gaussian points:" % (pre_gaussian_count - cur_gaussian_count),
                          " pruned points:", prune_number.item())
                    low_opacity_mask = torch.zeros((len(gaussians._xyz), 1), device='cuda')
                    low_opacity_scale_mask = torch.zeros((len(gaussians._xyz), 1), device='cuda')

            if iteration == args.net_itr:
                current_opt_dict = gaussians.optimizer.state_dict()

                gaussians.construct_net()
                gaussians.training_setup(opt)

                current_opt_dict = gaussians.filter_optimizer_state_net(current_opt_dict)
                gaussians.optimizer.load_state_dict(current_opt_dict)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                gaussians.opacity_nn_optimizer.step()
                gaussians.opacity_nn_optimizer.zero_grad(set_to_none=True)


                if iteration > args.net_itr:
                    gaussians.optimizer_net.step()
                    gaussians.optimizer_net.zero_grad(set_to_none=True)
                    gaussians.scheduler_net.step()
                if iteration >= args.svq_itr:
                    gaussians.optimizer_code.step()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    print("final Gaussian number:", len(gaussians._xyz))

    return 


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))                    


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")
                
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[47000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)




    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)


    # All done
    print("\nTraining complete.")
