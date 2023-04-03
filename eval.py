
from basicsr.utils.img_util import tensor2img
import yaml
from basicsr.models.archs.DDT_arch import DDT
from basicsr.data.paired_image_dataset import Dataset_PairedImage
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from torchvision import transforms
import os
from torchvision.utils import make_grid



def eval_model(ckp_path, config_path, visualize=False):
    print("load ckp: " + str(ckp_path))
    ckp_name = ckp_path.split("/")[-1][0:-4]
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if visualize:
        if not os.path.exists('./visualization/' + ckp_name):
            os.mkdir('./visualization/' + ckp_name)
        print("start visualization.")

    model = DDT(inp_channels= cfg['network_g']['inp_channels'], 
                        out_channels=cfg['network_g']['out_channels'], 
                        dim = cfg['network_g']['dim'],
                        num_blocks = cfg['network_g']['num_blocks'], 
                        num_refinement_blocks = cfg['network_g']['num_refinement_blocks'],
                        heads = cfg['network_g']['heads'],
                        groups = cfg['network_g']['groups'],
                        ffn_expansion_factor = cfg['network_g']['ffn_expansion_factor'],
                        bias = cfg['network_g']['bias'],
                        LayerNorm_type = cfg['network_g']['LayerNorm_type']
                        )

    model.load_state_dict(torch.load(ckp_path, map_location='cpu')['params'])
    print("load checkpoint" + str(ckp_path))
    
    model = model.to(device)
    eval_dataset = Dataset_PairedImage(cfg['datasets']['val'])
    eval_dataloader = DataLoader(eval_dataset,
                                batch_size = 1,
                                shuffle = False)
    
    model.eval()

    with torch.no_grad():
        psnr_total = []
        ssim_total = []
        data_length = len(eval_dataset)
        for test_data in tqdm(eval_dataloader):
            lq_test = test_data['lq'].to(device)
            gt_test = test_data['gt'].to(device)
            output_test = model(lq_test)

            ssim = calculate_ssim(gt_test*255, output_test*255, crop_border=0, input_order='CHW', test_y_channel=True)
            ssim_total.append(ssim)

            psnr = calculate_psnr(gt_test, output_test, crop_border=0, input_order='CHW', test_y_channel=False)
            psnr_total.append(psnr)
            
            if visualize:
                lq_path = test_data['lq_path'][0]
                visualizion(output_test , ckp_name, lq_path)

        psnr_mean = sum(psnr_total)/data_length
        ssim_mean = sum(ssim_total)/data_length

        print("test_psnr: "+str(psnr_mean))
        print("test_ssim: "+str(ssim_mean))

def visualizion(output, ckp_name, lq_path):
    tensor2img = transforms.ToPILImage() 
    output = tensor2img(output.squeeze(0).cpu())
    lq_path = lq_path.split("/")[-1]
    output.save("".join(["./visualization/", str(ckp_name), "/", str(lq_path)]))


if __name__ == '__main__':
    ckp_path = "./Denoising/Pretrained/DDT_RealDenoising.pth"
    config_path = "./Denoising/Options/RealDenoising_DDT.yml"
    eval_model(ckp_path, config_path, visualize=False)

    
        