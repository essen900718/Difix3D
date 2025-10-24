import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from einops import rearrange, repeat

import torch
import torch.nn as nn

def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
    
    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_difix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_difix.unet.load_state_dict(_sd_unet)
        
    optimizer.load_state_dict(sd["optimizer"])
    
    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_difix.target_modules_vae
    sd["rank_vae"] = net_difix.lora_rank_vae
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net_difix.vae.state_dict().items() if "lora" in k or "skip" in k}
    
    if net_difix.use_cross_attention:
        sd["cross_attn"] = net_difix.cross_attn.state_dict() 
    
    sd["optimizer"] = optimizer.state_dict()   
    
    torch.save(sd, outf)

# # V1 - simple cross attention module
# class SimpleCrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5
        
#         self.to_q = nn.Linear(dim, dim, bias=False)
#         self.to_k = nn.Linear(dim, dim, bias=False)
#         self.to_v = nn.Linear(dim, dim, bias=False)
#         self.to_out = nn.Linear(dim, dim)
        
#     def forward(self, x, context):
#         """
#         x: input latent [B, C, H, W]
#         context: reference latent [B, C, H, W]
#         """
#         B, C, H, W = x.shape
        
#         # reshape to sequence format
#         x_seq = x.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
#         context_seq = context.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
        
#         # compute attention
#         q = self.to_q(x_seq)  # [B, H*W, C]
#         k = self.to_k(context_seq)  # [B, H*W, C]
#         v = self.to_v(context_seq)  # [B, H*W, C]
        
#         # multi-head attention
#         q = q.view(B, H*W, self.num_heads, C//self.num_heads).transpose(1, 2)  # [B, heads, H*W, C//heads]
#         k = k.view(B, H*W, self.num_heads, C//self.num_heads).transpose(1, 2)
#         v = v.view(B, H*W, self.num_heads, C//self.num_heads).transpose(1, 2)
        
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
        
#         out = attn @ v  # [B, heads, H*W, C//heads]
#         out = out.transpose(1, 2).contiguous().view(B, H*W, C)  # [B, H*W, C]
#         out = self.to_out(out)
        
#         # reshape back to spatial format
#         out = out.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
#         return out

# V2 - cross attention with higher dimension projection
class SimpleCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, hidden_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.scale = (self.hidden_dim // num_heads) ** -0.5
        
        self.input_proj = nn.Linear(dim, self.hidden_dim)
        self.context_proj = nn.Linear(dim, self.hidden_dim)
        
        self.to_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)  
        )
        
    def forward(self, x, context):
        """
        x: input latent [B, C, H, W]
        context: reference latent [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # reshape to sequence format
        x_seq = x.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
        context_seq = context.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
        
        # project to higher dimension
        x_proj = self.input_proj(x_seq)  # [B, H*W, hidden_dim]
        context_proj = self.context_proj(context_seq)  # [B, H*W, hidden_dim]
        
        # compute attention
        q = self.to_q(x_proj)  # [B, H*W, hidden_dim]
        k = self.to_k(context_proj)  # [B, H*W, hidden_dim]
        v = self.to_v(context_proj)  # [B, H*W, hidden_dim]
        
        # multi-head attention
        head_dim = self.hidden_dim // self.num_heads
        q = q.view(B, H*W, self.num_heads, head_dim).transpose(1, 2)  # [B, heads, H*W, head_dim]
        k = k.view(B, H*W, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, H*W, self.num_heads, head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v  # [B, heads, H*W, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, H*W, self.hidden_dim)  # [B, H*W, hidden_dim]
        out = self.to_out(out)  # project back to original space [B, H*W, C]
        
        # reshape back to spatial format
        out = out.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        return out

class Difix(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_vae=4, mv_unet=False, timestep=999, use_cross_attention=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        
        if mv_unet:
            from mv_unet import UNet2DConditionModel
            print("Using multi-view UNet")
        else:
            from diffusers import UNet2DConditionModel
            print("Using original UNet")

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # # V1 - simple cross attention
        # self.cross_attn = SimpleCrossAttention(dim=4, num_heads=1).cuda()

        # V2 - cross attention with higher dimension projection
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn = SimpleCrossAttention(dim=4, num_heads=8, hidden_dim=64)


        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            # 加载cross attention参数
            if "cross_attn" in sd:
                self.cross_attn.load_state_dict(sd["cross_attn"])

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            target_modules_vae = []

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        if self.use_cross_attention:
            self.cross_attn.to("cuda")

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        # print number of trainable parameters
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        if self.use_cross_attention:
            print(f"Number of trainable parameters in Cross Attention: {sum(p.numel() for p in self.cross_attn.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.use_cross_attention:
            self.cross_attn.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)
        if self.use_cross_attention:
            self.cross_attn.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        
        num_views = x.shape[1]
        batch_size = x.shape[0]

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
            caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
            
        x = x.permute(1, 0, 2, 3, 4)  # [B, V, C, H, W] -> [V, B, C, H, W]
        
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 
        
        # Cross attention fusion if we have multiple views
        if self.use_cross_attention and num_views > 1:

            z_main = z[:z.shape[0]//2]  # view[0] for input image
            z_ref = z[z.shape[0]//2:]   # view[1] for reference image
            
            # cross attention for fusion
            z_attended = self.cross_attn(z_main, z_ref)
            
            # residual connection
            z_main = z_main + z_attended
            
            # concatenate the main view twice to keep the batch size consistent
            # z = torch.cat([z_main, z_main], dim=0)
            z = torch.cat([z_main, z_ref], dim=0)
        
        # caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=batch_size)
    
        unet_input = z
        
        model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)
        # print("output_image.shape:", output_image.shape)
        return output_image

    # def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
    #     input_width, input_height = image.size
    #     new_width = image.width - image.width % 8
    #     new_height = image.height - image.height % 8
    #     image = image.resize((new_width, new_height), Image.LANCZOS)
        
    #     T = transforms.Compose([
    #         transforms.Resize((height, width), interpolation=Image.LANCZOS),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ])
    #     if ref_image is None:
    #         x = T(image).unsqueeze(0).unsqueeze(0).cuda()
    #     else:
    #         ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
    #         x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()
        
    #     output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]
    #     output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    #     output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        
    #     return output_pil
    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        input_width, input_height = images[0].size
        new_width = input_width - input_width % 8
        new_height = input_height - input_height % 8

        # transform 定義
        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # 處理所有輸入圖片
        tensors = []
        for img in images:
            img = img.resize((new_width, new_height), Image.LANCZOS)
            tensors.append(T(img))
        x = torch.stack(tensors, dim=0).unsqueeze(0).cuda()  # [1, N, 3, H, W]

        # reference image 處理
        if ref_image is not None:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            ref_tensor = T(ref_image).unsqueeze(0).unsqueeze(0).cuda()
            x = torch.cat([x, ref_tensor], dim=1)  # [1, N+1, 3, H, W]

        # # forward
        # output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]

        # # decode 第一張輸出圖
        # output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        # output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        # forward 輸出 (1, 2, 3, H, W)
        output_images = self.forward(x, timesteps, prompt, prompt_tokens)[0]  # -> (2, 3, H, W)

        output_pils = []
        for i in range(output_images.shape[0]):
            img = output_images[i].cpu() * 0.5 + 0.5  # [-1,1] → [0,1]
            img_pil = transforms.ToPILImage()(img)
            img_pil = img_pil.resize((input_width, input_height), Image.LANCZOS)
            output_pils.append(img_pil)

        # 例如 return 兩張
        return [output_pils[0], output_pils[1]]


    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        if self.use_cross_attention:
            sd["cross_attn"] = self.cross_attn.state_dict()
        
        sd["optimizer"] = optimizer.state_dict()
        
        torch.save(sd, outf)
