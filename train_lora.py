# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm


from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
# import model_util
import prompt_util
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
# import debug_util
import config_util
from config_util import RootConfig
from utils import sample_utils
from samplers.ddim import DDIMSampler
import debug_util

# import wandb


def flush():
    torch.cuda.empty_cache()
    gc.collect()

def describe_model(unet):
    cnt = 0
    with open("output.txt", "w") as file:
        for name, module in unet.named_modules():
            for child_name, child_module in module.named_modules():
                cnt += 1
                print(f"Name: {child_module.__class__.__name__}", file=file)
    print(cnt)


def train(
    config: RootConfig,
    prompts: List[PromptSettings],
    device: int
):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    # print(metadata)
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    # if config.logging.use_wandb:
    #     wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)
    from omegaconf import OmegaConf
    config_LDVM = OmegaConf.load('text2video.yaml')
    unet, _, _ = sample_utils.load_model(config_LDVM, config.pretrained_model.name_or_path)
    
    unet.to(device, dtype=weight_dtype)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Total number of parameters: {total_params}")
    # unet.requires_grad_(False)
    # unet.eval()
    # describe_model(unet)
    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    total_params = 0
    for lora in network.unet_loras:
        total_params += sum(p.numel() for p in lora.parameters())
    print(f"Total number of parameters: {total_params}")
    
    
    # gradients = []
    # for lora in network.unet_loras:
    #     for param in lora.parameters():
    #         print(param)
    #         if param.grad is not None:
    #             print(param)
    #             gradients.append(param.grad.view(-1))  # Flatten gradients

    # debug_util.check_requires_grad(network)
    # debug_util.check_training_mode(network)
    optimizer_module = train_util.get_optimizer(config.train.optimizer) #AdamW
    optimizer_kwargs = {}
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    

    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler, # constant
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    # # print("Prompts")
    # # for settings in prompts:
    # #     print(settings)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    with torch.no_grad():
        for settings in prompts:
            # print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                # print(prompt)
                if isinstance(prompt, list):
                    if prompt == settings.positive:
                        key_setting = 'positive'
                    else:
                        key_setting = 'attributes'
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = sample_utils.get_conditions(prompt, unet, batch_size=1)
                else:
                    if cache[prompt] == None:
                        cache[prompt] = sample_utils.get_conditions(prompt, unet, batch_size=1)
            # # print(criteria,cache[settings.target],cache[settings.positive],cache[settings.unconditional],cache[settings.neutral])
            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )
    print(len(prompt_pairs))
    

    flush()
    
    pbar = tqdm(range(config.train.iterations)) # 1000 iterations
    # # print(len(pbar)) 
    for i in pbar:
        
        ddim_sampler = DDIMSampler(unet)      
        with torch.no_grad():
                  
            # noise_scheduler.set_timesteps(
            #     config.train.max_denoising_steps, device=device
            # )



            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]
            

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps, (1,)
            ).item()
    
            

            with network:
                # ちょっとデノイズされれたものが返る
                cond_embd = prompt_pair.target
                uncond_embd = prompt_pair.unconditional
                noise_shape = sample_utils.make_model_input_shape(unet, batch_size=1, T=16) # num_frames = 16 (default)
                dnoised_latents = sample_utils.sample_batch_custom(unet, noise_shape, cond_embd,
                                            sample_type="ddim",
                                            sampler=ddim_sampler,
                                            start_timestep=50,
                                            end_timestep=timesteps_to,
                                            ddim_steps=50, # default 50
                                            eta=1.0, # default 1.0
                                            unconditional_guidance_scale=15, # default 15 
                                            uc=uncond_embd,
                                            denoising_progress=False, # default False
                                            )
                

            ddim_sampler.make_schedule(ddim_num_steps=1000,ddim_eta=1.0,verbose=False)

            current_timestep = [int(timesteps_to * 1000 / config.train.max_denoising_steps)]
            current_timestep = torch.tensor(current_timestep, device=device)
            
            cond_embd = prompt_pair.positive
            uncond_embd = prompt_pair.unconditional
            positive_latents = ddim_sampler.p_sample_ddim_predict_noise(dnoised_latents,cond_embd,current_timestep,1000-current_timestep-1,unconditional_guidance_scale=15,unconditional_conditioning=uncond_embd)            
           
            cond_embd = prompt_pair.neutral
            uncond_embd = prompt_pair.unconditional
            neutral_latents = ddim_sampler.p_sample_ddim_predict_noise(dnoised_latents,cond_embd,current_timestep,1000-current_timestep-1,unconditional_guidance_scale=15,unconditional_conditioning=uncond_embd) 
            cond_embd = prompt_pair.unconditional
            uncond_embd = prompt_pair.unconditional
            unconditional_latents = ddim_sampler.p_sample_ddim_predict_noise(dnoised_latents,cond_embd,current_timestep,1000-current_timestep-1,unconditional_guidance_scale=15,unconditional_conditioning=uncond_embd)

            
            
    #         #########################
    #         if config.logging.verbose:
    #             print("positive_latents:", positive_latents[0, 0, :5, :5])
    #             print("neutral_latents:", neutral_latents[0, 0, :5, :5])
    #             print("unconditional_latents:", unconditional_latents[0, 0, :5, :5])

        with network:
            cond_embd = prompt_pair.target
            uncond_embd = prompt_pair.unconditional
            target_latents = ddim_sampler.p_sample_ddim_predict_noise(dnoised_latents,cond_embd,current_timestep,1000-current_timestep-1,unconditional_guidance_scale=15,unconditional_conditioning=uncond_embd)
            # for lora in network.unet_loras:
            #     for name, param in lora.named_parameters():
            #         grad = torch.autograd.grad(target_latents, param, retain_graph=True, allow_unused=True)[0]
            #         if grad is not None:
            #             print(f"Output depends on parameter: {name}")
            #         else:
            #             print(f"Output does NOT depend on parameter: {name}")
    #         target_latents = train_util.predict_noise(
    #             unet,
    #             noise_scheduler,
    #             current_timestep,
    #             denoised_latents,
    #             train_util.concat_embeddings(
    #                 prompt_pair.unconditional,
    #                 prompt_pair.target,
    #                 prompt_pair.batch_size,
    #             ),
    #             guidance_scale=1,
    #         ).to(device, dtype=weight_dtype)
            
    #         #########################

    #         if config.logging.verbose:
    #             print("target_latents:", target_latents[0, 0, :5, :5])
        positive_latents.to(device,dtype=weight_dtype)
        neutral_latents.to(device,dtype=weight_dtype)
        unconditional_latents.to(device,dtype=weight_dtype)
        target_latents.to(device,dtype=weight_dtype)

        # target_latents.requires_grad = True
        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            unconditional_latents=unconditional_latents,
        )
        # if not loss.requires_grad:
        #     loss.requires_grad = True
        print("loss calculated")
             
    #     # 1000倍しないとずっと0.000...になってしまって見た目的に面白くない
    #     pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
    #     if config.logging.use_wandb:
    #         wandb.log(
    #             {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
    #         )
        print(loss)    
        loss.backward()
        gradients = []
        for lora in network.unet_loras:
            for param in lora.parameters():
                with open('output.txt','a') as f:
                    print(param,file=f)
                    print(param.grad,file=f)
                    print('\n\n',file=f)
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))  # Flatten gradients
        # for param in network.prepare_optimizer_params():
        #         if param.grad is not None:
        #             gradients.append(param.grad.view(-1))  # Flatten gradients

        # Concatenate all gradients into a single tensor for easy plotting
        all_gradients = torch.cat(gradients).cpu().numpy()

        # Plotting the gradient histogram
        import matplotlib.pyplot as plt
        plt.hist(all_gradients, bins=30, edgecolor='black')
        plt.xlabel("Gradient values")
        plt.ylabel("Frequency")
        plt.title("Gradient Histogram")
        plt.savefig("gradient_histogram.png", format='png', dpi=300)
        plt.close()  # Close the plot if you don't need to display it
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            target_latents,
            # latents,
        )
        flush()
        # break

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.pt",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.pt",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]

    if args.prompts_file is not None:
        config.prompts_file = args.prompts_file
    if args.alpha is not None:
        config.network.alpha = args.alpha
    if args.rank is not None:
        config.network.rank = args.rank
    config.save.name += f'_alpha{config.network.alpha}'
    config.save.name += f'_rank{config.network.rank}'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    # print(config.prompts_file)
    # print(attributes)
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    # print(prompts)
    device = torch.device(f"cuda:{args.device}")
    print(device)
    train(config, prompts, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    parser.add_argument(
        "--prompts_file",
        required=False,
        help="Prompts file for training.",
        default=None
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=None,
        help="LoRA weight.",
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=None,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle (comma seperated string)",
    )
    
    # --attributes 'male, female'
    
    args = parser.parse_args()

    main(args)
