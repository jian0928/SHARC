import os
import torch
from diffusers import DDPMScheduler


class SynthesizerTrainer:
    def __init__(self, model=None, vae=None, loss_fn=None, optimizer=None, device=None, iterations=None, gradient_accumulation_steps=None, log_interval=None, checkpoint_interval=None, output_dir=None):
        self.model = model
        self.vae = vae
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.iterations = iterations
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        self.scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')

    def _encode(self, images):
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def _add_noise(self, latents, noise, timesteps):
        return self.scheduler.add_noise(latents, noise, timesteps)

    def train_step(self, batch):
        x_global = self._encode(batch['target_color'])
        x_local = self._encode(batch['face_crop'])
        condition_global = self.model.conditioning.build_condition(image=batch['image'], smplx_normal=batch['smplx_normal'], smplx_silhouette=batch['smplx_silhouette'])
        condition_local = self.model.conditioning.build_condition(image=batch['face_crop'])
        noise_global = torch.randn_like(x_global)
        noise_local = torch.randn_like(x_local)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x_global.shape[0],), device=self.device).long()
        x_t_global = self._add_noise(x_global, noise_global, timesteps)
        x_t_local = self._add_noise(x_local, noise_local, timesteps)
        self.model.set_face_mask(batch['face_mask'])
        self.model.alignment.enabled = False
        pred_global_plain, pred_local = self.model(
            x_t_global,
            x_t_local,
            timesteps,
            encoder_hidden_states=None,
            condition_latents_global=condition_global,
            condition_latents_local=condition_local,
        )
        self.model.alignment.enabled = True
        pred_global_cbfa, _ = self.model(
            x_t_global,
            x_t_local,
            timesteps,
            encoder_hidden_states=None,
            condition_latents_global=condition_global,
            condition_latents_local=condition_local,
        )
        loss = self.loss_fn(pred_global_plain, pred_global_cbfa, pred_local, noise_global, noise_local)
        return loss

    def train(self, dataloader):
        self.model.train()
        iteration = 0
        while iteration < self.iterations:
            for batch in dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss = self.train_step(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                if (iteration + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                iteration += 1
                if self.log_interval is not None and iteration % self.log_interval == 0:
                    print('iteration', iteration, 'loss', loss.item())
                if self.checkpoint_interval is not None and iteration % self.checkpoint_interval == 0:
                    self.save_checkpoint(iteration)
                if iteration >= self.iterations:
                    break

    def save_checkpoint(self, iteration):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'ckpt_%d.pt' % iteration)
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'iteration': iteration}, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['iteration']
