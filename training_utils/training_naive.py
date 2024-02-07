# Now you train the model
global_step = 0
for epoch in range(num_epochs):
    progress_bar = tqdm(total=len(train_dataloader))
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dataloader):
        images = batch['image']
        images = images.permute(0, 3, 1, 2)
        x_pr = torch.randn(images.shape).to(images.device)

        # create c_tar using clip
        labels_tr = batch['label_tr']
        tokens_tr = Tokenizer(labels_tr, padding=True, return_tensors="pt")
        c_tar = Encoder(**tokens_tr).last_hidden_state

        # create c_sou using clip
        labels_so = batch['label_so']
        tokens_so = Tokenizer(labels_so, padding=True, return_tensors="pt")
        c_sou = Encoder(**tokens_so).last_hidden_state

        # Sample noise to add to the images
        z = VAE.encode(images).latent_dist.sample()
        z_pr = VAE.encode(x_pr).latent_dist.sample()
        noise = torch.randn(z.shape).to(z.device)
        bs = z.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, Scheduler.num_train_timesteps, (bs,), device=z.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        z_t = Scheduler.add_noise(z, noise, timesteps)

        # Get the random noise z_pr_t
        z_pr_t = Scheduler.add_noise(z_pr, noise, timesteps)

        with torch.no_grad():
            # Predict the noise residual
            z_pr_sou = UNetLocked(z_pr_t, timesteps, c_sou)["sample"]

        # Predict the noise residual
        z_ada = UNetTrained(z_t, timesteps, c_tar)["sample"]
        z_pr_ada = UNetTrained(z_pr_t, timesteps, c_sou)["sample"]
        loss = Loss(z_ada, z, z_pr_sou, z_pr_ada, images)
        loss.backward()
        Optimizer.step()
        lr_scheduler.step()
        Optimizer.zero_grad()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[
            0], "step": global_step}
        progress_bar.set_postfix(**logs)
        global_step += 1
