import torch


def elbo(self, latent_dist, output_dist, x, epoch):    ## here x is the batch of images
    
    """
        Here is the elbo loss calculated as described in the Paper(https://arxiv.org/abs/2003.05991) we can replace the construction 
        loss by the elbo loss to train the vae but as we observed straight forward construction loss 
        is giving much more better result than the complex elbo loss 
    """
    
    # Unpacking mean and log variance for latent and output distributions
    z, mu_z, log_var_z = latent_dist['x'], latent_dist['mu'], latent_dist['log_var']
    x, mu_x, log_var_x = output_dist['x'], output_dist['mu'], output_dist['log_var']
    
    half_ln2pi = (1/2) * torch.log(torch.tensor(2.0) * torch.pi)
    
    logqz = (-1) * half_ln2pi - (1/2) * log_var_z - ((z - mu_z)**2)/(2 * torch.exp(log_var_z))
    logpz = (-1) * half_ln2pi - ((z)**2)/(2)
    
    logpx_z = (-1) * half_ln2pi - (1/2) * log_var_x - ((x - mu_x)**2)/(2 * torch.exp(log_var_x))   ## log(p(x|z))
    
    ## putting all together
# Computing the expected values
    # Average log likelihood of reconstruction
    logpx_z_mean = torch.mean(logpx_z)
    # Average KL divergence
    kl_divergence = torch.mean(logqz - logpz)

    # ELBO: reconstruction term - KL divergence
    elbo = logpx_z_mean - kl_divergence

    return elbo

