# Latent Variable model(Variational Autoencoder)
Variational Autoencoder implementation

## Model Architecture
* Latent dim = 2
* encoder layers = 2
* decoder layers = 2

![](./assets/vae.png)

## Optimize
$P_{\theta}$ = Encoder, $q_{\phi}$ = Decoder, $P(z)$ = Gaussian Distribution   
$P_{\theta}(X) = \sum_zP_\theta (X,z) = \sum_z\frac{q_{\phi}(z|X)}{q_{\phi}(z|X)}P_\theta(X,z)=E_{z \sim q_{\phi}(z|X)}[\frac{P_\theta(X,z)}{q_{\phi}(z|X)}]$   
$\log(E_{z \sim q_{\phi}(z|X)}[\frac{P_\theta(X,z)}{q_{\phi}(z|X)}]) \ge E_{z \sim q_{\phi}(z|X)}[\log(\frac{P_\theta(X,z)}{q_{\phi}(z|X)})]$ (Jenson Inequality)   
$ELBO = E_{z \sim q_{\phi}(z|X)}[\log(\frac{P_\theta(X,z)}{q_{\phi}(z|X)})]$   
$E_{z \sim q_{\phi}(z|X)}[\log(\frac{P_\theta(X,z)}{q_{\phi}(z|X)})] = E_{z \sim q_{\phi}(z|X)}[\log P_{\theta}(X,z)] - E_{z \sim q_{\phi}(z|X)}[\log q_{\phi}(z|X)]$   
$\qquad \qquad \qquad \qquad \quad = E_{z \sim q_{\phi}(z|X)}[\log P_{\theta}(X,z)] - E_{z \sim q_{\phi}(z|X)}[P(z)]$   
$\qquad \qquad \qquad \qquad \qquad \qquad - E_{z \sim q_{\phi}(z|X)}[\log q_{\phi}(z|X)] + E_{z \sim q_{\phi}(z|X)}[P(z)]$   
$\qquad \qquad \qquad \qquad \quad = E_{z \sim q_{\phi}(z|X)}[\log P_{\theta}(X|z)] - D_{KL}(q_{\phi}(z|X)||P(z))$    