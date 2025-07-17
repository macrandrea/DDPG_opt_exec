#import pdb
import torch
import torch.nn as nn
import numpy as np
import random as rnd

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rnd.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Check for GPU and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Optionally, you can set the default device for tensors
torch.set_default_tensor_type('torch.FloatTensor' if device.type == 'cpu' else 'torch.cuda.FloatTensor')

class Environment():
    def __init__(self, 
                 S_0 = 100,
                 q_0 = 1000,
                 lambd = 0.0, #penal
                 kappa  = 0.25, #perm_imp
                 alpha = 0.05,
                 theta = 0, 
                 sigma  = 1,    #vola
                 mu     = 0.0,  #drift
                 dt     = 1,    #time step
                 T      = int(60*60),
                 delta  = 0,    #pmt non lineare
                 device ='cuda'):
        
        self.kappa = kappa
        self.alpha = alpha
        self.sigma = sigma
        self.theta = theta
        self.lambd = lambd
        self.S_0 = S_0
        self.q_0 = q_0
        self.device = device
        self.mu = mu
        self.dt = dt
        self.T = T
        self.N = int(self.T / self.dt) + 1

        self.delta = delta
        self.device = device

    def ABM(self, S, kappa, sigma, mu, dt, delta, v_t):

        # Simulate a single step of the ABM process
        return torch.abs(S + mu - v_t * kappa * dt + sigma * dt**0.5 * torch.randn(1, device=self.device))
        
    def abm_increment(self, S, mu, sigma, dt):
        """
        Calcola l'incremento di un moto browniano aritmetico (ABM).
        S: valore corrente
        mu: drift
        sigma: volatilità
        dt: intervallo temporale
        """
        dW = torch.sqrt(torch.tensor(dt, device=self.device)) * torch.randn(1, device=self.device)
        return mu * dt + sigma * dW
    
    def reset(self):
        """
        Resetta l'ambiente e restituisce lo stato iniziale.
        """
        S_0 = torch.tensor([self.S_0], device=self.device)
        I_0 = torch.tensor([self.q_0], device=self.device)
        t_0 = torch.tensor([0.0     ], device=self.device)

        return torch.cat((S_0.unsqueeze(1), I_0.unsqueeze(1), t_0.unsqueeze(1)), dim=1)
    
    def step(self, stato, v_t, done=False):
        """
        Esegue un passo dell'ambiente.
        S: stato corrente
        v_t: azione corrente
        """

        S = stato[:, 0]
        I = stato[:, 1]
        t = stato[:, 2]
        done = False

        # Calcola il nuovo stato usando l'ABM
        new_S = self.ABM(S, self.kappa, self.sigma, self.mu, self.dt, self.delta, v_t)

        new_I = I - v_t

        new_t = t + torch.ones_like(t, device=self.device)
        
        # Calcola la ricompensa (ad esempio, differenza tra il nuovo e il vecchio stato)
        reward = -(self.S_0 * self.q_0)/self.N + (S * v_t - self.alpha * v_t ** 2) - self.lambd/2 *  (I ** 2 *self.sigma ** 2) * (t - self.N)

        # Controlla se l'episodio è terminato
        if new_t >= self.N:
            done = True
            # Assicura che new_I non sia negativo e rimanga un tensore
            if new_I < 0:
                new_I = torch.tensor([0.0], device=self.device)

        # Rendi le dimensioni coerenti per la concatenazione (tutti 2D)
        new_state = torch.cat((new_S.unsqueeze(1), new_I.unsqueeze(1), new_t.unsqueeze(1)), dim=1)
        
        return new_state, reward, done, 0
    

    def normalize_state(self, state):
        """
        Normalizza lo stato per migliorare la stabilità dell'addestramento.
        """
        S = state[:, 0]
        I = state[:, 1]
        t = state[:, 2]

        # Normalizza S e I rispetto ai loro valori iniziali
        S_norm = (S - self.S_0) / self.S_0
        I_norm = (I - self.q_0) / self.q_0
        t_norm = t / self.N

        return torch.stack((S_norm, I_norm, t_norm), dim=1)